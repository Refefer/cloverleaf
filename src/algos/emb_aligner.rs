use rand::prelude::*;
use rand_distr::Uniform;
use rand_xorshift::XorShiftRng;

use crate::autograd::*;
use crate::algos::ann::Ann;
use crate::algos::graph_ann::NodeDistance;
use crate::distance::Distance;
use crate::embeddings::NodeEmbeddings;

/// Maps an embedding from one embedding space into another, using nodes embedded in both as
/// anchors: the query's nearest neighbours in the original space, found with `Ann`, plus
/// `random_nodes` random ones.  The mapped embedding is placed in the translated space so its
/// distances to the anchors there match the query's distances to them in the original space.
///
/// The typical use is neighbourhood alignment for a node that has no edges yet: embed it from its
/// features in the original (pre-alignment) space, and map it into the aligned space.
#[derive(Clone,Debug)]
pub struct EmbeddingAligner {
    /// Nearest neighbours to anchor on.
    pub num_nodes: usize,
    /// Random nodes added as extra anchors.
    pub random_nodes: usize,
    /// Learning rate.
    pub alpha: f32,
    /// Stop once the error changes by less than this.
    pub error: f32,
    /// Maximum optimisation steps.
    pub max_iters: usize
}

impl EmbeddingAligner {
    /// An aligner anchoring on `num_nodes` neighbours, with the defaults the Python bindings use.
    pub fn new(num_nodes: usize) -> Self {
        EmbeddingAligner { num_nodes, random_nodes: 0, alpha: 1e-1, error: 1e-2, max_iters: 10 }
    }

    /// Maps `query`, an embedding in `orig`'s space, into `translated`'s space.  `orig_ann` must
    /// index `orig`.  Anchors missing from `translated` are skipped.
    pub fn align(
        &self,
        orig: &NodeEmbeddings,
        orig_ann: &Ann,
        translated: &NodeEmbeddings,
        query: &[f32],
        seed: u64
    ) -> Vec<f32> {
        let mut neighbors = orig_ann.predict(&orig.embeddings, query, self.num_nodes, None);
        if self.random_nodes > 0 && orig.len() > 0 {
            let mut rng = XorShiftRng::seed_from_u64(seed);
            let dist = Uniform::new(0, orig.len());
            for _ in 0..self.random_nodes {
                let node_id = dist.sample(&mut rng);
                let d = Distance::Euclidean.compute(query, orig.embeddings.get_embedding(node_id));
                neighbors.push(NodeDistance::new(d, node_id));
            }
        }

        let anchors: Vec<_> = neighbors.iter().filter_map(|nd| {
            let (orig_id, dist) = nd.to_tup();
            orig.vocab.get_name(*orig_id)
                .and_then(|(node_type, name)| translated.vocab.get_node_id(node_type.as_str(), name))
                .map(|t_id| (translated.embeddings.get_embedding(t_id), dist))
        }).collect();

        align_embedding(query, &anchors, self.alpha, self.error, self.max_iters)
    }
}

pub fn align_embedding(
    embedding: &[f32],
    t_embeddings: &[(&[f32], f32)],
    alpha: f32,
    eps: f32,
    max_epochs: usize
) -> Vec<f32> {
    use_shared_pool(false);

    let embs: Vec<_> = t_embeddings.iter().map(|(e, _d)| {
        Constant::new(e.to_vec())
    }).collect();

    let mut new_emb = embedding.to_vec();
    let mut last_err = std::f32::INFINITY;
    let mut i = 0;
    loop {
        // Put the embedding in a variable for graph
        let ne = Variable::pooled(new_emb.as_slice());

        // Compute the distances between the current embedding and the neighbor embeddings
        // we weight it based on position of the anchors
        let buff: Vec<_> = embs.iter().zip(t_embeddings.iter()).enumerate().map(|(pos, (e, (_, d)))| {
            // Clamped so the gradient stays finite when the embedding sits on an anchor.
            let euc_dist = (e.clone() - &ne).pow(2f32).sum().maximum(1e-12).sqrt();
            (euc_dist - *d).pow(2.) / ((pos + 1) as f32).sqrt()
        }).collect();

        let loss = buff.sum_all();

        let mut graph = Graph::new();
        graph.backward(&loss);

        // Simple Backpropagation
        let grad = graph.get_grad(&ne).unwrap();
        new_emb.iter_mut().zip(grad.iter()).for_each(|(ei, gi)| {
            *ei -= alpha * *gi;          
        });

        let cur_err = loss.value()[0];
        if (last_err - cur_err).abs() / last_err < eps {
            break
        }
        last_err = cur_err;
        i += 1;
        if i >= max_epochs { break }
    }
    new_emb
}

#[cfg(test)]
mod emb_aligner_tests {
    use super::*;
    use crate::embeddings::NodeEmbeddingsBuilder;

    fn embeddings(nodes: &[(&str, [f32; 2])]) -> NodeEmbeddings {
        let mut b = NodeEmbeddingsBuilder::new(Distance::Euclidean);
        for (name, e) in nodes {
            b.add_embedding("n", name, e).unwrap();
        }
        b.build().unwrap()
    }

    fn fitted_ann(e: &NodeEmbeddings) -> Ann {
        let mut ann = Ann::new();
        ann.fit(e.store(), 2, 10, None, None, None, 1);
        ann
    }

    /// A query on top of an anchor used to take a sqrt at zero, making every output NaN.
    #[test]
    fn test_align_embedding_is_finite_on_an_anchor() {
        let out = align_embedding(&[1., 0.], &[(&[1., 0.], 0.), (&[0., 1.], 1.)], 0.1, 1e-2, 10);
        assert!(out.iter().all(|v| v.is_finite()), "{:?}", out);
    }

    #[test]
    fn test_align_moves_toward_anchor_distances() {
        let orig = embeddings(&[("a", [0., 0.]), ("b", [1., 0.]), ("c", [0., 1.])]);
        let translated = embeddings(&[("a", [0., 0.]), ("b", [2., 0.]), ("c", [0., 2.])]);
        let ann = fitted_ann(&orig);
        let aligner = EmbeddingAligner { max_iters: 50, error: 0., ..EmbeddingAligner::new(3) };
        let out = aligner.align(&orig, &ann, &translated, &[0.5, 0.5], 1);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|v| v.is_finite()), "{:?}", out);
        // Distances to the anchors match better than they did before alignment.
        let loss = |q: &[f32]| ["a", "b", "c"].iter().map(|n| {
            let d_orig = Distance::Euclidean.compute(&[0.5, 0.5], orig.get_embedding("n", n).unwrap());
            let d_new = Distance::Euclidean.compute(q, translated.get_embedding("n", n).unwrap());
            (d_new - d_orig).powi(2)
        }).sum::<f32>();
        assert!(loss(&out) < loss(&[0.5, 0.5]), "{} >= {}", loss(&out), loss(&[0.5, 0.5]));
    }

    /// Anchors missing from the translated space are skipped; random anchors are seeded.
    #[test]
    fn test_align_skips_missing_anchors_and_is_seeded() {
        let orig = embeddings(&[("a", [0., 0.]), ("b", [1., 0.]), ("only_orig", [0., 1.])]);
        let translated = embeddings(&[("a", [0., 0.]), ("b", [1., 0.])]);
        let ann = fitted_ann(&orig);
        let aligner = EmbeddingAligner { random_nodes: 2, ..EmbeddingAligner::new(3) };
        let first = aligner.align(&orig, &ann, &translated, &[0.2, 0.3], 7);
        let second = aligner.align(&orig, &ann, &translated, &[0.2, 0.3], 7);
        assert!(first.iter().all(|v| v.is_finite()), "{:?}", first);
        assert_eq!(first, second);
    }
}
