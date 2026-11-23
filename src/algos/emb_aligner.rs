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
/// `random_nodes` random ones.  `method` says how the anchors place it in the translated space.
///
/// The typical use is neighbourhood alignment for a node that has no edges yet: embed it from its
/// features in the original (pre-alignment) space, and map it into the aligned space.
#[derive(Clone,Debug)]
pub struct EmbeddingAligner {
    /// Nearest neighbours to anchor on.
    pub num_nodes: usize,
    /// Random nodes added as extra anchors.
    pub random_nodes: usize,
    pub method: AlignMethod
}

/// How the anchors place a mapped embedding.
#[derive(Clone,Copy,Debug,PartialEq)]
pub enum AlignMethod {
    /// Place it so its distances to the anchors in the translated space match its distances to
    /// them in the original space, by gradient descent from where it started.
    Distances {
        /// Learning rate.
        alpha: f32,
        /// Stop once the error changes by less than this fraction.
        error: f32,
        /// Maximum optimisation steps.
        max_iters: usize
    },

    /// Move it by the anchors' mean displacement, query + mean(translated - original): it moves
    /// as its neighbours did.  After neighbourhood alignment, where each node moved to
    /// w * itself + (1 - w) * its neighbours' mean, this is the alignment the query would have
    /// had if its anchors' neighbourhoods were its own, without needing it to have edges.
    MeanShift
}

impl AlignMethod {
    /// `Distances` with the defaults the Python bindings use.
    pub fn distances() -> Self {
        AlignMethod::Distances { alpha: 1e-1, error: 1e-2, max_iters: 10 }
    }
}

impl EmbeddingAligner {
    /// An aligner anchoring on `num_nodes` neighbours with the `Distances` method, with the
    /// defaults the Python bindings use.
    pub fn new(num_nodes: usize) -> Self {
        EmbeddingAligner { num_nodes, random_nodes: 0, method: AlignMethod::distances() }
    }

    /// Maps `query`, an embedding in `orig`'s space, into `translated`'s space.  `orig_ann` must
    /// index `orig`.  Anchors missing from `translated` are skipped; with none left the query is
    /// returned unchanged.
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

        // (original id, translated embedding, distance in the original space)
        let anchors: Vec<_> = neighbors.iter().filter_map(|nd| {
            let (orig_id, dist) = nd.to_tup();
            orig.vocab.get_name(*orig_id)
                .and_then(|(node_type, name)| translated.vocab.get_node_id(node_type.as_str(), name))
                .map(|t_id| (*orig_id, translated.embeddings.get_embedding(t_id), dist))
        }).collect();
        if anchors.is_empty() {
            return query.to_vec()
        }

        match self.method {
            AlignMethod::Distances { alpha, error, max_iters } => {
                let targets: Vec<_> = anchors.iter().map(|(_, t, d)| (*t, *d)).collect();
                align_embedding(query, &targets, alpha, error, max_iters)
            },
            AlignMethod::MeanShift => {
                let mut out = query.to_vec();
                let share = 1f32 / anchors.len() as f32;
                for (orig_id, t, _) in anchors.iter() {
                    let o = orig.embeddings.get_embedding(*orig_id);
                    out.iter_mut().zip(t.iter().zip(o.iter())).for_each(|(v, (ti, oi))| *v += share * (ti - oi));
                }
                out
            }
        }
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
        let aligner = EmbeddingAligner {
            method: AlignMethod::Distances { alpha: 1e-1, error: 0., max_iters: 50 }, ..EmbeddingAligner::new(3)
        };
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

    /// Mean shift moves the query by its anchors' mean displacement.
    #[test]
    fn test_mean_shift_moves_by_the_anchors_mean_displacement() {
        let orig = embeddings(&[("a", [0., 0.]), ("b", [1., 0.]), ("far", [10., 10.])]);
        let translated = embeddings(&[("a", [1., 1.]), ("b", [3., 1.]), ("far", [-50., -50.])]);
        let ann = fitted_ann(&orig);
        let aligner = EmbeddingAligner { method: AlignMethod::MeanShift, ..EmbeddingAligner::new(2) };
        // Anchors a and b moved by (1, 1) and (2, 1); "far" isn't among the 2 nearest.
        let out = aligner.align(&orig, &ann, &translated, &[0.5, 0.], 1);
        assert_eq!(out, vec![2.0, 1.0]);
    }

    /// With no anchor in the translated space, both methods leave the query where it is.
    #[test]
    fn test_no_anchors_leaves_the_query_unchanged() {
        let orig = embeddings(&[("a", [0., 0.]), ("b", [1., 0.])]);
        let translated = embeddings(&[("other", [5., 5.])]);
        let ann = fitted_ann(&orig);
        for method in [AlignMethod::distances(), AlignMethod::MeanShift] {
            let aligner = EmbeddingAligner { method, ..EmbeddingAligner::new(2) };
            assert_eq!(aligner.align(&orig, &ann, &translated, &[0.3, 0.4], 1), vec![0.3, 0.4]);
        }
    }
}
