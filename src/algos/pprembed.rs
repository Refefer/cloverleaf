//! Uses page rank to construct a local neighborhood, then fuses features.
use std::fmt::Write;

use rayon::prelude::*;
use hashbrown::HashMap;

use crate::algos::rwr::RWR;
use crate::algos::utils::{Sample,FeatureHasher};
use crate::embeddings::{EmbeddingStore, NodeEmbeddings};
use crate::distance::Distance;
use crate::error::Error;
use crate::feature_store::{FeatureSet, FeatureStore};
use crate::graph::{CDFGraph,Graph};
use crate::named_graph::NamedGraph;
use crate::progress::CLProgressBar;

pub struct PPREmbed {

    /// Number of random walks per node to estimate neighborhood
    pub num_walks: usize,

    /// Restart criteria
    pub steps: Sample,

    /// Beta Parameter 
    pub beta: f32,

    /// Dimensions for FeatureHashing
    pub dims: usize,

    /// Minimum weight to combine
    pub eps: f32,

    /// Random seed
    pub seed: u64,
}

impl PPREmbed {
    /// Degree discount the Python bindings use when none is given.
    pub const DEFAULT_BETA: f32 = 0.8;

    /// Minimum weight the Python bindings use when none is given.
    pub const DEFAULT_EPS: f32 = 1e-5;

    /// `dims`-dimensional embeddings from `num_walks` walks per node.  `steps` >= 1 walks that
    /// many steps, 0 < `steps` < 1 stops after each step with that probability; anything else is
    /// an [`Error::InvalidArgument`].
    pub fn new(
        dims: usize,
        num_walks: usize,
        steps: f32,
        beta: f32,
        eps: f32,
        seed: u64
    ) -> Result<Self, Error> {
        let steps = Sample::steps(steps)
            .map_err(|_| Error::InvalidArgument("Steps must be between [0, inf)".into()))?;
        Ok(PPREmbed { num_walks, steps, beta, dims, eps, seed })
    }

    /// Embeds every node of `graph` from the features of its neighbourhood, with cosine
    /// distance.  `features` must index the graph's nodes, else an [`Error::VocabMismatch`];
    /// featureless nodes are given a feature of their own first.
    pub fn learn_features(
        &self,
        graph: &NamedGraph,
        features: &mut FeatureSet
    ) -> Result<NodeEmbeddings, Error> {
        features.prepare_for_learning(graph, None)?;
        NodeEmbeddings::new(graph.vocab().clone(), self.learn(graph.graph(), features.store()))
    }

    /// Learns the feature embeddings.
    pub fn learn<G: Graph + CDFGraph + Send + Sync>(
        &self, 
        graph: &G, 
        features: &FeatureStore
    ) -> EmbeddingStore {
        let embs = EmbeddingStore::new(graph.len(), self.dims, Distance::Cosine);
        let hasher = FeatureHasher::new(self.dims);
        let pb = CLProgressBar::new(graph.len() as u64, true);
        pb.update_message(|msg| write!(msg, "Embedding...").expect("Shouldn't fail"));
        (0..graph.len()).into_par_iter().for_each(|node_id| {
            let rwr = RWR {
                steps: self.steps,
                walks: self.num_walks,
                beta: self.beta,
                single_threaded: false,
                seed: self.seed + node_id as u64
            };

            let mut feat_maps = HashMap::new();
            rwr.sample_bfs(graph, node_id)
                .into_iter()
                .for_each(|(node_id, weight)| {
                    features.get_features(node_id).iter().for_each(|feat_id| {
                        let e = feat_maps.entry(*feat_id).or_insert(0f32);
                        *e += weight;
                    });
                });

            let emb = embs.get_embedding_mut_hogwild(node_id);
            let n = graph.len();
            feat_maps.into_iter()
                .filter(|(_,w)| *w > self.eps)
                .for_each(|(feat_id, weight)| {
                for hash_num in 0..3 {
                    let (sign, dim) = hasher.hash(feat_id, hash_num);
                    emb[dim] += sign as f32 * (weight * n as f32).ln().max(0f32);
                }
            });

            pb.inc(1);
        });
        pb.finish();

        embs
    }
}


#[cfg(test)]
mod pprembed_tests {
    use super::*;
    use crate::graph::EdgeType;
    use crate::named_graph::GraphBuilder;

    fn triangle(node_type: &str) -> NamedGraph {
        let mut gb = GraphBuilder::new();
        for (a, b) in [("a", "b"), ("b", "c"), ("c", "a")] {
            gb.add_edge((node_type, a), (node_type, b), 1., EdgeType::Undirected);
        }
        gb.build(true).unwrap()
    }

    #[test]
    fn test_new_validates_steps() {
        let ppr = PPREmbed::new(8, 10, 0.5, PPREmbed::DEFAULT_BETA, PPREmbed::DEFAULT_EPS, 1).unwrap();
        assert!(matches!(ppr.steps, Sample::Probability(p) if p == 0.5));
        assert!(matches!(PPREmbed::new(8, 10, 3., 0.8, 1e-5, 1).unwrap().steps, Sample::Fixed(3)));
        for bad in [0., -1., f32::NAN] {
            assert!(matches!(PPREmbed::new(8, 10, bad, 0.8, 1e-5, 1), Err(Error::InvalidArgument(_))));
        }
    }

    #[test]
    fn test_learn_features_fills_missing_nodes() {
        let g = triangle("n");
        let mut features = FeatureSet::from_graph(&g);
        let ppr = PPREmbed::new(8, 10, 2., 0.8, 1e-5, 1).unwrap();
        let ne = ppr.learn_features(&g, &mut features).unwrap();
        assert!(std::sync::Arc::ptr_eq(ne.vocab(), g.vocab()));
        assert_eq!((ne.len(), ne.dims()), (3, 8));
        // Every node got a feature of its own.
        assert_eq!(features.num_features(), 3);
    }

    /// Features over another graph's nodes are rejected rather than misindexed.
    #[test]
    fn test_learn_features_rejects_misaligned_features() {
        let g = triangle("n");
        let mut other = FeatureSet::from_graph(&triangle("m"));
        let ppr = PPREmbed::new(8, 10, 2., 0.8, 1e-5, 1).unwrap();
        assert!(matches!(ppr.learn_features(&g, &mut other), Err(Error::VocabMismatch(_))));
        assert_eq!(other.num_features(), 0, "features were changed before the check");
    }
}
