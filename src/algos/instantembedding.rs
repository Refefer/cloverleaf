//! 
use rayon::prelude::*;

use crate::algos::rwr::{RWR,ppr_estimate};
use crate::algos::utils::{Sample,FeatureHasher};
use crate::graph::{Graph as CGraph, CDFGraph};
use crate::embeddings::{EmbeddingStore, NodeEmbeddings};
use crate::distance::Distance;
use crate::error::Error;
use crate::named_graph::NamedGraph;
use crate::progress::CLProgressBar;

/// How InstantEmbeddings estimates each node's personalized PageRank.  Build one with
/// [`random_walk`](Self::random_walk) or [`sparse_ppr`](Self::sparse_ppr), which check the
/// parameters.
#[derive(Clone,Copy, Debug)]
pub enum Estimator {
    RandomWalk {
        steps: Sample,
        walks: usize,
        beta: f32,
        seed: u64
    },
    SparsePPR {
        p: f32,
        eps: f32
    }
}

impl Estimator {
    /// Degree discount the Python bindings use when none is given.
    pub const DEFAULT_BETA: f32 = 0.8;

    /// Error tolerance the Python bindings use when none is given.
    pub const DEFAULT_EPS: f32 = 1e-5;

    /// `walks` random walks per node, discounted by degree to the power `beta`.  `steps` >= 1
    /// walks that many steps, 0 < `steps` < 1 stops after each step with that probability;
    /// anything else is an [`Error::InvalidArgument`].
    pub fn random_walk(walks: usize, steps: f32, beta: f32, seed: u64) -> Result<Self, Error> {
        let steps = Sample::steps(steps)
            .map_err(|_| Error::InvalidArgument("Steps must be between [0, inf)".into()))?;
        Ok(Estimator::RandomWalk { steps, walks, beta, seed })
    }

    /// The deterministic sparse estimate, restarting with probability `p`, which must be in
    /// (0, 1), with tolerance `eps`, which must be positive and finite: the checks
    /// [`SparsePpr::new`](crate::algos::rwr::SparsePpr::new) makes, else an
    /// [`Error::InvalidArgument`].
    pub fn sparse_ppr(p: f32, eps: f32) -> Result<Self, Error> {
        crate::algos::rwr::SparsePpr::new(p, eps)?;
        Ok(Estimator::SparsePPR { p, eps })
    }
}

#[derive(Clone,Copy,Debug)]
pub struct InstantEmbeddings {
    pub estimator: Estimator,
    pub dims: usize,
    pub hashes: usize
}

impl InstantEmbeddings {

    /// Embeds every node of `graph`, with cosine distance.
    pub fn learn(&self, graph: &NamedGraph) -> Result<NodeEmbeddings, Error> {
        NodeEmbeddings::new(graph.vocab().clone(), self.compute(graph.graph()))
    }

    /// Embeds every node, indexed by node id.
    pub fn compute<G: CGraph + CDFGraph + Send + Sync>(
        &self, 
        graph: &G
    ) -> EmbeddingStore {
        let n = graph.len();
        let es = EmbeddingStore::new(n, self.dims, Distance::Cosine);
        let fh = FeatureHasher::new(self.dims);
        let pb = CLProgressBar::new(n as u64, true);
        (0..graph.len()).into_par_iter().for_each(|node_id| {
            let ppr = match self.estimator {
                Estimator::RandomWalk {steps, walks, beta, seed} => {
                    let rwr = RWR {
                        steps: steps,
                        walks: walks,
                        beta: beta,
                        single_threaded: false,
                        seed: seed + node_id as u64
                    };

                    rwr.sample_bfs(graph, node_id)
                },
                Estimator::SparsePPR { p, eps } => ppr_estimate(graph, node_id, p, eps)
            };
            
            let embs = es.get_embedding_mut_hogwild(node_id);
            ppr.into_iter().for_each(|(node_id, weight)| {
                for hi in 0..self.hashes {
                    let (sign, dim) = fh.hash(node_id, hi);
                    embs[dim] += sign as f32 * (weight * n as f32).ln().max(0f32);
                }
            });
            pb.inc(1);

        });
        pb.finish();
        es
    }
}


#[cfg(test)]
mod instant_embedding_tests {
    use super::*;
    use crate::graph::EdgeType;
    use crate::named_graph::GraphBuilder;

    #[test]
    fn test_estimators_validate() {
        assert!(matches!(Estimator::random_walk(5, 3., Estimator::DEFAULT_BETA, 1),
                         Ok(Estimator::RandomWalk { steps: Sample::Fixed(3), walks: 5, .. })));
        assert!(matches!(Estimator::random_walk(5, 0.5, 0.8, 1),
                         Ok(Estimator::RandomWalk { steps: Sample::Probability(_), .. })));
        assert!(matches!(Estimator::sparse_ppr(0.5, Estimator::DEFAULT_EPS),
                         Ok(Estimator::SparsePPR { p, eps }) if p == 0.5 && eps == 1e-5));
        for bad in [0., -1., f32::NAN] {
            assert!(matches!(Estimator::random_walk(5, bad, 0.8, 1), Err(Error::InvalidArgument(_))));
        }
        for bad in [0., 1., -1., f32::NAN] {
            assert!(matches!(Estimator::sparse_ppr(bad, 1e-5), Err(Error::InvalidArgument(_))));
        }
    }

    #[test]
    fn test_learn_names_the_graphs_nodes() {
        let mut gb = GraphBuilder::new();
        for (a, b) in [("a", "b"), ("b", "c"), ("c", "a")] {
            gb.add_edge(("n", a), ("n", b), 1., EdgeType::Undirected);
        }
        let g = gb.build(true).unwrap();
        for estimator in [Estimator::random_walk(20, 2., 0.8, 1).unwrap(), Estimator::sparse_ppr(0.3, 1e-5).unwrap()] {
            let ie = InstantEmbeddings { estimator, dims: 8, hashes: 2 };
            let ne = ie.learn(&g).unwrap();
            assert!(std::sync::Arc::ptr_eq(ne.vocab(), g.vocab()));
            assert_eq!((ne.len(), ne.dims()), (3, 8));
            assert!(matches!(ne.distance(), Distance::Cosine));
        }
    }
}
