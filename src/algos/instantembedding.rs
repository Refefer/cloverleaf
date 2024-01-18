//! 
use rayon::prelude::*;

use crate::algos::utils::FeatureHasher;
use crate::algos::rwr::{Steps,RWR,ppr_estimate};
use crate::graph::{Graph as CGraph, CDFGraph};
use crate::embeddings::{EmbeddingStore,Distance};
use crate::progress::CLProgressBar;

#[derive(Clone,Copy, Debug)]
pub enum Estimator {
    RandomWalk {
        steps: Steps,
        walks: usize,
        beta: f32,
        seed: u64
    },
    SparsePPR {
        p: f32,
        eps: f32
    }
}

pub struct InstantEmbeddings {
    pub estimator: Estimator,
    pub dims: usize,
    pub hashes: usize
}

impl InstantEmbeddings {

    /// Learns the feature embeddings.
    pub fn learn<G: CGraph + CDFGraph + Send + Sync>(
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

