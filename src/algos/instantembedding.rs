//! 
use rayon::prelude::*;

use crate::algos::utils::FeatureHasher;
use crate::algos::rwr::{Steps,RWR};
use crate::graph::{Graph as CGraph, CDFGraph};
use crate::embeddings::{EmbeddingStore,Distance};
use crate::progress::CLProgressBar;

pub struct InstantEmbeddings {
    pub dims: usize,
    pub hashes: usize,
    pub steps: Steps,
    pub walks: usize,
    pub beta: f32,
    pub seed: u64
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
            let rwr = RWR {
                steps: self.steps,
                walks: self.walks,
                beta: self.beta,
                seed: self.seed + node_id as u64
            };

            let ppr = rwr.sample_bfs(graph, node_id);
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
