extern crate hashbrown;

use rayon::prelude::*;

use crate::graph::CDFGraph;
use crate::algos::pagerank::PageRank;
use crate::embeddings::EmbeddingStore;

static EPS: f64 = 1e-8;

pub struct LSR {
    pub passes: usize
}

impl LSR {
    
    pub fn compute(
        &self, 
        graph: &impl CDFGraph, 
        degrees: &EmbeddingStore,
        indicator: bool
    ) -> Vec<f64> {
        // Find the page rank of the tournament graph
        let page_rank = PageRank::new(self.passes, 1f64, EPS);
        let mut scores = page_rank.compute(graph, indicator);

        // Get the log norm of the scores
        let mean = scores.par_iter_mut().enumerate().map(|(node_id, s)| {
            *s = (*s / degrees.get_embedding(node_id)[0] as f64).ln();
            *s
        }).sum::<f64>() / scores.len() as f64;

        // Center the scores
        scores.par_iter_mut().for_each(|s| {
            *s -= mean;
        });

        scores
        
    }

}

