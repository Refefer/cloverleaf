extern crate hashbrown;

use rayon::prelude::*;

use crate::graph::CDFGraph;
use crate::algos::pagerank::PageRank;
use crate::embeddings::EmbeddingStore;

static EPS: f32 = 1e-8;

pub struct LSR {
    pub passes: usize
}

impl LSR {
    
    pub fn compute(
        &self, 
        graph: &impl CDFGraph, 
        degrees: &EmbeddingStore,
        indicator: bool
    ) -> Vec<f32> {
        // Find the page rank of the tournament graph
        let page_rank = PageRank::new(self.passes, 1f32, EPS);
        let mut scores = page_rank.compute(graph, indicator);

        // Get the log norm of the scores
        let mean = scores.par_iter_mut().enumerate().map(|(node_id, s)| {
            *s = (*s / degrees.get_embedding(node_id)[0]).ln();
            *s
        }).sum::<f32>() / scores.len() as f32;

        // Center the scores
        scores.par_iter_mut().for_each(|s| {
            *s -= mean;
        });

        scores
        
    }

}

