/// Supervized Monte Carlo Iteration
use std::collections::{VecDeque,BinaryHeap};
use std::cmp::Reverse;
use std::sync::Mutex;

use rayon::prelude::*;
use rand::prelude::*;
use rand_xorshift::XorShiftRng;

use crate::graph::{Graph, NodeID, CDFGraph};
use crate::embeddings::{EmbeddingStore};
use crate::hogwild::{Hogwild};
use crate::sampler::{Weighted,Unweighted,Sampler};
use crate::algos::rwr::{RWR};

pub struct SupervisedMCIteration {
    iterations: usize,
    num_walks: usize,
    discount: f32,
    explore_pct: f32,
    restart_pro: f32,
    seed: u64
}

impl SupervisedMCIteration {
    pub fn learn(
        &self,
        graph: &(impl CDFGraph + Send + Sync),
        rewards: Vec<(NodeID, NodeID, f32)>,
        distances: EmbeddingStore
    ) -> () {

        let v_state = vec![(0., 1.); graph.len()];
        let h_v_state = Hogwild::new(v_state);

        let mut rng = XorShiftRng::seed_from_u64(self.seed);
        for pass in 0..self.iterations {
            
        }
    }
    
}

#[cfg(test)]
mod test_supervised_mc_iteration {
    use super::*;


}
