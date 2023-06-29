use std::fmt::Write;

use rayon::prelude::*;

use crate::graph::{CDFGraph, NodeID, CDFtoP};
use crate::progress::CLProgressBar;

pub struct PageRank {
    damping: f32,
    iterations: usize,
    eps: f32
}

impl PageRank {

    pub fn new(iterations:usize, damping: f32, eps: f32) -> Self {
        PageRank {damping, iterations, eps}
    }

    pub fn compute(&self, graph: &impl CDFGraph) -> Vec<f32> {
        let n = graph.len();
        let mut policy = vec![1. / n as f32; n];

        let mut next_policy = vec![0.; n];

        let pb = CLProgressBar::new(self.iterations as u64, true);
        let mut err = std::f32::INFINITY;
        for iter in 0..self.iterations {
            pb.update_message(|msg| {
                msg.clear();
                write!(msg, "Error: {:.5}", err).expect("Should never fail!");
            });
            next_policy.par_iter_mut().for_each(|vi| *vi = 0.);

            for node_id in 0..n {
                let (edges, weights) = graph.get_edges(node_id);
                for (edge, pr_k) in edges.iter().zip(CDFtoP::new(weights)) {
                    next_policy[*edge] += pr_k * policy[node_id];
                }
            }

            let s = next_policy.par_iter_mut().map(|pi| {
                *pi = *pi * self.damping + 1f32 / n as f32;
                *pi
            }).sum::<f32>();

            err = next_policy.par_iter_mut().zip(policy.par_iter()).map(|(npi, pi)| {
                *npi /= s;
                (*npi - *pi).powf(2.)
            }).sum::<f32>().sqrt();

            std::mem::swap(&mut policy, &mut next_policy);
            pb.inc(1);
            if err < self.eps { break }
        }
        pb.finish();

        policy
        
    }

}
