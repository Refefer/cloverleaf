use rayon::prelude::*;

use std::fmt::Write;
use crate::graph::{CDFGraph, CDFtoP};
use crate::progress::CLProgressBar;

pub struct PageRank {
    damping: f64,
    iterations: usize,
    eps: f64
}

impl PageRank {

    pub fn new(iterations:usize, damping: f64, eps: f64) -> Self {
        PageRank {damping, iterations, eps}
    }

    pub fn compute(&self, graph: &impl CDFGraph, indicator: bool) -> Vec<f64> {
        let n = graph.len();
        let mut policy = vec![1. / n as f64; n];

        let mut next_policy = vec![0.; n];

        let pb = CLProgressBar::new(self.iterations as u64, indicator);
        let mut err = std::f64::INFINITY;
        for _iter in 0..self.iterations {
            pb.update_message(|msg| {
                msg.clear();
                write!(msg, "Error: {:.5}", err).expect("Should never fail!");
            });
            // Zero out new policy
            next_policy.par_iter_mut().for_each(|vi| *vi = 0.);

            let mut dead_end_weight = 0f64;
            for node_id in 0..n {

                let node_policy = policy[node_id];
                let (edges, weights) = graph.get_edges(node_id);
                
                // Uniformly teleport to all nodes if we're at a dead end
                if edges.len() == 0 {
                    dead_end_weight += node_policy / n as f64;
                } else {
                    for (edge, pr_k) in edges.iter().zip(CDFtoP::new(weights)) {
                        next_policy[*edge] += pr_k as f64 * node_policy;
                    }
                }
            }

            let s = next_policy.par_iter_mut().map(|pi| {
                // Dangling mass is followed like any other edge, so it is damped too
                *pi = (dead_end_weight + *pi) * self.damping + (1f64 - self.damping) / n as f64;
                *pi
            }).sum::<f64>();

            // Error is the difference between the original and next policy
            // Normalize the policy to 1
            err = next_policy.par_iter_mut().zip(policy.par_iter()).map(|(npi, pi)| {
                *npi = *npi / s;
                (*npi - *pi).powf(2.)
            }).sum::<f64>().sqrt();

            std::mem::swap(&mut policy, &mut next_policy);
            pb.inc(1);
            if err < self.eps { break }
        }
        pb.finish();

        policy
        
    }

}

#[cfg(test)]
mod pagerank_tests {
    use super::*;
    use crate::graph::{CSR, CumCSR};

    /// 0 -> 1, and 1 has no out edges.  Standard PageRank with d = 0.85 spreads the dangling
    /// node's mass uniformly *and* damps it: p0 = d * p1 / 2 + (1 - d) / 2, so p0 = 0.5 / 1.425.
    #[test]
    fn test_pagerank_damps_dangling_mass() {
        let graph = CumCSR::convert(CSR::construct_from_edges(vec![(0, 1, 1.)], false));
        let pr = PageRank::new(1000, 0.85, 1e-14).compute(&graph, false);
        let p0 = 0.5 / 1.425;
        assert!((pr[0] - p0).abs() < 1e-6, "{:?}, expected p0 = {}", pr, p0);
        assert!((pr[0] + pr[1] - 1.).abs() < 1e-9, "{:?}", pr);
    }
}
