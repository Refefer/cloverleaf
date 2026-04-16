//! Policy Evaluation and Biased Random Walk generation.
//!
//! Evaluates the graph given its existing edge weights (transitions) and a
//! reward function R(s) per node. Computes the expected value function V(s)
//! using the Bellman expectation equation:
//!
//!   V(s) = R(s) + gamma * sum_{s' in neighbors(s)} ( P(s'|s) * V(s') )
//!
//! The resulting value function is then converted into a stochastic policy
//! via softmax over neighbor values, producing new edge weights that bias
//! random walks toward higher-value states, acting as a correction to the
//! original weights.

use std::fmt::Write;

use crate::graph::{convert_edges_to_cdf, CDFtoP, Graph};
use crate::progress::CLProgressBar;

pub struct PolicyEvaluation {
    pub gamma: f32,
    pub iterations: usize,
    pub eps: f32,
    pub temperature: f32,
    pub indicator: bool,
}

impl PolicyEvaluation {
    pub fn new(gamma: f32, iterations: usize, eps: f32, temperature: f32, indicator: bool) -> Self {
        PolicyEvaluation {
            gamma,
            iterations,
            eps,
            temperature,
            indicator,
        }
    }

    /// Runs policy evaluation to convergence using the L-infinity norm.
    /// Returns the expected V(s) for every node in the graph under current weights.
    pub fn compute(&self, graph: &impl Graph, rewards: &[f32]) -> Vec<f32> {
        let n = graph.len();
        assert_eq!(rewards.len(), n);

        let mut v = rewards.to_vec();
        let mut next_v = vec![0f32; n];

        let pb = CLProgressBar::new(self.iterations as u64, self.indicator);
        let mut err = 0f32;

        for _ in 0..self.iterations {
            pb.update_message(|msg| {
                msg.clear();
                write!(msg, "PE Error: {:.6}", err).expect("Should never fail!");
            });

            for node_id in 0..n {
                let (edges, weights) = graph.get_edges(node_id);

                let expected_future_value = if edges.is_empty() {
                    0f32
                } else {
                    // SUM instead of MAX: we evaluate the existing stochastic policy
                    edges
                        .iter()
                        .zip(CDFtoP::new(weights))
                        .map(|(&edge, prob)| prob * v[edge])
                        .sum::<f32>()
                };

                next_v[node_id] = rewards[node_id] + self.gamma * expected_future_value;
            }

            err = next_v
                .iter()
                .zip(v.iter())
                .map(|(nv, ov)| (nv - ov).abs())
                .fold(0f32, f32::max);

            std::mem::swap(&mut v, &mut next_v);
            pb.inc(1);

            if err < self.eps {
                break;
            }
        }

        pb.finish();
        v
    }

    /// Extracts a stochastic policy from the value function as CDF edge weights.
    /// Blends the original transition probabilities with a softmax over neighbor values:
    ///   w(s, s') = P_orig(s'|s) * exp(V(s') / tau) / Z
    pub fn to_policy_weights(&self, graph: &impl Graph, values: &[f32]) -> Vec<f32> {
        let mut weights = vec![0f32; graph.edges()];

        for node_id in 0..graph.len() {
            // Note: We now capture the original weights instead of ignoring them with `_`
            let (edges, orig_cdf_weights) = graph.get_edges(node_id);
            let (start, stop) = graph.get_edge_range(node_id);

            if edges.is_empty() {
                continue;
            }

            // Convert original CDF weights back to probabilities to use as our base/prior
            let orig_probs: Vec<f32> = CDFtoP::new(orig_cdf_weights).collect();

            let max_v = edges
                .iter()
                .map(|&e| values[e])
                .fold(f32::NEG_INFINITY, f32::max);

            if self.temperature <= f32::EPSILON {
                // Greedy selection: distribute probability among max-value neighbors
                // proportionally to their original probabilities.
                let mut sum_orig_max = 0f32;
                for (i, &edge) in edges.iter().enumerate() {
                    if (values[edge] - max_v).abs() <= f32::EPSILON {
                        weights[start + i] = orig_probs[i];
                        sum_orig_max += orig_probs[i];
                    }
                }
                
                // Normalize among the greedy choices
                if sum_orig_max > 0.0 {
                    for w in &mut weights[start..stop] {
                        *w /= sum_orig_max;
                    }
                }
            } else {
                // Softmax selection: Multiply original probability by the exponential bias
                let mut sum_exp = 0f32;
                for (i, &edge) in edges.iter().enumerate() {
                    let exp_v = ((values[edge] - max_v) / self.temperature).exp();
                    let biased_prob = orig_probs[i] * exp_v;
                    
                    weights[start + i] = biased_prob;
                    sum_exp += biased_prob;
                }
                
                // Normalize the newly biased probabilities
                if sum_exp > 0f32 {
                    for w in &mut weights[start..stop] {
                        *w /= sum_exp;
                    }
                }
            }
        }

        // Convert the updated probability chunks back to CDFs
        for node_id in 0..graph.len() {
            let (start, stop) = graph.get_edge_range(node_id);
            if start < stop {
                convert_edges_to_cdf(&mut weights[start..stop]);
            }
        }

        weights
    }

}

#[cfg(test)]
mod policy_evaluation_tests {
    use super::*;
    use crate::graph::{CumCSR, CSR};

    fn build_linear_edges() -> Vec<(usize, usize, f32)> {
        vec![(0, 1, 1.), (0, 2, 1.), (1, 3, 1.), (2, 3, 1.)]
    }

    #[test]
    fn test_policy_evaluation_converges() {
        let csr = CSR::construct_from_edges(build_linear_edges(), false);
        let graph = CumCSR::convert(csr);

        let rewards = vec![0., 0., 0., 10.];
        let pe = PolicyEvaluation::new(0.9, 100, 1e-6, 1.0, false);
        let values = pe.compute(&graph, &rewards);

        assert!(values[3] > values[1]);
        assert!(values[3] > values[2]);
        assert!(values[1] > values[0]);
        assert!(values[2] > values[0]);
    }

    #[test]
    fn test_policy_evaluation_exact() {
        let csr = CSR::construct_from_edges(build_linear_edges(), false);
        let graph = CumCSR::convert(csr);

        let rewards = vec![0., 0., 0., 10.];
        let pe = PolicyEvaluation::new(0.9, 100, 1e-8, 1.0, false);
        let values = pe.compute(&graph, &rewards);

        let v3 = 10.0;
        let v1 = 0.9 * v3;
        let v2 = 0.9 * v3;
        let v0 = 0.9 * v1;

        assert!((values[3] - v3).abs() < 1e-4);
        assert!((values[1] - v1).abs() < 1e-4);
        assert!((values[2] - v2).abs() < 1e-4);
        assert!((values[0] - v0).abs() < 1e-4);
    }

    #[test]
    fn test_policy_weights_respects_values() {
        let csr = CSR::construct_from_edges(build_linear_edges(), false);
        let graph = CumCSR::convert(csr);

        let rewards = vec![0., 0., 0., 10.];
        let pe = PolicyEvaluation::new(0.9, 100, 1e-8, 0.1, false);
        let values = pe.compute(&graph, &rewards);
        let weights = pe.to_policy_weights(&graph, &values);

        // Node 0 has two neighbors: 1 and 2. Both lead to 3 with equal value,
        // so their policy weights should be equal.
        let (edges_0, _) = graph.get_edges(0);
        let probs_0: Vec<f32> = {
            let (start, stop) = graph.get_edge_range(0);
            CDFtoP::new(&weights[start..stop]).collect()
        };
        assert_eq!(edges_0.len(), 2);
        assert!((probs_0[0] - probs_0[1]).abs() < 1e-4);

        // Node 1 and 2 both point to 3 only, so weight should be 1.0
        let (start1, stop1) = graph.get_edge_range(1);
        let p1: Vec<f32> = CDFtoP::new(&weights[start1..stop1]).collect();
        assert!((p1[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_policy_weights_sharp_temperature() {
        let edges = vec![(0, 1, 1.), (0, 2, 1.)];
        let csr = CSR::construct_from_edges(edges, false);
        let graph = CumCSR::convert(csr);

        // Give node 2 a much higher value than node 1
        let values = vec![0., 1., 100.];
        let pe = PolicyEvaluation::new(0.9, 1, 1e-6, 0.01, false);
        let weights = pe.to_policy_weights(&graph, &values);

        let (start, stop) = graph.get_edge_range(0);
        let probs: Vec<f32> = CDFtoP::new(&weights[start..stop]).collect();
        // With very low temperature, edge to node 2 should dominate
        assert!(probs[1] > 0.99);
    }

    #[test]
    fn test_dead_end_node() {
        let edges = vec![(0, 1, 1.)];
        let csr = CSR::construct_from_edges(edges, false);
        let graph = CumCSR::convert(csr);

        let rewards = vec![0., 5.];
        let pe = PolicyEvaluation::new(0.9, 100, 1e-8, 1.0, false);
        let values = pe.compute(&graph, &rewards);

        assert!((values[1] - 5.0).abs() < 1e-4);
        assert!((values[0] - 4.5).abs() < 1e-4);
    }
}
