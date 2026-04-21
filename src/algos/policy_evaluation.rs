//! Policy Evaluation and Biased Random Walk generation.
//!
//! Evaluates the graph given its existing edge weights (transitions) and a
//! reward function R(s) per node. Computes the expected value function V(s)
//! using the Bellman expectation equation.
//!
//! The resulting value function is then converted into a stochastic policy
//! via softmax over neighbor values, producing new edge weights that bias
//! random walks toward higher-value states, acting as a correction to the
//! original weights.

use rayon::prelude::*;
use std::fmt::Write;

use crate::graph::{Graph, NodeID, ParallelModifiableGraph, convert_edges_to_cdf};
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
    pub fn compute(&self, graph: &(impl Graph + Sync), rewards: &[f32]) -> Vec<f32> {
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

            // Calculate expected future values for each node
            next_v.par_iter_mut().enumerate().for_each(|(node_id, nv)| {
                let (edges, weights) = graph.get_edges(node_id);

                let expected_future_value = if edges.is_empty() {
                    0.0
                } else if edges.len() == 1 {
                    // Base case for single-edge nodes
                    v[edges[0]] * weights[0]
                } else {
                    // Safety check to ensure we are actually working with a normalized CDF.
                    // This compiles away in release mode but saves hours of debugging in debug mode.
                    debug_assert!(
                        (weights.last().unwrap() - 1.0).abs() < 1e-4,
                        "Node {}'s weights are not a normalized CDF! Last weight: {}",
                        node_id,
                        weights.last().unwrap()
                    );

                    // Summation by parts: completely stateless and fully parallel over the edges
                    let sum: f32 = edges
                        .par_windows(2)
                        .zip(weights.par_iter())
                        .map(|(edge_window, &cdf_i)| {
                            let v_curr = v[edge_window[0] as usize];
                            let v_next = v[edge_window[1] as usize];
                            cdf_i * (v_curr - v_next)
                        })
                        .sum();

                    let last_val = v[*edges.last().unwrap() as usize];
                    let last_cdf = *weights.last().unwrap();

                    sum + (last_val * last_cdf)
                };

                *nv = rewards[node_id] + self.gamma * expected_future_value;
            });

            // Find the maximum absolute error (L-infinity norm)
            err = next_v
                .par_iter()
                .zip(v.par_iter())
                .map(|(nv, ov)| (nv - ov).abs())
                .reduce(|| 0f32, f32::max);

            std::mem::swap(&mut v, &mut next_v);
            pb.inc(1);

            if err < self.eps {
                break;
            }
        }

        pb.finish();
        v
    }

    /// Extracts a stochastic policy from the value function and updates the graph weights IN-PLACE.
    /// Blends the original transition probabilities with a softmax over neighbor values.
    pub fn update_policy_weights_in_place<G>(&self, graph: &mut G, values: &[f32])
        where G: ParallelModifiableGraph + Graph {

        let temperature = self.temperature;

        graph.par_iter_mut().for_each(|(edges, weights)| {
            update_one_node(edges, weights, values, temperature);
        });
    }
}

/// Per-node softmax weight update for [`PolicyEvaluation::update_policy_weights_in_place`].
///
/// Exposed at crate visibility so benchmarks can drive this without reconstructing
/// a full graph. Production code reaches this via `update_policy_weights_in_place`,
/// which calls it inside `graph.par_iter_mut()` — so nodes are already processed
/// in parallel, and the per-node body stays sequential to avoid nested-Rayon
/// contention.
///
/// Preconditions:
///   - `weights` is a normalized CDF on entry (last element ~= 1.0).
///   - `edges.len() == weights.len()`.
///   - All edge ids index into `values`.
///
/// On exit, `weights` is a normalized CDF over the softmax-reweighted probabilities.
pub fn update_one_node(
    edges: &mut [NodeID],
    weights: &mut [f32],
    values: &[f32],
    temperature: f32,
) {
    // 1. In-place CDF -> probabilities (sequential backwards pass).
    for i in (1..weights.len()).rev() {
        weights[i] = (weights[i] - weights[i - 1]).max(0.0);
    }

    let mut max_v = f32::NEG_INFINITY;
    for &e in edges.iter() {
        max_v = f32::max(max_v, values[e as usize]);
    }

    let mut sum = 0.0;
    if temperature <= f32::EPSILON {
        for (&edge, w) in edges.iter().zip(weights.iter_mut()) {
            if (values[edge as usize] - max_v).abs() <= f32::EPSILON {
                sum += *w;
            } else {
                *w = 0.0;
            }
        }
    } else {
        for (&edge, w) in edges.iter().zip(weights.iter_mut()) {
            let exp_v = ((values[edge as usize] - max_v) / temperature).exp();
            *w *= exp_v;
            sum += *w;
        }
    }

    // Back to CDF.
    convert_edges_to_cdf(weights, Some(sum));
}
