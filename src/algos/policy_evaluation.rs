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

// Assumes your Graph and ModifiableGraph traits are available in crate::graph
use crate::graph::{Graph, ModifiableGraph}; 
use crate::progress::CLProgressBar;

/// Utility to ensure the graph's edge weights are a normalized CDF.
/// Run this ONCE on your graph before passing it to PolicyEvaluation::compute()
/// if your initial edge weights are raw frequencies, distances, or a PDF.
pub fn normalize_graph_to_cdf<G>(graph: &mut G)
where
    G: ModifiableGraph + Graph,
{
    eprintln!("normalizing graph to CDF...");
    for node_id in 0..graph.len() {
        let (edges, weights) = graph.modify_edges(node_id as _);
        
        if edges.is_empty() {
            continue;
        }

        // 1. Sum up all raw weights (treating current weights as a PDF/raw values)
        let total: f32 = weights.iter().sum();

        // 2. Convert to normalized CDF
        if total > 0.0 {
            let mut accum = 0.0;
            for w in weights.iter_mut() {
                accum += *w / total;
                *w = accum;
            }
            // Force the last element to exactly 1.0 to avoid floating point drift
            if let Some(last) = weights.last_mut() {
                *last = 1.0;
            }
        }
    }
}

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

        let max_reward = rewards.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_reward = rewards.iter().cloned().fold(f32::INFINITY, f32::min);
        let num_nonzero_rewards = rewards.iter().filter(|r| **r != 0.0).count();
        eprintln!("[PE] n={} gamma={} iters={} eps={} max_reward={:.4} min_reward={:.4} nonzero_rewards={}/{}", n, self.gamma, self.iterations, self.eps, max_reward, min_reward, num_nonzero_rewards, n);

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

            // Map-Reduce to find the maximum absolute error (L-infinity norm)
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
        where G: ModifiableGraph + Graph {

    eprintln!("updating graph weights in place...");
    
    // Threshold for when parallelization becomes worth the overhead.
    // You might need to tune this (e.g., 1024, 10_000, etc.)
    const PARALLEL_THRESHOLD: usize = 10_000; 

    for node_id in 0..graph.len() {
        let (edges, weights) = graph.modify_edges(node_id as _);
        
        // 1. In-place CDF to Probabilities (Sequential Backwards)
        // Kept sequential because dependencies make it hard to parallelize trivially
        for i in (1..weights.len()).rev() {
            weights[i] -= weights[i - 1];
        }

        if edges.len() >= PARALLEL_THRESHOLD {
            // === HEAVY NODE: USE RAYON ===
            let max_v = edges
                .par_iter()
                .map(|&e| values[e as usize])
                .reduce(|| f32::NEG_INFINITY, f32::max);

            let sum = if self.temperature <= f32::EPSILON {
                edges.par_iter().zip(weights.par_iter_mut()).map(|(&edge, w)| {
                    if (values[edge as usize] - max_v).abs() <= f32::EPSILON { *w } else { *w = 0.0; 0.0 }
                }).sum()
            } else {
                edges.par_iter().zip(weights.par_iter_mut()).map(|(&edge, w)| {
                    let exp_v = ((values[edge as usize] - max_v) / self.temperature).exp();
                    *w *= exp_v;
                    *w
                }).sum()
            };

            normalize_to_cdf(weights, sum);

        } else {
            // === TYPICAL NODE: FAST SEQUENTIAL (Auto-vectorized by LLVM) ===
            let mut max_v = f32::NEG_INFINITY;
            for &e in edges.iter() {
                max_v = f32::max(max_v, values[e as usize]);
            }

            let mut sum = 0.0;
            if self.temperature <= f32::EPSILON {
                for (&edge, w) in edges.iter().zip(weights.iter_mut()) {
                    if (values[edge as usize] - max_v).abs() <= f32::EPSILON {
                        sum += *w;
                    } else {
                        *w = 0.0;
                    }
                }
            } else {
                for (&edge, w) in edges.iter().zip(weights.iter_mut()) {
                    let exp_v = ((values[edge as usize] - max_v) / self.temperature).exp();
                    *w *= exp_v;
                    sum += *w;
                }
            }

            normalize_to_cdf(weights, sum);
        }
    }
}
}

// Helper function to keep the code DRY
#[inline]
fn normalize_to_cdf(weights: &mut [f32], sum: f32) {
    if sum > 0.0 {
        let mut accum = 0.0;
        for w in weights.iter_mut() {
            accum += *w / sum;
            *w = accum;
        }
        if let Some(last) = weights.last_mut() {
            *last = 1.0; 
        }
    }

}