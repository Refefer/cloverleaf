/// Supervized Monte Carlo Iteration
use rayon::prelude::*;
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use hashbrown::HashSet;

use std::ops::Deref;

use crate::graph::*;
use crate::embeddings::{EmbeddingStore,Entity};
use crate::hogwild::{Hogwild};
use crate::sampler::{GreedySampler};
use crate::algos::rwr::{rollout,Steps};

pub struct SupervisedMCIteration {
    // number of full passes over the dataset
    iterations: usize,

    // number of walks per node, which we aggregate
    num_walks: usize,

    // how much should we interpolate between the original graph and the 
    // learned graph? alpha * orig + (1-alpha) * learned.
    alpha: f32,

    // gamma: how much we discount each step
    discount: f32,

    // ensure we're exploring at least P percentage of the time
    explore_pct: f32,

    // MCVI is a trajectory based optimizer; we stop each trajectory
    // with P probability after a step
    restart_prob: f32,

    // Random seed
    seed: u64
}

impl SupervisedMCIteration {
    pub fn learn(
        &self,
        graph: &(impl CDFGraph + Send + Sync),
        rewards: &[(NodeID, NodeID, f32)],
        distances: EmbeddingStore
    ) -> Vec<f32> {

        // Track the current states, we use hogwild to allow for parallel
        // processing of paths
        // MCVI computes the average reward at a state, so track aggregate sums
        // and counts
        let h_v_state = Hogwild::new(vec![(0f32, 1usize); graph.len()]);

        // Store the updated versions of the graph
        let mut t_graph = OptCDFGraph::clone_from_cdf(graph);
        let sampler = GreedySampler(self.explore_pct);
        for pass in 0..self.iterations {
            rewards.par_iter().enumerate().for_each(|(i, (start_node, end_node, reward))| {
                // For each node, rollout num_walks times and compute the rewards
                let seed = (self.seed + (graph.len() * pass) as u64) + i as u64;
                let mut rng = XorShiftRng::seed_from_u64(seed);
                let mut trajectory = Vec::new();
                let mut seen = HashSet::new();
                for _ in 0..self.num_walks {
                    trajectory.clear();
                    
                    // Compute rollout
                    rollout(&t_graph, Steps::Probability(self.restart_prob), 
                            &sampler, *start_node, &mut rng, &mut trajectory);

                    // Scoring is a function of distance from the terminal node to the reward node.
                    // This allows us to extract value from every rollout regardless of whether it
                    // lands on the reward node
                    let terminal = trajectory[trajectory.len() - 1];
                    let dist = distances.compute_distance(&Entity::Node(*end_node), 
                                                          &Entity::Node(terminal));
                    
                    // Needs more love to figure out the right scaling function
                    let actual_reward = reward / (dist + 1f32);
                    let traj_len = trajectory.len() - 1;

                    // Update the rewards for the graph
                    seen.clear();
                    for (i, node) in trajectory.iter().enumerate() {
                        if !seen.contains(node) {
                            seen.insert(*node);
                            let r = actual_reward * self.discount.powf((traj_len - i) as f32);
                            let mut agg = h_v_state.get()[*node];
                            agg.0 += r;
                            agg.1 += 1;
                        }
                    }
                }
                
            });

            // Create new edges from V(S)
            let agg = h_v_state.deref();
            let mut new_edges = t_graph.into_weights();
            for node_id in 0..graph.len() {
                let edges = graph.get_edges(node_id).0;
                let (start, stop) = graph.get_edge_range(node_id);

                let weights = &mut new_edges[start..stop];

                let (r, c) = agg[node_id];
                let fn_vs = r / c as f32;
                for (wi, out_node) in weights.iter_mut().zip(edges.iter()) {
                    let (r, c) = agg[*out_node];
                    let tn_vs = r / c as f32;
                    
                    // If the node we're moving to is worse than the node we're on, set to zero
                    *wi = (tn_vs - fn_vs).max(0.);
                }
            }
            t_graph = OptCDFGraph::new(graph, new_edges);
        }

        let mut weights = t_graph.into_weights();
        interpolate_edges(self.alpha, graph, &mut weights);
        weights

    }
}

fn interpolate_edges(alpha: f32, g: &impl CDFGraph, weights: &mut [f32]) {
    for node_id in 0..g.len() {
        let ow = g.get_edges(node_id).1;
        let (start, stop) = g.get_edge_range(node_id);
        let mut nw = &mut weights[start..stop];
        normalize(&mut nw);
        CDFtoP::new(ow).zip(nw.iter_mut()).for_each(|(owi, nwi)| {
            let w = alpha * owi + (1f32 - alpha) * *nwi;
            *nwi = w;
        });

        convert_edges_to_cdf(&mut nw);
    }
}

fn normalize(w: &mut [f32]) {
    let mut s: f32 = w.iter().sum();
    if s == 0f32 { s = 1.; }
    w.iter_mut().for_each(|wi| *wi /= s);
}

#[cfg(test)]
mod test_supervised_mc_iteration {
    use super::*;
}
