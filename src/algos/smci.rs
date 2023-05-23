/// Supervized Monte Carlo Iteration
use rayon::prelude::*;
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use hashbrown::HashSet;
use float_ord::FloatOrd;

use std::ops::Deref;

use crate::graph::*;
use crate::embeddings::{EmbeddingStore,Entity};
use crate::hogwild::{Hogwild};
use crate::sampler::{GreedySampler};
use crate::algos::rwr::{rollout,Steps};
use crate::vocab::TranslationTable;

pub struct SupervisedMCIteration {
    // number of full passes over the dataset
    pub iterations: usize,

    // number of walks per node, which we aggregate
    pub num_walks: usize,

    // how much should we interpolate between the original graph and the 
    // learned graph? alpha * orig + (1-alpha) * learned.
    pub alpha: f32,

    // gamma: how much we discount each step
    pub discount: f32,

    // How much we penalize each step
    pub step_penalty: f32,

    // ensure we're exploring at least P percentage of the time
    pub explore_pct: f32,

    // contrast the probability distribution
    pub compression: f32,

    // MCVI is a trajectory based optimizer; we stop each trajectory
    // with P probability after a step
    pub restart_prob: f32,

    // Random seed
    pub seed: u64
}

impl SupervisedMCIteration {
    pub fn learn(
        &self,
        graph: &(impl CDFGraph + Send + Sync),
        rewards: &[(NodeID, NodeID, f32)],
        distances: Option<(&EmbeddingStore, TranslationTable)>,
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
            let average_reward = rewards.par_iter().enumerate().map(|(i, (start_node, end_node, reward))| {
                // For each node, rollout num_walks times and compute the rewards
                let seed = (self.seed + (graph.len() * pass) as u64) + i as u64;
                let mut rng = XorShiftRng::seed_from_u64(seed);
                let mut trajectory = Vec::new();
                let mut seen = HashSet::new();
                let mut err = 0.0;
                for _ in 0..self.num_walks {
                    trajectory.clear();
                    
                    // Compute rollout
                    rollout(&t_graph, Steps::Probability(self.restart_prob), 
                            &sampler, *start_node, &mut rng, &mut trajectory);

                    // Scoring is a function of distance from the terminal node to the reward node.
                    // This allows us to extract value from every rollout regardless of whether it
                    // lands on the reward node
                    let terminal = trajectory[trajectory.len() - 1];
                    let dist = if terminal == *end_node {
                        Some(0f32)
                    } else if let Some((embs, trans_table)) = &distances {
                        match (trans_table[*end_node], trans_table[terminal]) {
                            (Some(en), Some(tn)) => {
                                let d = embs.compute_distance(&Entity::Node(*end_node), &Entity::Node(terminal));
                                Some(d)
                            }
                            _ => None
                        }
                        
                    } else {
                        None
                    };
                    
                    // Needs more love to figure out the right scaling function
                    let traj_len = trajectory.len() - 1;
                    let actual_reward = if let Some(d) = dist {
                        reward / (d + 1f32)
                    } else {
                        0.0
                    } + self.step_penalty * traj_len as f32; 

                    // Update the rewards for the graph
                    seen.clear();
                    for (i, node) in trajectory.iter().enumerate() {
                        if !seen.contains(node) {
                            seen.insert(*node);
                            let r = actual_reward * self.discount.powf((traj_len - i) as f32);
                            let mut agg = &mut h_v_state.get()[*node];
                            agg.0 += r;
                            agg.1 += 1;
                        }
                    }

                    err += actual_reward;
                }

                err / self.num_walks as f32
                
            }).sum::<f32>();

            println!("Average Reward: {}", average_reward / rewards.len() as f32);

            // Create new edges from V(S)
            let mut agg = h_v_state.deref();
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
                    
                    *wi = tn_vs - fn_vs;
                }


                softmax(weights);
                scale_weights(weights, self.compression);
            }
            h_v_state.get().iter_mut().for_each(|r| {
                r.0 = 0f32;
                r.1 = 1;
            });
            t_graph = OptCDFGraph::new(graph, new_edges);
        }

        let mut weights = t_graph.into_weights();
        interpolate_edges(self.alpha, graph, &mut weights);
        weights

    }
}

fn interpolate_edges(alpha: f32, g: &impl CDFGraph, weights: &mut [f32]) {
    let mut t = Vec::new();
    for node_id in 0..g.len() {
        let ow = g.get_edges(node_id).1;
        let (start, stop) = g.get_edge_range(node_id);
        let mut nw = &mut weights[start..stop];
        let owi = CDFtoP::new(ow);
        let nwi = CDFtoP::new(nw);
        t.clear();
        owi.zip(nwi).for_each(|(owi, nwi)| {
            let w = alpha * owi + (1f32 - alpha) * nwi;
            t.push(w);
        });
        nw.clone_from_slice(&t);
        convert_edges_to_cdf(&mut nw);
    }
}

fn softmax(weights: &mut [f32]) {
    let max = weights.iter()
        .max_by_key(|v| FloatOrd(**v))
        .map(|v| *v);

    if let Some(max) = max {
        weights.iter_mut().for_each(|v| *v = (*v - max).exp());
        let mut denom = weights.iter().sum::<f32>();
        weights.iter_mut().for_each(|v| *v /= denom);
    }
}

fn scale_weights(weights: &mut [f32], pow: f32) {
    weights.iter_mut().for_each(|wi| *wi = wi.powf(pow));
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
