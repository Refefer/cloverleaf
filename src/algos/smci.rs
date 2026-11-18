/// Supervized Monte Carlo Iteration
/// We oscillate between running monte carlo trajectories, computing rewards, and updating the
/// polic with what we've learned from previous runs.  We can also provide embeddings to increase
/// the signal when the reward maps are sparse.
use rayon::prelude::*;
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use hashbrown::HashSet;
use float_ord::FloatOrd;

use std::ops::Deref;

use crate::algos::rwr::rollout;
use crate::algos::utils::Sample;
use crate::embeddings::{EmbeddingStore,Entity};
use crate::graph::*;
use crate::hogwild::{Hogwild};
use crate::sampler::{GreedySampler};
use crate::vocab::{TranslationTable, Vocab};
use crate::embeddings::NodeEmbeddings;
use crate::named_graph::NamedGraph;

/// Maps graph node ids to embedding ids, so SMCI can index the embeddings with graph nodes.
fn graph_to_embedding_table(graph_vocab: &Vocab, embedding_vocab: &Vocab) -> TranslationTable {
    // create_translation_table is indexed by the receiver's ids and yields ids in `to_vocab`.
    graph_vocab.create_translation_table(embedding_vocab)
}

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
    /// Learns new edge weights for a named graph from `(from, to, reward)` rewards, returning a
    /// copy of the graph with them.  `embeddings`, when given, are matched to the graph's nodes
    /// by name.
    pub fn learn_graph(
        &self,
        graph: &NamedGraph,
        rewards: &[(NodeID, NodeID, f32)],
        embeddings: Option<&NodeEmbeddings>
    ) -> NamedGraph {
        let distances = embeddings.map(|e| (&e.embeddings, graph_to_embedding_table(graph.vocab(), &e.vocab)));
        let weights = self.learn(graph.graph(), rewards, distances);
        let new_graph = graph.graph().clone_with_edges(weights).expect("one weight per edge");
        NamedGraph { graph: std::sync::Arc::new(new_graph), vocab: graph.vocab().clone() }
    }

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
                    rollout(&t_graph, Sample::Probability(self.restart_prob), 
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
                                let d = embs.compute_distance(&Entity::Node(en), &Entity::Node(tn));
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
                            let agg = &mut h_v_state.get()[*node];
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
                    
                    *wi = tn_vs - fn_vs;
                }


                softmax(weights);
                scale_weights(weights, self.compression);
            }
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
        convert_edges_to_cdf(&mut nw, None);
    }
}

fn softmax(weights: &mut [f32]) {
    let max = weights.iter()
        .max_by_key(|v| FloatOrd(**v))
        .map(|v| *v);

    if let Some(max) = max {
        weights.iter_mut().for_each(|v| *v = (*v - max).exp());
        let denom = weights.iter().sum::<f32>();
        weights.iter_mut().for_each(|v| *v /= denom);
    }
}

fn scale_weights(weights: &mut [f32], pow: f32) {
    weights.iter_mut().for_each(|wi| *wi = wi.powf(pow));
}

#[cfg(test)]
mod test_supervised_mc_iteration {
    use super::*;

    /// SMCI indexes the table with graph node ids and uses the result as embedding ids.
    #[test]
    fn test_graph_to_embedding_table_direction() {
        let mut graph_vocab = Vocab::new();
        for n in ["a", "b", "c"] { graph_vocab.get_or_insert("n", n); }
        let mut emb_vocab = Vocab::new();
        for n in ["c", "a"] { emb_vocab.get_or_insert("n", n); }

        let table = graph_to_embedding_table(&graph_vocab, &emb_vocab);
        let g = |n: &str| graph_vocab.get_node_id("n", n).unwrap();
        let e = |n: &str| emb_vocab.get_node_id("n", n);

        assert_eq!(table.len(), graph_vocab.len());
        assert_eq!(table[g("a")], e("a"));
        assert_eq!(table[g("b")], None);
        assert_eq!(table[g("c")], e("c"));
    }

    /// a -> {b, c}, b -> d, c -> d, d -> a.
    fn diamond() -> NamedGraph {
        let mut gb = crate::named_graph::GraphBuilder::new();
        for (f, t) in [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("d", "a")] {
            gb.add_edge(("n", f), ("n", t), 1., crate::graph::EdgeType::Directed);
        }
        gb.build(false).unwrap()
    }

    fn smci() -> SupervisedMCIteration {
        SupervisedMCIteration {
            iterations: 10, num_walks: 20, alpha: 0., discount: 0.9, step_penalty: 0.,
            explore_pct: 0., compression: 1., restart_prob: 0.1, seed: 3
        }
    }

    fn prob(g: &NamedGraph, from: &str, to: &str) -> f32 {
        let to_id = g.node_id("n", to).unwrap();
        let (edges, probs) = g.get_edges("n", from, true).unwrap();
        edges.iter().zip(probs).find(|(id, _)| **id == to_id).unwrap().1
    }

    /// Rewarding a -> d through b moves a's probability toward b, in a copy of the graph.
    #[test]
    fn test_learn_graph_shifts_toward_rewarded_paths() {
        let g = diamond();
        let a = g.node_id("n", "a").unwrap();
        let b = g.node_id("n", "b").unwrap();
        let learned = smci().learn_graph(&g, &[(a, b, 1.)], None);
        assert!(std::sync::Arc::ptr_eq(g.vocab(), learned.vocab()));
        assert_eq!((learned.nodes(), learned.edges()), (g.nodes(), g.edges()));
        assert!((prob(&g, "a", "b") - 0.5).abs() < 1e-6, "original graph untouched");
        assert!(prob(&learned, "a", "b") > prob(&learned, "a", "c"),
                "{} <= {}", prob(&learned, "a", "b"), prob(&learned, "a", "c"));
    }

    /// Embeddings with their own vocab are matched to the graph by name.
    #[test]
    fn test_learn_graph_with_embeddings() {
        let g = diamond();
        let mut b = crate::embeddings::NodeEmbeddingsBuilder::new(crate::distance::Distance::Euclidean);
        for (n, e) in [("d", [0., 1.]), ("b", [0., 0.9]), ("c", [5., 5.])] {
            b.add_embedding("n", n, &e).unwrap();
        }
        let embs = b.build().unwrap();
        let a = g.node_id("n", "a").unwrap();
        let d = g.node_id("n", "d").unwrap();
        let learned = smci().learn_graph(&g, &[(a, d, 1.)], Some(&embs));
        assert_eq!(learned.edges(), g.edges());
        let total: f32 = learned.get_edges("n", "a", true).unwrap().1.iter().sum();
        assert!((total - 1.).abs() < 1e-5 && total.is_finite());
    }
}
