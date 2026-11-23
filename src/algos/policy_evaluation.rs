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

use crate::error::Error;
use crate::io::read_node_values;
use crate::named_graph::NamedGraph;
use crate::vocab::Vocab;
use rayon::prelude::*;
use std::fmt::Write;
use std::sync::Arc;

use crate::graph::{Graph, NodeID, ParallelModifiableGraph, convert_edges_to_cdf};
use crate::progress::CLProgressBar;

/// A reward for every node of a graph, for [`PolicyEvaluation::optimize_rewards`].  Every node
/// starts at 0.
#[derive(Clone,Debug)]
pub struct NodeRewards {
    vocab: Arc<Vocab>,
    values: Vec<f32>
}

impl NodeRewards {
    /// A reward of 0 for each of the graph's nodes.
    pub fn new(graph: &NamedGraph) -> Self {
        NodeRewards { vocab: graph.vocab().clone(), values: vec![0f32; graph.nodes()] }
    }

    /// Sets a node's reward, or errors with [`Error::UnknownNode`].
    pub fn set(&mut self, node_type: &str, name: &str, value: f32) -> Result<(), Error> {
        let node_id = self.vocab.node_id(node_type, name)?;
        self.values[node_id] = value;
        Ok(())
    }

    /// A node's reward, or [`Error::UnknownNode`].
    pub fn get(&self, node_type: &str, name: &str) -> Result<f32, Error> {
        Ok(self.values[self.vocab.node_id(node_type, name)?])
    }

    /// Sets rewards from a `node_type<TAB>name<TAB>reward` file; nodes not in it keep theirs.
    /// The whole file is read first, so a malformed line or unknown node changes nothing.
    pub fn load(&mut self, path: &str) -> Result<(), Error> {
        for (node_id, value) in read_node_values(path, &self.vocab)? {
            self.values[node_id] = value;
        }
        Ok(())
    }

    pub fn vocab(&self) -> &Arc<Vocab> {
        &self.vocab
    }

    /// Every node's reward, indexed by node id.
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

#[derive(Clone,Copy,Debug)]
pub struct PolicyEvaluation {
    pub gamma: f32,
    pub iterations: usize,
    pub eps: f32,
    pub temperature: f32,
    pub indicator: bool,
}

impl PolicyEvaluation {
    /// Runs policy evaluation with a reward per node and rewrites the graph's edge weights to
    /// the resulting policy.  A graph shared with other `NamedGraph`s is copied first; check
    /// [`NamedGraph::graph_mut`] beforehand to avoid that.
    pub fn optimize(&self, graph: &mut NamedGraph, rewards: &[f32]) -> Result<(), Error> {
        if rewards.len() != graph.nodes() {
            return Err(Error::InvalidArgument(format!(
                "{} rewards for {} nodes", rewards.len(), graph.nodes())))
        }
        let csr = graph.make_graph_mut();
        let values = self.compute(csr, rewards);
        self.update_policy_weights_in_place(csr, &values);
        Ok(())
    }

    /// [`optimize`](Self::optimize) with named rewards, which must have been built from this
    /// graph or a clone of it (the same vocab, not merely an equal one), else an
    /// [`Error::VocabMismatch`].  Unless `allow_copy`, a graph shared with other owners is an
    /// [`Error::WouldCopyGraph`] rather than copied.  Either error leaves the graph unchanged.
    pub fn optimize_rewards(
        &self,
        graph: &mut NamedGraph,
        rewards: &NodeRewards,
        allow_copy: bool
    ) -> Result<(), Error> {
        if !Arc::ptr_eq(graph.vocab(), rewards.vocab()) {
            return Err(Error::VocabMismatch("NodeRewards and Graph"))
        }
        if !allow_copy && graph.graph_mut().is_none() {
            return Err(Error::WouldCopyGraph)
        }
        self.optimize(graph, rewards.values())
    }
}

/// The defaults the Python bindings use.
impl Default for PolicyEvaluation {
    fn default() -> Self {
        PolicyEvaluation::new(0.99, 100, 1e-6, 1.0, true)
    }
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

#[cfg(test)]
mod policy_evaluation_tests {
    use super::*;
    use crate::graph::EdgeType;
    use crate::named_graph::GraphBuilder;

    /// a -> b, a -> c, b -> c, c -> a.
    fn graph() -> NamedGraph {
        let mut gb = GraphBuilder::new();
        for (f, t) in [("a", "b"), ("a", "c"), ("b", "c"), ("c", "a")] {
            gb.add_edge(("n", f), ("n", t), 1., EdgeType::Directed);
        }
        gb.build(true).unwrap()
    }

    fn quiet() -> PolicyEvaluation {
        PolicyEvaluation { indicator: false, ..PolicyEvaluation::default() }
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("cloverleaf-pe-{}-{}", std::process::id(), name))
    }

    #[test]
    fn test_node_rewards_set_and_load() {
        let g = graph();
        let mut rewards = NodeRewards::new(&g);
        assert!(Arc::ptr_eq(rewards.vocab(), g.vocab()));
        assert_eq!(rewards.values(), &[0., 0., 0.]);
        rewards.set("n", "b", 2.).unwrap();
        assert!(matches!(rewards.set("n", "zzz", 1.), Err(Error::UnknownNode { .. })));
        assert_eq!(rewards.get("n", "b").unwrap(), 2.);

        let path = temp_path("rewards");
        std::fs::write(&path, "n\tc\t-1.5\n").unwrap();
        rewards.load(path.to_str().unwrap()).unwrap();
        // A bad line anywhere changes nothing.
        std::fs::write(&path, "n\ta\t3\nn\tzzz\t1\n").unwrap();
        let unknown = rewards.load(path.to_str().unwrap());
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(unknown, Err(Error::UnknownNode { .. })));
        assert_eq!((rewards.get("n", "a").unwrap(), rewards.get("n", "b").unwrap(), rewards.get("n", "c").unwrap()),
                   (0., 2., -1.5));
    }

    #[test]
    fn test_optimize_rewards_matches_optimize() {
        let mut g = graph();
        let mut rewards = NodeRewards::new(&g);
        rewards.set("n", "b", 1.).unwrap();
        let mut expected = NamedGraph::new(g.vocab().clone(), Arc::new(g.graph().clone())).unwrap();
        quiet().optimize(&mut expected, rewards.values()).unwrap();
        quiet().optimize_rewards(&mut g, &rewards, false).unwrap();
        assert_eq!(g.get_edges("n", "a", true).unwrap(), expected.get_edges("n", "a", true).unwrap());
        // a now prefers b, the rewarded node.
        let (edges, probs) = g.get_edges("n", "a", true).unwrap();
        let to_b = edges.iter().position(|e| *e == g.node_id("n", "b").unwrap()).unwrap();
        assert!(probs[to_b] > 0.5);
    }

    /// Rewards from an equal but separately built graph are rejected: only the same vocab counts.
    #[test]
    fn test_optimize_rewards_needs_the_same_vocab() {
        let mut g = graph();
        let rewards = NodeRewards::new(&graph());
        assert!(g.vocab().is_aligned_with(rewards.vocab()));
        let before = g.get_edges("n", "a", false).unwrap().1;
        assert!(matches!(quiet().optimize_rewards(&mut g, &rewards, true), Err(Error::VocabMismatch(_))));
        assert_eq!(g.get_edges("n", "a", false).unwrap().1, before);
    }

    /// A shared graph is an error unless copying is allowed, and then the other owner keeps the
    /// original.
    #[test]
    fn test_optimize_rewards_copies_only_when_allowed() {
        let mut g = graph();
        let shared = g.clone();
        let mut rewards = NodeRewards::new(&g);
        rewards.set("n", "b", 1.).unwrap();
        assert!(matches!(quiet().optimize_rewards(&mut g, &rewards, false), Err(Error::WouldCopyGraph)));
        assert!(Arc::ptr_eq(g.shared_graph(), shared.shared_graph()));

        quiet().optimize_rewards(&mut g, &rewards, true).unwrap();
        assert!(!Arc::ptr_eq(g.shared_graph(), shared.shared_graph()));
        assert_ne!(g.get_edges("n", "a", true).unwrap().1, shared.get_edges("n", "a", true).unwrap().1);
        // No longer shared, so in place is fine.
        quiet().optimize_rewards(&mut g, &rewards, false).unwrap();
    }
}
