//! Graphs whose nodes have names: a [`CumCSR`] plus the [`Vocab`] that names its node ids.
use std::collections::HashSet;
use std::io::Write;
use std::sync::Arc;

use crate::algos::connected::prune_graph_components;
use crate::error::Error;
use crate::graph::{CDFtoP, CumCSR, CSR, EdgeType, Graph, NodeID, Transpose};
use crate::io::{GraphLoadError, GraphReader, open_file_for_writing};
use crate::vocab::Vocab;

/// A graph with named nodes.  Both parts are shared, so cloning is cheap and the vocab can be
/// shared with feature sets and embeddings built from the graph.
#[derive(Clone)]
pub struct NamedGraph {
    pub(crate) graph: Arc<CumCSR>,
    pub(crate) vocab: Arc<Vocab>
}

/// How [`NamedGraph::load`] reads a graph file.
#[derive(Clone,Copy,Debug)]
pub struct GraphLoadOptions {
    /// Lines parsed per parallel chunk.
    pub chunk_size: usize,
    /// Header lines to skip.
    pub skip_rows: usize,
    /// Read the weight field; otherwise every edge weighs 1.
    pub weighted: bool,
    /// Merge repeated edges.
    pub deduplicate: bool
}

impl Default for GraphLoadOptions {
    fn default() -> Self {
        GraphLoadOptions { chunk_size: 1, skip_rows: 0, weighted: true, deduplicate: false }
    }
}

impl NamedGraph {
    /// Pairs a graph with the vocab that names its nodes.
    pub fn new(vocab: Arc<Vocab>, graph: Arc<CumCSR>) -> Result<Self, Error> {
        if vocab.len() != graph.len() {
            return Err(Error::InvalidArgument(format!(
                "vocab has {} nodes but the graph has {}", vocab.len(), graph.len())))
        }
        Ok(NamedGraph { graph, vocab })
    }

    /// Reads a `from_type<TAB>from_name<TAB>to_type<TAB>to_name<TAB>weight` file, which may be
    /// gzipped.
    pub fn load(path: &str, edge_type: EdgeType, options: &GraphLoadOptions) -> Result<Self, GraphLoadError> {
        let (vocab, graph) = GraphReader::load(
            path, edge_type, options.chunk_size, options.skip_rows, options.weighted,
            options.deduplicate)?;
        Ok(NamedGraph { graph: Arc::new(graph), vocab: Arc::new(vocab) })
    }

    /// Writes every edge in the format `load` reads, with transition probabilities as weights.
    pub fn save(&self, path: &str, comp_level: Option<u32>) -> std::io::Result<()> {
        let mut out = open_file_for_writing(path, comp_level)?;
        for node in 0..self.graph.len() {
            let (f_node_type, f_name) = self.vocab.get_name(node).expect("every node is named");
            let (edges, weights) = self.graph.get_edges(node);
            for (out_node, weight) in edges.iter().zip(CDFtoP::new(weights)) {
                let (t_node_type, t_name) = self.vocab.get_name(*out_node).expect("every node is named");
                writeln!(out, "{}\t{}\t{}\t{}\t{}", f_node_type, f_name, t_node_type, t_name, weight)?;
            }
        }
        out.flush()
    }

    pub fn graph(&self) -> &CumCSR {
        &self.graph
    }

    pub fn shared_graph(&self) -> &Arc<CumCSR> {
        &self.graph
    }

    /// Mutable access to the graph, or None while it's shared with another `NamedGraph` (or any
    /// other clone of its `Arc`).
    pub fn graph_mut(&mut self) -> Option<&mut CumCSR> {
        Arc::get_mut(&mut self.graph)
    }

    /// Mutable access to the graph, copying it first if it's shared.
    pub fn make_graph_mut(&mut self) -> &mut CumCSR {
        Arc::make_mut(&mut self.graph)
    }

    pub fn vocab(&self) -> &Arc<Vocab> {
        &self.vocab
    }

    pub fn nodes(&self) -> usize {
        self.graph.len()
    }

    pub fn edges(&self) -> usize {
        self.graph.edges()
    }

    pub fn node_id(&self, node_type: &str, name: &str) -> Result<NodeID, Error> {
        self.vocab.node_id(node_type, name)
    }

    pub fn contains_node(&self, node_type: &str, name: &str) -> bool {
        self.vocab.get_node_id(node_type, name).is_some()
    }

    /// A node's out edges with their transition probabilities if `normalized`, otherwise their
    /// cumulative distribution (which ends at 1).
    pub fn get_edges(&self, node_type: &str, name: &str, normalized: bool) -> Result<(&[NodeID], Vec<f32>), Error> {
        let (edges, weights) = self.graph.get_edges(self.node_id(node_type, name)?);
        let weights = if normalized { CDFtoP::new(weights).collect() } else { weights.to_vec() };
        Ok((edges, weights))
    }

    /// The graph with every edge reversed.
    pub fn transpose(&self) -> Self {
        NamedGraph { graph: Arc::new(self.graph.transpose()), vocab: self.vocab.clone() }
    }

    /// Splits the nodes into those whose type is in `node_types` and the rest.
    pub fn split_by_node_types(&self, node_types: &HashSet<String>) -> (Vec<NodeID>, Vec<NodeID>) {
        (0..self.graph.len()).partition(|id| self.vocab.has_node_type(*id, node_types))
    }

    /// A new graph of only the `k` largest connected components, with its own vocab.  None if
    /// nothing is left.
    pub fn prune_largest_components(&self, k: usize) -> Option<NamedGraph> {
        let mut gb = GraphBuilder::new();
        for (f, t, w) in prune_graph_components(self.graph.as_ref(), k) {
            let (fnt, fnn) = self.vocab.get_name(f).expect("every node is named");
            let (tnt, tnn) = self.vocab.get_name(t).expect("every node is named");
            gb.add_edge((fnt.as_str(), fnn), (tnt.as_str(), tnn), w, EdgeType::Directed);
        }
        gb.build(false)
    }
}

/// Builds a [`NamedGraph`] one edge at a time.
pub struct GraphBuilder {
    pub(crate) vocab: Vocab,
    edges: Vec<(NodeID, NodeID, f32)>
}

impl GraphBuilder {
    /// Whether the Python bindings' `build_graph` merges repeated edges when not told.  Loading a
    /// graph file doesn't by default ([`GraphLoadOptions::default`]).
    pub const DEFAULT_DEDUPLICATE: bool = true;

    pub fn new() -> Self {
        GraphBuilder { vocab: Vocab::new(), edges: Vec::new() }
    }

    /// Adds an edge between `(node_type, name)` nodes; undirected edges are added both ways.
    pub fn add_edge(&mut self, from: (&str, &str), to: (&str, &str), weight: f32, edge_type: EdgeType) {
        let f_id = self.vocab.get_or_insert(from.0, from.1);
        let t_id = self.vocab.get_or_insert(to.0, to.1);
        self.edges.push((f_id, t_id, weight));
        if edge_type == EdgeType::Undirected {
            self.edges.push((t_id, f_id, weight));
        }
    }

    pub fn nodes(&self) -> usize {
        self.vocab.len()
    }

    pub fn edges(&self) -> usize {
        self.edges.len()
    }

    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    /// Takes the edges added so far, leaving the builder empty, and builds the graph; `deduplicate`
    /// merges repeated edges.  None if no edges were added.
    pub fn build(&mut self, deduplicate: bool) -> Option<NamedGraph> {
        if self.edges.is_empty() {
            return None
        }
        // Take the buffers rather than copy them.
        let vocab = std::mem::replace(&mut self.vocab, Vocab::new());
        let edges = std::mem::take(&mut self.edges);
        let graph = CumCSR::convert(CSR::construct_from_edges(edges, deduplicate));
        Some(NamedGraph { graph: Arc::new(graph), vocab: Arc::new(vocab) })
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        GraphBuilder::new()
    }
}

#[cfg(test)]
mod named_graph_tests {
    use super::*;

    /// a -> b -> c, plus an isolated x <-> y pair of another type.
    fn graph() -> NamedGraph {
        let mut gb = GraphBuilder::new();
        gb.add_edge(("n", "a"), ("n", "b"), 1., EdgeType::Directed);
        gb.add_edge(("n", "b"), ("n", "c"), 1., EdgeType::Directed);
        gb.add_edge(("n", "a"), ("n", "c"), 3., EdgeType::Directed);
        gb.add_edge(("m", "x"), ("m", "y"), 1., EdgeType::Undirected);
        gb.build(false).unwrap()
    }

    fn names(g: &NamedGraph, ids: &[NodeID]) -> Vec<(String, String)> {
        let mut names: Vec<_> = ids.iter().map(|id| g.vocab().get_name_owned(*id).unwrap()).collect();
        names.sort();
        names
    }

    fn node(t: &str, n: &str) -> (String, String) {
        (t.to_string(), n.to_string())
    }

    #[test]
    fn test_new_rejects_mismatched_sizes() {
        let g = graph();
        let mut vocab = Vocab::new();
        vocab.get_or_insert("n", "a");
        assert!(matches!(NamedGraph::new(Arc::new(vocab), g.shared_graph().clone()),
                         Err(Error::InvalidArgument(_))));
        assert!(NamedGraph::new(g.vocab().clone(), g.shared_graph().clone()).is_ok());
    }

    #[test]
    fn test_builder_is_emptied_by_build() {
        let mut gb = GraphBuilder::new();
        assert!(gb.build(false).is_none());
        gb.add_edge(("n", "a"), ("n", "b"), 1., EdgeType::Undirected);
        assert_eq!((gb.nodes(), gb.edges()), (2, 2));
        let g = gb.build(false).unwrap();
        assert_eq!((g.nodes(), g.edges()), (2, 2));
        assert_eq!((gb.nodes(), gb.edges()), (0, 0));
        assert!(gb.build(false).is_none());
    }

    #[test]
    fn test_get_edges_weights() {
        let g = graph();
        let (edges, cumulative) = g.get_edges("n", "a", false).unwrap();
        assert_eq!(names(&g, edges), vec![node("n", "b"), node("n", "c")]);
        assert_eq!(cumulative.last(), Some(&1.));
        assert!(cumulative.windows(2).all(|w| w[0] <= w[1]));
        let (_, probs) = g.get_edges("n", "a", true).unwrap();
        assert!((probs.iter().sum::<f32>() - 1.).abs() < 1e-6);
        assert!(matches!(g.get_edges("n", "nope", true), Err(Error::UnknownNode { .. })));
    }

    #[test]
    fn test_transpose_reverses_edges_and_shares_vocab() {
        let g = graph();
        let t = g.transpose();
        assert!(Arc::ptr_eq(g.vocab(), t.vocab()));
        assert_eq!(t.edges(), g.edges());
        let (into_c, _) = t.get_edges("n", "c", true).unwrap();
        assert_eq!(names(&t, into_c), vec![node("n", "a"), node("n", "b")]);
        assert!(t.get_edges("n", "a", true).unwrap().0.is_empty());
    }

    #[test]
    fn test_split_by_node_types() {
        let g = graph();
        let (m, rest) = g.split_by_node_types(&HashSet::from(["m".to_string()]));
        assert_eq!(names(&g, &m), vec![node("m", "x"), node("m", "y")]);
        assert_eq!(names(&g, &rest), vec![node("n", "a"), node("n", "b"), node("n", "c")]);
        let (none, all) = g.split_by_node_types(&HashSet::new());
        assert!(none.is_empty());
        assert_eq!(all.len(), g.nodes());
    }

    #[test]
    fn test_prune_largest_components() {
        let g = graph();
        let pruned = g.prune_largest_components(1).unwrap();
        assert!(!Arc::ptr_eq(g.vocab(), pruned.vocab()));
        assert_eq!(pruned.nodes(), 3);
        assert!(pruned.contains_node("n", "a") && !pruned.contains_node("m", "x"));
        assert_eq!(pruned.edges(), 3);
        let (edges, probs) = pruned.get_edges("n", "a", true).unwrap();
        let mut got: Vec<_> = edges.iter().zip(probs)
            .map(|(id, p)| (pruned.vocab().get_name_owned(*id).unwrap(), (p * 4.).round()))
            .collect();
        got.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(got, vec![(node("n", "b"), 1.), (node("n", "c"), 3.)]);

        let both = g.prune_largest_components(2).unwrap();
        assert_eq!((both.nodes(), both.edges()), (g.nodes(), g.edges()));
    }
}
