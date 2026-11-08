//! Finds all connected components in the graph and returns a connected component list
use std::cmp::Reverse;
use std::collections::{BinaryHeap,HashSet};

use crate::NodeID;
use crate::graph::{Graph,CDFtoP};

use crate::embeddings::EmbeddingStore;
use crate::distance::Distance;

/// Find the connected components of a graph.  Labels are stored as f32, which is exact only up
/// to 2^24 components; use `component_labels` when exact labels matter.
pub fn find_connected_components(
    graph: &impl Graph
) -> EmbeddingStore {
    let n = graph.len();
    // u32 and f32 have the same size and alignment, so this reuses the label buffer in place.
    let labels: Vec<f32> = component_labels(graph).into_iter().map(|l| l as f32).collect();
    EmbeddingStore::new_with_vec(n, 1, Distance::Hamming, labels)
        .expect("one label per node")
}

/// Component label for every node, following out edges; labels are 1-based.  u32 keeps labels
/// exact past 2^24 components at the same 4 bytes per node as an f32, and a zero label doubles as
/// the "seen" marker so no separate bitset is needed.
pub fn component_labels(graph: &impl Graph) -> Vec<u32> {
    let n = graph.len();
    let mut labels = vec![0u32; n];
    let mut buffer: Vec<NodeID> = Vec::new();
    let mut label = 0u32;
    for start in 0..n {
        if labels[start] != 0 { continue }
        label = label.checked_add(1).expect("more than u32::MAX components");
        labels[start] = label;
        buffer.push(start);
        while let Some(node_id) = buffer.pop() {
            for edge in graph.get_edges(node_id).0.iter() {
                if labels[*edge] == 0 {
                    labels[*edge] = label;
                    buffer.push(*edge);
                }
            }
        }
    }
    labels
}

pub fn prune_graph_components(
    graph: &impl Graph,
    k: usize
) -> Vec<(NodeID,NodeID,f32)> {
    let labels = component_labels(graph);

    // Labels are 1-based.  Size the counts by the components that actually exist rather than by
    // the node count, which can be far larger.
    let num_components = labels.iter().max().map(|m| *m as usize).unwrap_or(0);
    let mut counts = vec![0u32; num_components + 1];
    for label in labels.iter() {
        counts[*label as usize] += 1;
    }

    // Keep the K largest components with a K-sized min-heap instead of sorting every component.
    let mut heap = BinaryHeap::with_capacity(k + 1);
    for (component_id, count) in counts.iter().enumerate().skip(1) {
        heap.push(Reverse((*count, component_id)));
        if heap.len() > k { heap.pop(); }
    }
    drop(counts);
    let clusters: HashSet<usize> = heap.into_iter().map(|Reverse((_, c))| c).collect();

    let mut subgraph = Vec::new();
    for node_id in 0..graph.len() {
        if clusters.contains(&(labels[node_id] as usize)) {
            let (edges, weights) = graph.get_edges(node_id);
            let p = CDFtoP::new(weights);
            for (edge, weight) in edges.iter().zip(p) {
                subgraph.push((node_id, *edge, weight));
            }
        }
    }

    subgraph 
}
        

#[cfg(test)]
mod connected_tests {
    use super::*;
    use crate::graph::{CSR, CumCSR};

    fn build_edges() -> Vec<(usize, usize, f32)> {
        vec![
            // Cluster 1
            (0, 1, 1.),
            (0, 2, 1.),
            (1, 0, 1.),
            (1, 2, 3.),
            (2, 1, 3.),
            (2, 0, 2.5),
            
            // Cluster 2
            (3, 4, 0.5),
            (4, 3, 0.5),
            (4, 5, 0.5),
            (4, 6, 0.5),
            (5, 4, 0.5),
            (5, 6, 0.5),
            (6, 4, 0.5),
            (6, 5, 0.5),

            // Cluster 3
            (7, 8, 0.5),
            (8, 7, 0.5),
        ]
    }

    #[test]
    fn run_test() {
        let graph = CSR::construct_from_edges(build_edges(), false);
        let es = find_connected_components(&graph);
        assert_eq!(es.get_embedding(0), &[1.0]);
        assert_eq!(es.get_embedding(1), &[1.0]);
        assert_eq!(es.get_embedding(2), &[1.0]);

        assert_eq!(es.get_embedding(3), &[2.0]);
        assert_eq!(es.get_embedding(4), &[2.0]);
        assert_eq!(es.get_embedding(5), &[2.0]);
        assert_eq!(es.get_embedding(6), &[2.0]);

        assert_eq!(es.get_embedding(7), &[3.0]);
    }

    fn undirected_clique(nodes: std::ops::Range<usize>) -> Vec<(usize, usize, f32)> {
        let mut edges = Vec::new();
        for a in nodes.clone() {
            for b in nodes.clone() {
                if a != b { edges.push((a, b, 1.)); }
            }
        }
        edges
    }

    /// Component sizes must be computed over every node, not just the first n/2 + 1.
    #[test]
    fn test_prune_keeps_largest_component() {
        // 4-node component on 0..4, 8-node component on 4..12.
        let mut edges = undirected_clique(0..4);
        edges.extend(undirected_clique(4..12));
        let graph = CumCSR::convert(CSR::construct_from_edges(edges, false));

        let pruned = prune_graph_components(&graph, 1);
        assert!(!pruned.is_empty());
        for (f, t, _) in pruned {
            assert!(f >= 4 && t >= 4, "kept edge ({}, {}) from the smaller component", f, t);
        }
    }

    /// Singleton components (self loops) mean there can be more than n/2 components.
    #[test]
    fn test_prune_more_than_half_n_components() {
        let edges = vec![(0, 0, 1.), (1, 1, 1.), (2, 2, 1.), (3, 3, 1.)];
        let graph = CumCSR::convert(CSR::construct_from_edges(edges, false));

        let pruned = prune_graph_components(&graph, 2);
        assert_eq!(pruned.len(), 2);
    }

    /// Component ids past 2^24 can't be told apart in an f32.  Every node here except the last
    /// two-node pair is its own component, so the last two nodes need different labels.
    #[test]
    fn test_component_labels_past_f32_precision() {
        let n = (1usize << 24) + 3;
        let graph = CSR::construct_from_edges(vec![(n - 1, n - 1, 1.)], false);
        let labels = component_labels(&graph);
        assert_eq!(labels.len(), n);
        // Node i is the (i + 1)th component found.
        let tail: Vec<usize> = labels[n - 3..].iter().map(|l| *l as usize).collect();
        assert_eq!(tail, vec![n - 2, n - 1, n]);
    }
}
