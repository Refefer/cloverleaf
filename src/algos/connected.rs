//! Finds all connected components in the graph and returns a connected component list
use std::cmp::Reverse;
use std::collections::{BinaryHeap,HashSet};

use crate::NodeID;
use crate::bitset::BitSet;
use crate::graph::{Graph,CDFtoP};

use crate::embeddings::EmbeddingStore;
use crate::distance::Distance;

/// Find the connected components of a graph
pub fn find_connected_components(
    graph: &impl Graph
) -> EmbeddingStore {
    let mut es = EmbeddingStore::new(graph.len(), 1, Distance::Hamming);

    let mut last_start_idx = 0usize;
    let mut local = BitSet::new(graph.len());
    let mut buffer = Vec::new();

    let mut cluster_idx = 1;
    loop {
        let mut all_done = true;
        // Find the first unseen index.  We track the last start index so it's O(n) instead of
        // O(n^2)
        for idx in last_start_idx..graph.len() {
            if es.get_embedding(idx)[0] == 0.0 {
                last_start_idx = idx;
                all_done = false;
                break
            }
        }
        // We've seen all the nodes, all done
        if all_done { break }

        local.set_bit(last_start_idx);
        buffer.push(last_start_idx);

        while buffer.len() > 0 {
            let node_id = buffer.pop().expect("Shouldn't be empty!");
            es.get_embedding_mut(node_id)[0] = cluster_idx as f32;
            for edge in graph.get_edges(node_id).0.iter() {
                if !local.is_set(*edge) {
                    local.set_bit(*edge);
                    buffer.push(*edge);
                }
            }
        }

        cluster_idx += 1;
    }
    es
}

fn get_component(
    embs: &EmbeddingStore, 
    node_id: NodeID
) -> usize {
    embs.get_embedding(node_id)[0] as usize
}

pub fn prune_graph_components(
    graph: &impl Graph,
    k: usize
) -> Vec<(NodeID,NodeID,f32)> {
    let es = find_connected_components(graph);
    
    // Component ids are 1-based.  Size the counts by the components that actually exist rather
    // than by the node count, which can be far larger.
    let n = graph.len();
    let num_components = (0..n).map(|node_id| get_component(&es, node_id)).max().unwrap_or(0);
    let mut counts = vec![0usize; num_components + 1];
    for node_id in 0..n {
        counts[get_component(&es, node_id)] += 1;
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
        let component_id = get_component(&es, node_id);
        if clusters.contains(&component_id) {
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
}
