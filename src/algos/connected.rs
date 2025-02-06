//! Finds all connected components in the graph and returns a connected component list
use std::collections::HashSet;

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
    
    // A graph, at worse, has a max of N / 2 components
    let n = graph.len();
    let mut counts = vec![0usize; n / 2 + 1];

    for node_id in 0..counts.len() {
        let component_id = get_component(&es, node_id);
        counts[component_id] += 1;
    }
    let mut idxs = vec![0; n / 2 + 1];
    idxs.iter_mut().enumerate().for_each(|(i, v)| *v = i);
    idxs.sort_by_key(|v| counts[*v]);
    // Grab the top K clusters
    let clusters = idxs.iter().rev().take(k).collect::<HashSet<_>>();

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
    use crate::graph::CSR;

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
        let graph = CSR::construct_from_edges(build_edges());
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
}
