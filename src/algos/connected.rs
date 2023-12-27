//! Finds all connected components in the graph and returns a connected component list
use crate::bitset::BitSet;
use crate::graph::Graph;

use crate::embeddings::{EmbeddingStore, Distance};

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
