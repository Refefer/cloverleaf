/// This module defines the methods for computing distance embeddings using the Landmark selection
/// method.  It starts by selecting K landmarks (either randomly or by max degree), then computing
/// the distance between each node in the graph and each landmark.  
use std::collections::{VecDeque,BinaryHeap};
use std::cmp::Reverse;
use std::sync::Mutex;

use rayon::prelude::*;
use rand::prelude::*;
use rand_xorshift::XorShiftRng;

use crate::graph::{Graph,NodeID};
use crate::bitset::BitSet;
use crate::embeddings::{EmbeddingStore,Distance};

#[derive(Copy,Clone,Debug)]
pub enum LandmarkSelection {
    Random(u64),
    Degree
}

/// Ignores edge weights and computes it assuming each edge has the same weight.  This radically
/// changes the compute costs since we can avoid Djikstra's algorithm which is very expensive.
pub fn unweighted_walk_distance(
    graph: &impl Graph,
    start_node: NodeID
) -> Vec<u8> {
    let mut distance = vec![255; graph.len()];
    let mut seen = BitSet::new(graph.len());
    let mut queue = VecDeque::new();

    seen.set_bit(start_node);
    distance[start_node] = 0;
    queue.push_back((start_node, 0));

    while let Some((vert, cur_dist)) = queue.pop_front() {
        distance[vert] = cur_dist;
        for out_edge in graph.get_edges(vert).0.iter() {
            if !seen.is_set(*out_edge) {
                seen.set_bit(*out_edge);
                queue.push_back((*out_edge, cur_dist + 1));
            }
        }
    }   

    distance
}

/// Finds the top K nodes by degree.  Uses NodeID as a tie breaker.
/// TODO: We should use the TopK struct from ANN and refactor this away.
fn top_k_nodes(
    graph: &impl Graph,
    k: usize
) -> Vec<NodeID> {
    let mut bh = BinaryHeap::with_capacity(k + 1);
    for node_id in 0..graph.len() {
        let degrees = graph.degree(node_id);
        bh.push(Reverse((degrees, node_id)));
        if bh.len() > k {
            bh.pop();
        }
    }
    bh.into_iter().map(|Reverse((_, node_id))| node_id).collect()
}

/// Selects landmarks based on random selction.
fn rand_k_nodes(
    graph: &impl Graph,
    k: usize,
    seed: u64
) -> Vec<NodeID> {
    let mut rng = XorShiftRng::seed_from_u64(seed);
    (0..graph.len()).choose_multiple(&mut rng, k)
}

/// Constructs the distance embeddings, returning them upstream
pub fn construct_walk_distances(
    graph: &(impl Graph + Send + Sync),
    k: usize,
    ls: LandmarkSelection
) -> EmbeddingStore {
    let es = EmbeddingStore::new(graph.len(), k, Distance::ALT);
    let mut top_nodes = match ls {
        LandmarkSelection::Degree => top_k_nodes(graph, k),
        LandmarkSelection::Random(seed) => rand_k_nodes(graph, k, seed)
    };
    top_nodes.sort();

    let mes = Mutex::new(es);
    top_nodes.into_par_iter().enumerate().for_each(|(landmark_i, node_id)| {
        let node_distances = unweighted_walk_distance(graph, node_id);
        {
            let mut embeddings = mes.lock().unwrap();
            node_distances.into_iter().enumerate().for_each(|(idx, d)| {
                let embedding = embeddings.get_embedding_mut(idx);
                embedding[landmark_i] = d as f32;
            });
        }
    });
    mes.into_inner().expect("No references should be left!")
}

#[cfg(test)]
mod dist_tests {
    use super::*;
    use crate::graph::CSR;

    fn build_edges() -> Vec<(usize, usize, f32)> {
        vec![
            (0, 1, 1.),
            (1, 1, 3.),
            (1, 2, 2.5),
            (2, 0, 10.),
            (2, 3, 1.)
        ]
    }

    #[test]
    fn construct_distances() {
        let edges = build_edges();

        let csr = CSR::construct_from_edges(edges);
        let distances = unweighted_walk_distance(&csr, 0);
        assert_eq!(distances, vec![0, 1, 2, 3]);

        let distances = unweighted_walk_distance(&csr, 2);
        assert_eq!(distances, vec![1, 2, 0, 1]);

        let distances = unweighted_walk_distance(&csr, 3);
        assert_eq!(distances, vec![255, 255, 255, 0]);
    }

    #[test]
    fn test_top_k() {
        let edges = build_edges();

        let csr = CSR::construct_from_edges(edges);
        let top = top_k_nodes(&csr, 2);
        assert_eq!(top, vec![1, 2]);

    }

    #[test]
    fn test_distance_embeddings() {
        let edges = build_edges();

        let csr = CSR::construct_from_edges(edges);
        let es = construct_walk_distances(&csr, 2, LandmarkSelection::Degree);
        assert_eq!(es.get_embedding(0), vec![2f32, 1f32]);
        assert_eq!(es.get_embedding(1), vec![0f32, 2f32]);
        assert_eq!(es.get_embedding(2), vec![1f32, 0f32]);
        assert_eq!(es.get_embedding(3), vec![2f32, 1f32]);
    }

}
