use std::cmp::{Eq,PartialEq,Ordering,Reverse};
use std::collections::BinaryHeap;

use rayon::prelude::*;
use hashbrown::HashSet;
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rand_distr::{Distribution,Uniform};
use float_ord::FloatOrd;

use crate::graph::{Graph as CGraph,NodeID};
use crate::embeddings::{EmbeddingStore,Entity};
use crate::progress::CLProgressBar;

#[derive(Copy, Clone)]
pub struct NodeDistance(f32, NodeID);

impl NodeDistance {
    pub fn to_tup(&self) -> (NodeID, f32) {
        (self.1, self.0)
    }
}

// Min Heap, so reverse order
impl Ord for NodeDistance {
    fn cmp(&self, other: &Self) -> Ordering {
        FloatOrd(other.0).cmp(&FloatOrd(self.0))
            .then_with(|| other.1.cmp(&self.1))
    }
}

impl PartialOrd for NodeDistance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for NodeDistance {
    fn eq(&self, other: &Self) -> bool {
        FloatOrd(self.0) == FloatOrd(other.0) && self.1 == other.1
    }
}

impl Eq for NodeDistance {}

#[derive(Debug)]
pub struct Ann {
    k: usize,
    max_steps: usize,
    seed: u64
}

impl Ann {
    pub fn new(k: usize, max_steps: usize, seed: u64) -> Self {
        Ann {k, max_steps, seed}
    }

    pub fn find<G: CGraph + Send + Sync>(
        &self, 
        query: &[f32],
        graph: &G, 
        embeddings: &EmbeddingStore,
    ) -> Vec<NodeDistance> {
        let mut rng = XorShiftRng::seed_from_u64(self.seed);
        let distribution = Uniform::new(0, graph.len());
        let start_node = distribution.sample(&mut rng);
        hill_climb(
            Entity::Embedding(query), 
            start_node,
            graph,
            embeddings,
            self.k,
            self.max_steps)
    }
    
}

fn hill_climb<'a, G: CGraph>(
    needle: Entity<'a>, 
    start: NodeID, 
    graph: &G, 
    es: &EmbeddingStore,
    k: usize,
    mut max_steps: usize
) -> Vec<NodeDistance> {
    let mut heap = BinaryHeap::new();
    let mut best = BinaryHeap::new();
    let mut seen = HashSet::new();

    seen.insert(start.clone());
    let start_d = es.compute_distance(&needle, &Entity::Node(start.clone()));
    let start = NodeDistance(start_d, start);
    heap.push(start.clone());

    loop {
        let cur_node = heap.pop().expect("Shouldn't be empty!");
        // Add to best, potentially
        best.push(Reverse(cur_node.clone()));
        if best.len() > k {
            // throw away furthest away
            best.pop();
        }
        // Get edges, compute distances between them and needle, add to the heap
        for edge in graph.get_edges(cur_node.1).0.iter() {
            if !seen.contains(edge) {
                seen.insert(*edge);
                let dist = es.compute_distance(&needle, &Entity::Node(*edge));
                heap.push(NodeDistance(dist, *edge));
            }
        }

        max_steps -= 1;
        if max_steps == 0 || heap.len() == 0 {
            break
        }
    }
    let mut results: Vec<NodeDistance> = best.into_iter().map(|n| n.0).collect();
    results.sort();
    results.reverse();
    results
}

#[cfg(test)]
mod ann_tests {
    use super::*;
    use crate::graph::{CumCSR,CSR};

    fn build_star_edges() -> Vec<(usize, usize, f32)> {
        let mut edges = Vec::new();
        let max = 100;
        for ni in 0..max {
            for no in (ni+1)..max {
                edges.push((ni, no, 1f32));
                edges.push((no, ni, 1f32));
            }
        }
        edges
    }

    #[test]
    fn test_simple_learn_dist() {
    }

}
