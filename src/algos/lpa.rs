use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use hashbrown::HashMap;
use rand::prelude::*;
use rand_distr::{Distribution,Uniform};
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;

use crate::graph::{Graph,NodeID};
use crate::bitset::BitSet;
use crate::embeddings::{EmbeddingStore,Distance};

struct Counter<'a> {
    slice: &'a [usize],
    idx: usize
}

impl <'a> Counter<'a> {
    fn new(slice: &'a [usize]) -> Self {
        Counter {
            slice,
            idx: 0
        }
    }
}

impl <'a> Iterator for Counter<'a> {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        let start = self.idx;
        let mut count = 0;
        for _ in self.idx..self.slice.len() {
            if self.slice[self.idx] != self.slice[start] { 
                return Some((self.slice[start], count)) 
            }
            self.idx += 1;
            count += 1;
        }
        if count > 0 {
            return Some((self.slice[start], count)) 
        } 
        None
    }
}

fn get_best_count<R: Rng>(counts: &[usize], rng: &mut R) -> usize{
    let mut best_count = 0;
    let mut ties = Vec::new();
    for (cluster, count) in Counter::new(counts) {
        if count > best_count {
            best_count = count;
            ties.clear();
            ties.push(cluster)
        } else if count == best_count {
            ties.push(cluster)
        }
    }

    // We tie break by randomly choosing an item
    if ties.len() > 1 {
        *ties.as_slice()
            .choose(rng)
            .expect("If a node has no edges, code bug")
    } else {
        ties[0]
    }

}

pub fn lpa(
    graph: &impl Graph,
    passes: usize,
    seed: u64
) -> Vec<usize> {
    let mut rng = XorShiftRng::seed_from_u64(seed);
    let mut clusters: Vec<_> = (0..graph.len()).collect();
    let mut idxs = clusters.clone();
    let max_degree = (0..graph.len())
        .map(|ni| graph.degree(ni))
        .max()
        .unwrap_or(1);

    let mut counts = vec![0; max_degree];
    for i_pass in 0..passes {
        idxs.shuffle(&mut rng);
        for idx in idxs.iter() {
            // Count neighbors
            let node_edges = graph.get_edges(*idx).0;
            let n = node_edges.len();
            for (t_node, count) in node_edges.iter().zip(counts.iter_mut()) {
                *count = clusters[*t_node];
            }

            let mut slice = &mut counts[0..n];
            slice.sort_unstable();
            clusters[*idx] = get_best_count(&slice, &mut rng);
        }
    }
    clusters
}

pub fn construct_lpa_embedding(
    graph: &(impl Graph + Send + Sync),
    k: usize,
    passes: usize,
    seed: u64
) -> EmbeddingStore {
    let mut es = EmbeddingStore::new(graph.len(), k, Distance::Hamming);
    let mes = Mutex::new(es);

    println!("k={},passes={},seed={}", k, passes, seed);
    let count = AtomicUsize::new(0);
    // Compute LPA in parallel
    (0..k).into_par_iter().for_each(|k_idx| {
        let clusters = lpa(graph, passes, seed + k_idx as u64);
        {
            let mut embeddings = mes.lock().unwrap();
            clusters.into_iter().enumerate().for_each(|(idx, cluster)| {
                let embedding = embeddings.get_embedding_mut(idx);
                embedding[k_idx] = cluster as f32;
            });
        }
        let num_done = count.fetch_add(1, Ordering::Relaxed);
        println!("Finished {}/{}", num_done + 1, k);
    });

    mes.into_inner().expect("No references should be left!")

}

#[cfg(test)]
mod lpa_tests {
    use super::*;

    fn build_edges() -> Vec<(usize, usize, f32)> {
        vec![
            (0, 1, 1.),
            (1, 1, 3.),
            (1, 2, 2.),
            (2, 0, 2.5),
            (1, 0, 10.),
        ]
    }

    #[test]
    fn test_choose_best() {
        let counts = vec![0, 0, 1, 1, 1, 2];
        let mut rng = XorShiftRng::seed_from_u64(1);
        let best_count = get_best_count(&counts, &mut rng);
        assert_eq!(best_count, 1);
    }

    #[test]
    fn test_choose_one() {
        let counts = vec![0, 0, 0];
        let mut rng = XorShiftRng::seed_from_u64(1);
        let best_count = get_best_count(&counts, &mut rng);
        assert_eq!(best_count, 0);
    }

    #[test]
    fn test_choose_between() {
        let counts = vec![0, 1];
        let mut rng = XorShiftRng::seed_from_u64(1231232132);
        let best_count = get_best_count(&counts, &mut rng);
        assert_eq!(best_count, 0);
        let best_count = get_best_count(&counts, &mut rng);
        assert_eq!(best_count, 1);
        let best_count = get_best_count(&counts, &mut rng);
        assert_eq!(best_count, 1);
        let best_count = get_best_count(&counts, &mut rng);
        assert_eq!(best_count, 0);
    }

    #[test]
    fn test_choose_last() {
        let counts = vec![0, 1, 1];
        let mut rng = XorShiftRng::seed_from_u64(1231232132);
        let best_count = get_best_count(&counts, &mut rng);
        assert_eq!(best_count, 1);
    }

    #[test]
    fn test_choose_only() {
        let counts = vec![0, 0, 0];
        let mut rng = XorShiftRng::seed_from_u64(1231232132);
        let best_count = get_best_count(&counts, &mut rng);
        assert_eq!(best_count, 0);
    }

    #[test]
    fn test_counter() {
        let counts = [0, 0, 0, 1, 2, 2, 3];
        let mut counter = Counter::new(&counts);
        assert_eq!(counter.next(), Some((0, 3)));
        assert_eq!(counter.next(), Some((1, 1)));
        assert_eq!(counter.next(), Some((2, 2)));
        assert_eq!(counter.next(), Some((3, 1)));
        assert_eq!(counter.next(), None);

        let counts = [];
        let mut counter = Counter::new(&counts);
        assert_eq!(counter.next(), None);

        let counts = [0];
        let mut counter = Counter::new(&counts);
        assert_eq!(counter.next(), Some((0, 1)));
        assert_eq!(counter.next(), None);

    }



}

