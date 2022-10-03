use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;

use crate::graph::Graph;
use crate::embeddings::{EmbeddingStore,Distance};
use crate::algos::utils::get_best_count;

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
    for _i_pass in 0..passes {
        idxs.shuffle(&mut rng);
        for idx in idxs.iter() {
            // Count neighbors
            let node_edges = graph.get_edges(*idx).0;
            let n = node_edges.len();
            for (t_node, count) in node_edges.iter().zip(counts.iter_mut()) {
                *count = clusters[*t_node];
            }

            let slice = &mut counts[0..n];
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
    let es = EmbeddingStore::new(graph.len(), k, Distance::Hamming);
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

