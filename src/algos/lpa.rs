//! Classic label propagation algorithm for learning clusters based on the graph.  Not much to say;
//! it's single threaded, fast, and a bit finicky on the number of passes (overfitting can produce
//! worse clusters), but it's a good baseline.
use std::fmt::Write;

use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;

use crate::progress::CLProgressBar;
use crate::graph::Graph;
use crate::embeddings::{EmbeddingStore,Distance};
use crate::algos::utils::get_best_count;

/// Computes the LPA
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

/// this runs LPA with multiple seeds to normalize away some of the randomness of the LPA
/// algorithm.  We can run it in parallel and update our embedding without collisions.  Fast,
/// produces reasonably good homophily embeddings.  Another good baseline.
pub fn construct_lpa_embedding(
    graph: &(impl Graph + Send + Sync),
    k: usize,
    passes: usize,
    seed: u64
) -> EmbeddingStore {
    let es = EmbeddingStore::new(graph.len(), k, Distance::Hamming);

    println!("k={},passes={},seed={}", k, passes, seed);
    let work = passes * k;
    let pb = CLProgressBar::new(work as u64, true);
    pb.update_message(|msg| { write!(msg, "Clustering...").expect("Should never hit"); });

    // Compute LPA in parallel
    (0..k).into_par_iter().for_each(|k_idx| {
        let clusters = lpa(graph, passes, seed + k_idx as u64);
        pb.inc(passes as u64);
        clusters.into_iter().enumerate().for_each(|(idx, cluster)| {
            let embedding = es.get_embedding_mut_hogwild(idx);
            embedding[k_idx] = cluster as f32;
        });
    });
    pb.finish();
    es

}

