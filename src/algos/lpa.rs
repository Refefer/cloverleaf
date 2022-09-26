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

fn lpa(
    graph: &impl Graph,
    passes: usize,
    seed: u64
) -> Vec<usize> {
    let mut rng = XorShiftRng::seed_from_u64(seed);
    let mut clusters: Vec<_> = (0..graph.len()).collect();
    let mut idxs = clusters.clone();
    for i_pass in 0..passes {
        idxs.shuffle(&mut rng);
        for idx in idxs.iter() {
            let mut counts = HashMap::new();
            let mut ties = Vec::new();
            
            // Count neighbors
            let node_edges = graph.get_edges(*idx).0;
            for t_node in node_edges.iter() {
                let e = counts.entry(clusters[*t_node]).or_insert(0);
                *e += 1;
            }

            // Get the best ones.  If there are multiple ties, tie break.
            ties.clear();
			let mut best_count = 0;
			for (cluster, count) in counts.iter() {
				if *count > best_count {
					best_count = *count;
					ties.clear();
					ties.push(*cluster)
				} else if *count == best_count {
					ties.push(*cluster)
				}
			}
            
            // We tie break by randomly choosing an item
            clusters[*idx] = if ties.len() > 1 {
                *ties.as_slice()
                    .choose(&mut rng)
                    .expect("If a node has no edges, code bug")
            } else {
                ties[0]
            };

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
