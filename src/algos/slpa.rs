//! Speaker-Listener LPA algorithm.  This allows us to detect overlapping communities and is
//! conveniently parallel.  Another good baseline which is typically a strong performer.
use std::fmt::Write;

use rand::prelude::*;
use rand_distr::{Distribution,Uniform};
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;

use crate::graph::Graph;
use crate::progress::CLProgressBar;
use crate::embeddings::{EmbeddingStore, Distance};
use crate::algos::utils::{get_best_count,Counter};

#[derive(Clone,Copy)]
pub enum ListenerRule {
    Best,
    Probabilistic
}

/// Learns the SLPA algorithm
pub fn construct_slpa_embedding(

    graph: &(impl Graph + Send + Sync),

    // Listener rule
    rule: ListenerRule,

    // Number of passes to run
    t: usize,
    
    // Filter out communities with fewer than threshold occurences
    threshold: usize,
    
    // Size of the memory
    memory_size: usize,

    // Random seed
    seed: u64


) -> EmbeddingStore {
    println!("t: {}, threshold:{}, seed:{}", t, threshold, seed);

    let dims = (memory_size as f32 / threshold as f32).ceil() as usize + 1;
    let mut es = EmbeddingStore::new(graph.len(), dims, Distance::Jaccard);

    // Each node starts in its own cluster
    let mut memory = vec![0usize; graph.len() * memory_size];
    memory.par_iter_mut().chunks(memory_size).enumerate().for_each(|(i, mut mem)| {*mem[0] = i;}); 

    let pb = CLProgressBar::new(t as u64, true);
    pb.update_message(|msg| { write!(msg, "Clustering...").expect("Should never hit"); });

    // Temporary cluster assignment
    let mut pass_memory = vec![0; graph.len()];
    let mut idxs: Vec<_> = (0..graph.len()).collect();
    let mut pass_rng = XorShiftRng::seed_from_u64(seed);
    for pass in 1..t {
        idxs.shuffle(&mut pass_rng);
        // Select a node, look at its 
        idxs.par_iter().zip(pass_memory.par_iter_mut()).for_each(|(node_id, new_cluster)| {

            let mut rng = XorShiftRng::seed_from_u64(seed + pass as u64 + *node_id as u64 + 1);
            
            // Collect a cluster from each of its reports
            let edges = &graph.get_edges(*node_id).0;
            let mut proposed_clusters: Vec<_> = edges.iter().map(|edge_idx| {
                let memory_offset = edge_idx * memory_size;
                
                // Randomly sample one cluster id from range
                let bounds = pass.min(memory_size);
                let idx = Uniform::new(0, bounds).sample(&mut rng);
                memory[memory_offset+idx]
            }).collect();
            
            // Select the "best" cluster based on a rule
            *new_cluster = match rule {
                ListenerRule::Best => {
                    proposed_clusters.par_sort_unstable();
                    get_best_count(&proposed_clusters, &mut rng)
                },
                ListenerRule::Probabilistic => {
                    let idx = Uniform::new(0, proposed_clusters.len()).sample(&mut rng);
                    proposed_clusters[idx]
                }
            };

        });

        // Update entry
        idxs.iter().zip(pass_memory.iter()).for_each(|(node_id, new_cluster)| {
            let offset = node_id * memory_size;
            let mem_idx = pass % memory_size;
            memory[offset+mem_idx] = *new_cluster;
        });
        pb.inc(1);
    }
    pb.finish();

    // Threshold 
    memory.chunks_mut(memory_size).enumerate().for_each(|(node_id, node_clusters)| {
        let embedding = es.get_embedding_mut(node_id);
        embedding.iter_mut().for_each(|v| *v = -1.);
        
        // Get the counts for each cluster id
        node_clusters.sort_unstable();

        Counter::new(node_clusters)
            .filter(|(_cluster, cnt)| *cnt >= threshold)
            .enumerate()
            .for_each(|(idx, (cluster, _))| {
                embedding[idx] = cluster as f32;
            });
    });
    es
}

#[cfg(test)]
mod slpa_tests {
    use super::*;
    use crate::graph::CSR;

    fn build_edges() -> Vec<(usize, usize, f32)> {
        vec![
            (0, 1, 1.),
            (0, 2, 1.),
            (1, 0, 1.),
            (1, 2, 3.),
            (2, 1, 3.),
            (2, 0, 2.5),
            (2, 3, 2.5),
            (3, 2, 2.5),
            (3, 4, 0.5),
            (4, 3, 0.5),
            (4, 5, 0.5),
            (4, 6, 0.5),
            (5, 4, 0.5),
            (5, 6, 0.5),
            (6, 4, 0.5),
            (6, 5, 0.5),
        ]
    }

    #[test]
    fn run_test() {
        let graph = CSR::construct_from_edges(build_edges());
        let es = construct_slpa_embedding(&graph, 10, 0.3, 12345123);
        for node_id in 0..graph.len() {
            println!("{}: {:?}", node_id, es.get_embedding(node_id));
        }
        //panic!();
    }
}
