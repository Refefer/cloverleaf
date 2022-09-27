use hashbrown::HashMap;
use rand::prelude::*;
use rand_distr::{Distribution,Uniform};
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;

use crate::graph::{Graph,NodeID};

use crate::embeddings::{EmbeddingStore, Distance};
use crate::algos::utils::{get_best_count,Counter};

pub fn construct_slpa_embedding(
    graph: &(impl Graph + Send + Sync),
    k: usize,
    threshold: f32,
    seed: u64
) -> EmbeddingStore {
    let dims = (k / (k as f32 * threshold).ceil() as usize) + 1;
    let mut es = EmbeddingStore::new(graph.len(), dims, Distance::Jaccard);

    let mut clusters = vec![0usize; graph.len() * k];
    for i in 0..graph.len() {
        clusters[i*k] = i;
    }

    let mut rng = XorShiftRng::seed_from_u64(seed);

    let mut buffer = vec![0; graph.len()];
    for pass in 1..k {
        eprintln!("Pass {}/{}", pass, k);

        // Select a node, look at its 
        (0..graph.len()).into_par_iter().zip(buffer.par_iter_mut()).for_each(|(node_id, new_cluster)| {

            let mut rng = XorShiftRng::seed_from_u64(seed + pass as u64 + node_id as u64);
            
            // Collect a cluster from each of its reports
            let edges = &graph.get_edges(node_id).0;
            let mut proposed_clusters: Vec<_> = edges.iter().map(|idx| {
                let offset = idx * k;
                
                // Randomly sample one cluster id from range
                let idx = Uniform::new(0, pass).sample(&mut rng);
                clusters[offset+idx]
            }).collect();
            
            // Select the "best" cluster
            proposed_clusters.par_sort_unstable();
            *new_cluster = get_best_count(&proposed_clusters, &mut rng);
        });

        // Update entry
        buffer.par_iter().zip(clusters.par_iter_mut().chunks(k)).for_each(|(cluster_id, mut emb)| {
            *emb[pass] = *cluster_id;
        });
    }

    // Threshold is the l1norm score
    let min_count = (threshold * k as f32).ceil() as usize;
    println!("Min Count: {}", min_count);
    clusters.chunks_mut(k).enumerate().for_each(|(node_id, node_clusters)| {
        let embedding = es.get_embedding_mut(node_id);
        embedding.iter_mut().for_each(|v| *v = -1.);
        
        // Get the counts for each cluster id
        node_clusters.sort_unstable();

        Counter::new(node_clusters)
            .filter(|(cluster, cnt)| *cnt >= min_count)
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
