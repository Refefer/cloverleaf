//! Classic label propagation algorithm for learning clusters based on the graph.  Not much to say;
//! it's single threaded, fast, and a bit finicky on the number of passes (overfitting can produce
//! worse clusters), but it's a good baseline.
use std::fmt::Write;

use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;

use crate::progress::CLProgressBar;
use crate::graph::Graph;
use crate::embeddings::{EmbeddingStore, NodeEmbeddings};
use crate::distance::Distance;
use crate::error::Error;
use crate::named_graph::NamedGraph;
use crate::algos::utils::get_best_count;

/// Embeds each node by its cluster in `k` runs of LPA of `passes` passes each, run `i` seeded
/// with `seed + i`, for [`Distance::Hamming`].
#[derive(Copy,Clone,Debug)]
pub struct ClusterLpa {
    pub k: usize,
    pub passes: usize,
    pub seed: u64
}

impl ClusterLpa {
    /// Runs with the default seed.
    pub fn new(k: usize, passes: usize) -> Self {
        ClusterLpa { k, passes, seed: crate::DEFAULT_SEED }
    }

    /// Embeds every node of `graph`.
    pub fn learn(&self, graph: &NamedGraph) -> Result<NodeEmbeddings, Error> {
        let es = construct_lpa_embedding(graph.graph(), self.k, self.passes, self.seed);
        NodeEmbeddings::new(graph.vocab().clone(), es)
    }
}

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
            // Nothing to listen to; keep the current label.
            if n == 0 { continue }
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


#[cfg(test)]
mod lpa_tests {
    use super::*;
    use crate::graph::CSR;

    #[test]
    fn test_cluster_lpa_learns_named() {
        let mut gb = crate::named_graph::GraphBuilder::new();
        for (a, b) in [("a", "b"), ("b", "c")] {
            gb.add_edge(("n", a), ("n", b), 1., crate::graph::EdgeType::Undirected);
        }
        let g = gb.build(true).unwrap();
        let lpa = ClusterLpa { seed: 7, ..ClusterLpa::new(3, 2) };
        let ne = lpa.learn(&g).unwrap();
        assert_eq!((ne.len(), ne.dims()), (3, 3));
        assert!(matches!(ne.distance(), Distance::Hamming));
        let es = construct_lpa_embedding(g.graph(), 3, 2, 7);
        assert_eq!(ne.store().get_embedding(1), es.get_embedding(1));
    }

    /// Node 2 has an inbound edge but no outbound edges.
    #[test]
    fn test_lpa_handles_sink_nodes() {
        let graph = CSR::construct_from_edges(vec![(0, 1, 1.), (1, 0, 1.), (0, 2, 1.)], false);
        let clusters = lpa(&graph, 3, 1234);
        assert_eq!(clusters.len(), 3);
        // With no neighbors to listen to, a sink keeps its own label.
        assert_eq!(clusters[2], 2);
    }
}
