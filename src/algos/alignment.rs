//! The neighborhood alignment algorithm.  This takes node embeddings and shifts them toward the
//! centroid of their neighborhood.  For graphs which have strong homophily, it can help
//! signifciantly improve both transductive and inductive algorithms.  In cases where structure is
//! more important, tends to have a lower effect (although light alignments often are beneficial).
use itertools::Itertools;
use rayon::prelude::*;

use float_ord::FloatOrd;

use crate::graph::{Graph,NodeID,CDFtoP,CDFGraph};
use crate::embeddings::{EmbeddingStore, NodeEmbeddings};
use crate::error::{Error, ensure_aligned};
use crate::io::EmbeddingWriter;
use crate::named_graph::NamedGraph;
use crate::algos::rwr::RWR;
use crate::algos::utils::Sample;

/// Chooses the nodes a node is aligned toward, and their weights.  Implement it to align over
/// other neighbourhoods.
pub trait NeighborhoodSampler: Send + Sync {
    /// Appends the nodes `node` is aligned toward to `out`, with weights summing to 1; nothing
    /// when it has none, which leaves its embedding as it is.
    fn neighbours<G: Graph + CDFGraph + Send + Sync>(&self, graph: &G, node: NodeID, out: &mut Vec<(NodeID, f32)>);
}

/// A node's direct neighbours, weighted by edge weight.  For super nodes with millions of edges,
/// `max_neighbors` keeps the first K, reweighted to sum to 1.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct EdgeNeighbors {
    pub max_neighbors: Option<usize>
}

impl NeighborhoodSampler for EdgeNeighbors {
    fn neighbours<G: Graph + CDFGraph + Send + Sync>(&self, graph: &G, node: NodeID, out: &mut Vec<(NodeID, f32)>) {
        let (edges, weights) = graph.get_edges(node);
        // Edge weights are stored as a CDF; convert them to probabilities.
        let start = out.len();
        out.extend(edges.iter().copied().zip(CDFtoP::new(weights)).take(self.max_neighbors.unwrap_or(usize::MAX)));
        normalise(&mut out[start..]);
    }
}

/// A node's personalised PageRank neighbourhood: `walks` random walks from it, following edges
/// by weight, either of a fixed length (1 is its direct neighbours, 2 adds theirs, and so on) or
/// stopping after each step with a restart probability.  Each node a walk ends on scores
/// count / (walks * degree^beta) (beta 0: plain PPR; higher discounts popular nodes); the node
/// itself is left out, and its top `k` are weighted by score.  Computed per node as it's
/// aligned, so nothing is stored per node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PprNeighbors {
    steps: Sample,
    k: usize,
    walks: usize,
    beta: f32,
    seed: u64
}

impl PprNeighbors {
    pub const DEFAULT_K: usize = 30;
    pub const DEFAULT_WALKS: usize = 1000;
    pub const DEFAULT_BETA: f32 = 0.0;

    /// `steps` >= 1 is a fixed walk length (whole steps), 0 < `steps` < 1 a restart
    /// probability; `k` and `walks` at least 1; `beta` finite and non-negative.
    pub fn new(steps: f32, k: usize, walks: usize, beta: f32, seed: u64) -> Result<Self, Error> {
        let bad_steps = || Error::InvalidArgument(format!(
            "steps must be a whole walk length >= 1 or a restart probability in (0, 1), got {}", steps));
        if !steps.is_finite() || (steps >= 1. && steps.fract() != 0.) {
            return Err(bad_steps())
        }
        let steps = Sample::steps(steps).map_err(|_| bad_steps())?;
        if k == 0 || walks == 0 {
            return Err(Error::InvalidArgument(format!("k and walks must be at least 1, got {} and {}", k, walks)))
        }
        if !(beta.is_finite() && beta >= 0.) {
            return Err(Error::InvalidArgument(format!("beta must be finite and non-negative, got {}", beta)))
        }
        Ok(PprNeighbors { steps, k, walks, beta, seed })
    }

    pub fn steps(&self) -> Sample { self.steps }
    pub fn k(&self) -> usize { self.k }
    pub fn walks(&self) -> usize { self.walks }
    pub fn beta(&self) -> f32 { self.beta }
    pub fn seed(&self) -> u64 { self.seed }
}

impl NeighborhoodSampler for PprNeighbors {
    fn neighbours<G: Graph + CDFGraph + Send + Sync>(&self, graph: &G, node: NodeID, out: &mut Vec<(NodeID, f32)>) {
        // Single threaded: nodes are aligned in parallel already.
        let rwr = RWR { steps: self.steps, walks: self.walks, beta: self.beta, single_threaded: true,
                        seed: self.seed };
        let mut scores: Vec<(NodeID, f32)> = rwr.walk_scores(graph, node, true).into_iter()
            .filter(|(n, s)| *n != node && *s > 0.)
            .collect();
        // Ties broken by node id, so the neighbourhood doesn't depend on hash map order.
        scores.sort_by_key(|(n, s)| (FloatOrd(-*s), *n));
        scores.truncate(self.k);
        normalise(&mut scores);
        out.extend(scores);
    }
}

/// Either built-in sampler, chosen at run time (as the Python bindings do).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AnyNeighbors {
    Edges(EdgeNeighbors),
    Ppr(PprNeighbors)
}

impl NeighborhoodSampler for AnyNeighbors {
    fn neighbours<G: Graph + CDFGraph + Send + Sync>(&self, graph: &G, node: NodeID, out: &mut Vec<(NodeID, f32)>) {
        match self {
            AnyNeighbors::Edges(s) => s.neighbours(graph, node, out),
            AnyNeighbors::Ppr(s) => s.neighbours(graph, node, out)
        }
    }
}

/// Scales the weights to sum to 1 (leaving all-zero weights alone).
fn normalise(weighted: &mut [(NodeID, f32)]) {
    let total: f32 = weighted.iter().map(|(_, w)| *w).sum();
    if total > 0. {
        weighted.iter_mut().for_each(|(_, w)| *w /= total);
    }
}

/// Neighborhood Aligner: each node becomes alpha * itself + (1 - alpha) * the weighted mean of
/// the neighbourhood its sampler chooses (by default, its direct neighbours).
pub struct NeighborhoodAligner<S: NeighborhoodSampler = EdgeNeighbors> {
    /// Controls how much to bias toward the neighborhood: 0 means fully align, 1 means fully
    /// ignore.  When None, attempts to dynamically set alpha based on the number of neighbors.
    pub alpha: Option<f32>,

    /// Which nodes each node is aligned toward.
    pub sampler: S
}

impl NeighborhoodAligner<EdgeNeighbors> {
    /// Aligns toward direct neighbours.
    pub fn new(alpha: Option<f32>, max_neighbors: Option<usize>) -> Self {
        NeighborhoodAligner { alpha, sampler: EdgeNeighbors { max_neighbors } }
    }
}

impl <S: NeighborhoodSampler> NeighborhoodAligner<S> {
    /// Nodes [`align_to_disk`](Self::align_to_disk) aligns per parallel chunk when the Python
    /// bindings aren't told.
    pub const DEFAULT_CHUNK_SIZE: usize = 10_000;

    /// Aligns toward the neighbourhoods `sampler` chooses.
    pub fn with_sampler(alpha: Option<f32>, sampler: S) -> Self {
        NeighborhoodAligner { alpha, sampler }
    }

    pub fn align_all(&self, embeddings: &NodeEmbeddings, graph: &NamedGraph) -> Result<NodeEmbeddings, Error> {
        ensure_aligned(&graph.vocab, &embeddings.vocab, "Graph and NodeEmbeddings")?;
        let es = &embeddings.embeddings;
        let aligned = EmbeddingStore::new(es.len(), es.dims(), es.distance());
        (0..graph.nodes()).into_par_iter().for_each(|node| {
            // Each node writes only its own row.
            self.align(graph.graph(), es, node, aligned.get_embedding_mut_hogwild(node));
        });
        Ok(NodeEmbeddings { vocab: embeddings.vocab.clone(), embeddings: aligned })
    }

    /// `align_all`, streaming the aligned embeddings to `path` `chunk_size` nodes at a time
    /// instead of holding them all in memory.
    pub fn align_to_disk(
        &self,
        path: &str,
        embeddings: &NodeEmbeddings,
        graph: &NamedGraph,
        chunk_size: usize,
        comp_level: Option<u32>
    ) -> Result<(), Error> {
        ensure_aligned(&graph.vocab, &embeddings.vocab, "Graph and NodeEmbeddings")?;
        if chunk_size == 0 {
            return Err(Error::InvalidArgument("chunk_size must be at least 1".into()))
        }
        let mut writer = EmbeddingWriter::new(path, embeddings.vocab.as_ref(), comp_level)?;
        let mut buffer = vec![vec![0.; embeddings.dims()]; chunk_size];
        let mut ids = Vec::with_capacity(chunk_size);
        for chunk in &(0..graph.nodes()).chunks(chunk_size) {
            ids.clear();
            ids.extend(chunk);
            ids.par_iter().zip(buffer.par_iter_mut()).for_each(|(node, new_emb)| {
                new_emb.iter_mut().for_each(|wi| *wi = 0f32);
                self.align(graph.graph(), &embeddings.embeddings, *node, new_emb);
            });
            writer.stream(ids.iter().copied().zip(buffer.iter()))?;
        }
        writer.finish()?;
        Ok(())
    }

    pub fn align<G: Graph + CDFGraph + Send + Sync>(
        &self,
        graph: &G,
        embeddings: &EmbeddingStore, 
        node: NodeID,
        new_emb: &mut [f32]
    ) {
        let edges = graph.get_edges(node).0;

        // No neighborhood to align toward: keep the original embedding.
        if edges.is_empty() {
            new_emb.copy_from_slice(embeddings.get_embedding(node));
            return
        }

        let mut neighbours = Vec::new();
        self.sampler.neighbours(graph, node, &mut neighbours);
        if neighbours.is_empty() {
            new_emb.copy_from_slice(embeddings.get_embedding(node));
            return
        }

        // A weighted average over the neighbourhood.
        new_emb.iter_mut().for_each(|wi| *wi = 0.);
        neighbours.into_iter().for_each(|(out_node, weight)| {
            let e = embeddings.get_embedding(out_node);
            new_emb.iter_mut().zip(e.iter()).for_each(|(wi, ei)| {
                *wi += weight * ei;
            });
        });

        // Blend it with the original node, using alpha
        let orig_emb = embeddings.get_embedding(node);
        
        let alpha = if let Some(alpha) = self.alpha {
            // Static alpha
            alpha
        } else {
            // Adaptive alpha - we use the degree to determine
            // how much to use
            let degree = edges.len() as f32;
            // TODO: This is a heuristic which hasn't been well tested.  approximations of page
            // rank are a reasonable starting point but need to be further ablated.
            (1f32 / ((degree + 1f32).ln())).min(1f32)
        };

        // Combine node embedding with neighborhood embedding
        new_emb.iter_mut().zip(orig_emb.iter()).for_each(|(nwi, owi)| {
            *nwi = *nwi * (1f32 - alpha) + *owi * alpha;
        });
    }

}

#[cfg(test)]
mod alignment_tests {
    use super::*;
    use crate::graph::{CSR, CumCSR};
    use crate::distance::Distance;

    /// A node with no out edges has no neighborhood, so it keeps its embedding.
    #[test]
    fn test_align_sink_node_keeps_embedding() {
        let graph = CumCSR::convert(
            CSR::construct_from_edges(vec![(0, 1, 1.), (1, 0, 1.), (0, 2, 1.)], false));
        let es = EmbeddingStore::new(3, 2, Distance::Cosine);
        for node in 0..3 {
            let e = es.get_embedding_mut_hogwild(node);
            e[0] = node as f32 + 1.;
            e[1] = -(node as f32);
        }
        for alpha in [None, Some(0.5)] {
            let aligner = NeighborhoodAligner::new(alpha, None);
            let mut out = vec![0f32; 2];
            aligner.align(&graph, &es, 2, &mut out);
            assert_eq!(out, es.get_embedding(2).to_vec(), "alpha={:?}", alpha);
        }
    }

    /// A path 0 - 1 - 2 - 3 - 4, both directions, and node i's embedding [i, 0].
    fn path() -> (CumCSR, EmbeddingStore) {
        let mut edges = Vec::new();
        for i in 0..4 {
            edges.push((i, i + 1, 1.));
            edges.push((i + 1, i, 1.));
        }
        let graph = CumCSR::convert(CSR::construct_from_edges(edges, false));
        let es = EmbeddingStore::new(5, 2, Distance::Euclidean);
        for node in 0..5 {
            es.get_embedding_mut_hogwild(node)[0] = node as f32;
        }
        (graph, es)
    }

    fn sampled(s: &impl NeighborhoodSampler, graph: &CumCSR, node: NodeID) -> Vec<(NodeID, f32)> {
        let mut out = Vec::new();
        s.neighbours(graph, node, &mut out);
        out.sort_by_key(|(n, _)| *n);
        out
    }

    /// One-step walks recover the direct neighbours at their edge weights.
    #[test]
    fn test_ppr_one_step_is_the_direct_neighbourhood() {
        let (graph, _) = path();
        let one = PprNeighbors::new(1., 30, 20_000, 0., 7).unwrap();
        let got = sampled(&one, &graph, 2);
        assert_eq!(got.iter().map(|(n, _)| *n).collect::<Vec<_>>(), vec![1, 3]);
        assert!(got.iter().all(|(_, w)| (w - 0.5).abs() < 0.02), "{:?}", got);
        assert_eq!(sampled(&EdgeNeighbors::default(), &graph, 2), vec![(1, 0.5), (3, 0.5)]);
    }

    /// Two steps reach the nodes two hops away and never the node itself; k keeps the heaviest,
    /// reweighted to sum to 1.
    #[test]
    fn test_ppr_two_steps_and_top_k() {
        let (graph, _) = path();
        let two = PprNeighbors::new(2., 30, 20_000, 0., 7).unwrap();
        let got = sampled(&two, &graph, 2);
        // From 2, two-step walks end on 0, 2 (left out) or 4, a quarter each way.
        assert_eq!(got.iter().map(|(n, _)| *n).collect::<Vec<_>>(), vec![0, 4]);
        assert!((got.iter().map(|(_, w)| w).sum::<f32>() - 1.).abs() < 1e-5);
        // Restarts mix distances; k = 1 keeps one node at weight 1.
        let top = sampled(&PprNeighbors::new(0.5, 1, 5_000, 0., 7).unwrap(), &graph, 0);
        assert_eq!(top, vec![(1, 1.)]);
    }

    #[test]
    fn test_ppr_rejects_bad_settings() {
        for (steps, k, walks, beta) in [(0., 5, 10, 0.), (2.5, 5, 10, 0.), (f32::NAN, 5, 10, 0.),
                                        (f32::INFINITY, 5, 10, 0.), (2., 0, 10, 0.), (2., 5, 0, 0.),
                                        (2., 5, 10, -1.), (2., 5, 10, f32::NAN)] {
            assert!(PprNeighbors::new(steps, k, walks, beta, 1).is_err(), "{} {} {} {}", steps, k, walks, beta);
        }
        assert!(PprNeighbors::new(3., 5, 10, 0.5, 1).is_ok());
        assert!(PprNeighbors::new(0.3, 5, 10, 0., 1).is_ok());
    }

    /// Keeping the first K edges reweights them to sum to 1, so the node isn't shrunk.
    #[test]
    fn test_max_neighbors_reweights() {
        let (graph, es) = path();
        assert_eq!(sampled(&EdgeNeighbors { max_neighbors: Some(1) }, &graph, 2).len(), 1);
        let aligner = NeighborhoodAligner::new(Some(0.), Some(1));
        let mut out = vec![0f32; 2];
        aligner.align(&graph, &es, 2, &mut out);
        assert!(out[0] == 1. || out[0] == 3., "the one neighbour's embedding, got {:?}", out);
    }

    /// Alignment over a PPR neighbourhood blends the node with its neighbours' weighted mean.
    #[test]
    fn test_align_with_ppr_sampler() {
        let (graph, es) = path();
        let aligner = NeighborhoodAligner::with_sampler(Some(0.5), PprNeighbors::new(2., 30, 20_000, 0., 7).unwrap());
        let mut out = vec![0f32; 2];
        aligner.align(&graph, &es, 2, &mut out);
        // 0.5 * 2 + 0.5 * mean(0, 4) = 2.
        assert!((out[0] - 2.).abs() < 0.05, "{:?}", out);
        aligner.align(&graph, &es, 0, &mut out);
        // From 0, two steps end on 0 (left out) or 2: 0.5 * 0 + 0.5 * 2 = 1.
        assert!((out[0] - 1.).abs() < 1e-5, "{:?}", out);
    }
}
