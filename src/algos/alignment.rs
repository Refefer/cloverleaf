//! The neighborhood alignment algorithm.  This takes node embeddings and shifts them toward the
//! centroid of their neighborhood.  For graphs which have strong homophily, it can help
//! signifciantly improve both transductive and inductive algorithms.  In cases where structure is
//! more important, tends to have a lower effect (although light alignments often are beneficial).
use itertools::Itertools;
use rayon::prelude::*;

use crate::graph::{Graph,NodeID,CDFtoP};
use crate::embeddings::{EmbeddingStore, NodeEmbeddings};
use crate::error::{Error, ensure_aligned};
use crate::io::EmbeddingWriter;
use crate::named_graph::NamedGraph;

/// Neighborhood Aligner.  
pub struct NeighborhoodAligner {
    /// Controls how much to bias toward the neighborhood: 0 means fully align, 1 means fully
    /// ignore.  When None, attempts to dynamically set alpha based on the number of neighbors.
    pub alpha: Option<f32>,

    /// For super nodes with millions of edges, can be expensive.  max_neighbors takes the K
    /// first edges to construct the embedding on.
    pub max_neighbors: Option<usize>
}

impl NeighborhoodAligner {
    /// Nodes [`align_to_disk`](Self::align_to_disk) aligns per parallel chunk when the Python
    /// bindings aren't told.
    pub const DEFAULT_CHUNK_SIZE: usize = 10_000;

    pub fn new(alpha: Option<f32>, max_neighbors: Option<usize>) -> Self {
        NeighborhoodAligner { alpha, max_neighbors }
    }

    /// Aligns every node's embedding toward its neighbourhood in `graph`.  The embeddings must
    /// be indexed by the graph's nodes.
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

    pub fn align(
        &self,
        graph: &impl Graph,
        embeddings: &EmbeddingStore, 
        node: NodeID,
        new_emb: &mut [f32]
    ) {
        let (edges, weights) = graph.get_edges(node);

        // No neighborhood to align toward: keep the original embedding.
        if edges.is_empty() {
            new_emb.copy_from_slice(embeddings.get_embedding(node));
            return
        }

        // Convert CDF into probabilities
        let p = CDFtoP::new(weights);

        // limit by max neighbors
        let emb_set = edges.iter().zip(p);
        let it: Box<dyn Iterator<Item=_>> = if let Some(n) = self.max_neighbors {
            Box::new(emb_set.take(n))
        } else {
            Box::new(emb_set)
        };

        // Create a weighted average from all edge nodes,
        // weighted by p.
        it.for_each(|(out_node, weight)| {
            let e = embeddings.get_embedding(*out_node);
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
}
