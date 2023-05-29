//! The neighborhood alignment algorithm.  This takes node embeddings and shifts them toward the
//! centroid of their neighborhood.  For graphs which have strong homophily, it can help
//! signifciantly improve both transductive and inductive algorithms.  In cases where structure is
//! more important, tends to have a lower effect (although light alignments often are beneficial).
use crate::graph::{Graph,NodeID};
use crate::embeddings::EmbeddingStore;

/// Neighborhood Aligner.  
pub struct NeighborhoodAligner {
    /// Controls how much to bias toward the neighborhood: 0 means fully align, 1 means fully
    /// ignore.  When None, attempts to dynamically set alpha based on the number of neighbors.
    alpha: Option<f32>,

    /// For super nodes with millions of edges, can be expensive.  max_neighbors takes the K
    /// first edges to construct the embedding on.
    max_neighbors: Option<usize>
}

impl NeighborhoodAligner {

    pub fn new(alpha: Option<f32>, max_neighbors: Option<usize>) -> Self {
        NeighborhoodAligner { alpha, max_neighbors }
    }

    pub fn align(
        &self,
        graph: &impl Graph,
        embeddings: &EmbeddingStore, 
        node: NodeID,
        new_emb: &mut [f32]
    ) {
        let (edges, weights) = graph.get_edges(node);
        
        // Convert CSR into probabilities
        // TODO: Use the CDFtoP struct
        let p = std::iter::once(weights[0])
            .chain(weights.windows(2).map(|arr| arr[1] - arr[0]));

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
