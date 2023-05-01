use crate::graph::{Graph,NodeID};
use crate::embeddings::EmbeddingStore;

pub struct NeighborhoodAligner {
    alpha: Option<f32>,
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
            (1f32 / ((degree + 1f32).ln())).min(1f32)
        };

        new_emb.iter_mut().zip(orig_emb.iter()).for_each(|(nwi, owi)| {
            *nwi = *nwi * (1f32 - alpha) + *owi * alpha;
        });
    }

}
