use rayon::prelude::*;

use crate::embeddings::EmbeddingStore;
use crate::graph::{CDFGraph, CDFtoP};
use crate::progress::CLProgressBar;

/// Embedding Diffuser.  Will diffuse embeddings in the direction 
pub struct EmbeddingDiffuser {
    /// The residual weight
    alpha: f32,

    /// Scale factor to ensure diffusion doesn't oversmooth.
    gamma: f32,

    /// Number of iterations to run the power iteration
    iterations: usize
}

impl EmbeddingDiffuser {
    pub fn compute(
        &self, 
        graph: &impl CDFGraph, 
        es: EmbeddingStore,
        indicator: bool
    ) -> EmbeddingStore {
        
        /// Stubbed
        es

    }
}

