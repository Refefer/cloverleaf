/// This defines a simple method for propagating features between nodes with features and nodes
/// without features.  It uses a simple sum, filtering method to select discrete features.  This
/// can be helpful when we want to learn smoother embeddings constrained by features instead of
/// learning anonymous node embeddings.
use hashbrown::HashMap;
use float_ord::FloatOrd;

use crate::graph::Graph;
use crate::feature_store::FeatureStore;
use crate::bitset::BitSet;

/// Propagates features into a feature store.
pub fn propagate_features(
    /// Graph to guide propagation with
    graph: &(impl Graph + Send + Sync),
    
    /// Feature Store to propagate from and to
    features: &mut FeatureStore,

    /// Number of passes to propagate
    max_iters: usize,

    /// Max Number of features to propagate to each node
    k: usize,

    /// features that fall under this threshold will be filtered out
    threshold: f32
) {
    let mut is_propagated = BitSet::new(features.num_nodes());
    // We use is_propagated to check if a features has been fully
    // propagated.
    for (i, feats) in features.iter().enumerate() {
        if feats.len() > 0 {
            is_propagated.set_bit(i);
        }
    }

    for _iter in 0..max_iters {
        let mut working_map = HashMap::new();
        let mut working_vec = Vec::new();
        let mut is_done = true;
        for node_id in 0..features.num_nodes() {
            if is_propagated.is_set(node_id) { continue }

            let (edges, weights) = graph.get_edges(node_id);
            working_map.clear();
            let mut all_propagated = true;
            
            // Reconstructs the probability distribution
            // TODO: Replace with CDFtoP struct
            let wit = weights.iter().scan(0f32, |state, &w| {
                let p_x = w - *state;
                *state = w;
                Some(p_x)
            });

            for (edge, weight) in edges.iter().zip(wit) {
                let feats = features.get_features(*edge);
                
                // If all the constituent nodes are propagated, we consider this
                // propagated.
                all_propagated &= feats.len() > 0;
                for feat in feats.into_iter() {
                    let e = working_map.entry(*feat).or_insert(0.);
                    *e += weight;
                }
            }

            // L2norm the features
            working_vec.clear();
            working_vec.extend(working_map.drain());
            working_vec.sort_by_key(|(_, w)| FloatOrd(-*w));
            let norm = working_vec.iter()
                .map(|(_, w)| w.powf(2.))
                .sum::<f32>().sqrt();

            // Take the best K features after thresholding
            let top_k = working_vec.drain(..)
                .filter(|(_, w)| *w / norm > threshold)
                .take(k)
                .map(|(f, _)| f);

            // Update the feature store with the new features
            features.set_features_raw(node_id, top_k);
            if all_propagated { 
                is_propagated.set_bit(node_id) 
            } else {
                is_done = false;
            }
        }
        if is_done {
            break
        }
    }
}

#[cfg(test)]
mod pf_tests {
    use super::*;
    use crate::graph::CSR;

}
