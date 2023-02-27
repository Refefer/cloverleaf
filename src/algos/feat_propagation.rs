use hashbrown::HashMap;
use float_ord::FloatOrd;

use crate::graph::{Graph,NodeID};
use crate::algos::utils::FeatureStore;
use crate::bitset::BitSet;

pub fn propagate_features(
    graph: &(impl Graph + Send + Sync),
    features: &mut FeatureStore,
    max_iters: usize,
    k: usize,
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

    for iter in 0..max_iters {
        let mut working_map = HashMap::new();
        let mut working_vec = Vec::new();
        let mut is_done = true;
        for node_id in 0..features.num_nodes() {
            if is_propagated.is_set(node_id) { continue }

            let (edges, weights) = graph.get_edges(node_id);
            working_map.clear();
            let mut all_propagated = true;
            // Reconstructs the probability distribution
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

            // L2norm
            working_vec.clear();
            working_vec.extend(working_map.drain());
            working_vec.sort_by_key(|(_, w)| FloatOrd(-*w));
            let norm = working_vec.iter()
                .map(|(_, w)| w.powf(2.))
                .sum::<f32>().sqrt();

            let top_k = working_vec.drain(..)
                .filter(|(_, w)| *w / norm > threshold)
                .take(k)
                .map(|(f, _)| f);

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
