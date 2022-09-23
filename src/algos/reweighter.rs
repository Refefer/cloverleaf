use hashbrown::HashMap;
use rayon::prelude::*;

use crate::graph::NodeID;
use crate::embeddings::EmbeddingStore;

pub struct Reweighter {
    pub alpha: f32
}

impl Reweighter {
    pub fn new(alpha: f32) -> Self {
        Reweighter { alpha }
    }

    pub fn reweight(
        &self, 
        results: &mut HashMap<NodeID, f32>,
        embeddings: &EmbeddingStore,
        context_node: NodeID
    ) {

        // Compute distances for each item
        let mut distances: HashMap<_,_> = results.par_keys()
            .filter(|node| embeddings.is_set(**node))
            .map(|node| {
                let distance = embeddings.compute_distance(context_node, *node);
                (*node, distance)
            }).collect();

        if distances.len() > 2 {
            self.reweight_by_distance(results, &distances);
        }
    }

    #[inline]
    fn sigmoid(x: f32) -> f32 {
        1. / (1. + (-x).exp())
    }

    fn reweight_by_distance(&self, results: &mut HashMap<NodeID, f32>, distances: &HashMap<NodeID,f32>) {
        // Z Normalize the values to a unit Normal, then run it through a sigmoid
        // transform (pretending it's a logistic distribution) to convert to probabilities.
        // In cases where an embedding is missing, we set the distance to the expected value
        let (mu, sigma) = Reweighter::compute_stats(&distances);

        results.par_iter_mut().for_each(|(k, wi)| {
            let p = if let Some(d) = distances.get(k) {
                let nd = (d - mu) / sigma;
                // Lower is better!
                1. - Reweighter::sigmoid(nd)
            } else {
                0.5
            };
            *wi *= p.powf(self.alpha);
        });

    }

    fn compute_stats(distances: &HashMap<NodeID,f32>) -> (f32, f32) {
        let n = distances.len() as f32;

        // Z Normalize the values to a unit Normal, then run it through a sigmoid
        // transform (pretending it's a logistic distribution) to convert to probabilities.
        // In cases where an embedding is missing, we set the distance to the expected value
        let mu = distances.par_values().sum::<f32>() / n;

        let ss = distances.par_values()
            .map(|d| (*d - mu).powf(2.))
            .sum::<f32>();

        let sigma = (ss / n).sqrt();
        (mu, if sigma > 0. { sigma } else {1.})
    }
}

#[cfg(test)]
mod rwr_tests {
    use super::*;
    use float_ord::FloatOrd;

    fn build_edges() -> Vec<(usize, usize, f32)> {
        vec![
            (0, 1, 1.),
            (1, 1, 3.),
            (1, 2, 2.),
            (2, 1, 0.5),
            (1, 0, 10.),
        ]
    }

    #[test]
    fn test_compute_stats() {
        let mut hm = HashMap::new();
        hm.insert(0, 1.);
        hm.insert(1, 2.);
    }

}
