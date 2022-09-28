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
            Reweighter::reweight_by_distance(results, &distances, self.alpha);
        }
    }

    #[inline]
    fn sigmoid(x: f32) -> f32 {
        1. / (1. + (-x).exp())
    }

    fn reweight_by_distance(results: &mut HashMap<NodeID, f32>, distances: &HashMap<NodeID,f32>, alpha: f32) {
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
            //*wi = (1. - alpha) * *wi + alpha * p;
            *wi *= (p).powf(alpha);
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
mod reweighter_tests {
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

    fn build_counts() -> HashMap<usize, f32>{
        let mut hm = HashMap::new();
        hm.insert(0, 1.);
        hm.insert(1, 2.);
        hm
    }

    #[test]
    fn test_compute_stats() {
        let mut hm = build_counts();

        let (mu, sigma) = Reweighter::compute_stats(&hm);
        assert_eq!(mu, 1.5);
        assert_eq!(sigma, (0.5 / 2f32).sqrt());
    }

    #[test]
    fn test_reweight() {
        let mut counts = build_counts();

        let mut distances = HashMap::new();
        distances.insert(0, 0.2);
        distances.insert(1, 0.5);

        // Distance doesn't matter
        Reweighter::reweight_by_distance(&mut counts, &distances, 0.);
        assert_eq!(counts[&0], 1.);
        assert_eq!(counts[&1], 2.);
        
        // Distance only matters
        Reweighter::reweight_by_distance(&mut counts, &distances, 1.);
        let mut counts = counts.into_iter().collect::<Vec<_>>();
        counts.sort_by_key(|(_, w) | FloatOrd(-*w));
        println!("{:?}", counts);
        assert_eq!(counts[0].0, 0);
        assert_eq!(counts[1].0, 1);

    }

}
