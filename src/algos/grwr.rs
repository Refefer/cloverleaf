use hashbrown::HashMap;
use rand::prelude::*;
use rand_distr::Distribution;
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;

use crate::graph::{Graph,NodeID};
use crate::sampler::Sampler;
use crate::embeddings::EmbeddingStore;
use crate::algos::reweighter::Reweighter;

pub enum Steps {
    Fixed(usize),
    Probability(f32, usize)
}

impl Steps {
    fn iterations(&self) -> usize {
        match &self {
            Steps::Fixed(iters) => *iters,
            Steps::Probability(_p, iters) => *iters
        }
    }
}

pub struct GuidedRWR {
    pub steps: Steps,
    pub walks: usize,
    pub alpha: f32,
    pub beta: f32,
    pub seed: u64
}

impl GuidedRWR {

    pub fn sample<G: Graph + Send + Sync>(
        &self, 
        graph: &G, 
        sampler: &impl Sampler<G>,
        embeddings: &EmbeddingStore,
        start_node: NodeID,
        context_node: NodeID
    ) -> HashMap<NodeID, f32> {
        let mut counts = HashMap::new();
        let mut next = HashMap::new();
        // Start
        counts.insert(start_node, self.walks as f32);
        let mut rng = XorShiftRng::seed_from_u64(self.seed);
        for _ in 0..self.steps.iterations() {
            next.clear();
            
            // Run a pass, with teleport
            self.bfs_push(graph, sampler, &counts, &mut next, &start_node, &mut rng);

            // Reweight counts
            let reweighter = Reweighter { alpha: self.alpha };
            reweighter.reweight(&mut next, embeddings, context_node);

            // Renormalize counts
            let denom = next.par_values().sum::<f32>();
            next.par_values_mut().for_each(|w| {
                *w = (self.walks as f32 * *w / denom).round()
            });

            // Once more, with gusto!
            std::mem::swap(&mut counts, &mut next);
        }

        // Discount by node degree
        counts.par_iter_mut()
           .for_each(|(k, v)| {
               let d = (graph.degree(*k) as f32).powf(self.beta);
               *v /= (self.walks as f32) * d;
           });
        counts
    }

    fn bfs_push<G:Graph + Send + Sync, R: Rng>(
        &self, 
        graph: &G, 
        sampler: &impl Sampler<G>,
        counts: &HashMap<NodeID, f32>, 
        next: &mut HashMap<NodeID, f32>,
        start_node: &NodeID,
        rng: &mut R
    ) {
        // One pass of BFS
        let base_seed: u64 = rng.gen();
        counts.iter().for_each(|(node, cnt)| {
            // Make stable rng
            let mut new_rng = XorShiftRng::seed_from_u64(base_seed + *node as u64);
            for _ in 0..(*cnt as usize) {
                let next_node = sampler.sample(graph, *node, &mut new_rng).unwrap_or(*start_node);
                *next.entry(next_node).or_insert(0.) += 1.;
            }
        });

        // Compute teleport if restart probability
        if let Steps::Probability(p, _iterations) = self.steps {
            let teleport = next.par_values_mut().map(|cnt| {
                let discount = (*cnt * (1. - p)).floor();
                let teleport = *cnt - discount;
                *cnt = discount;
                teleport
            }).sum::<f32>();
            
            // Update start node
            *next.entry(*start_node).or_insert(0.) += teleport.round();
        }
    }
}

#[cfg(test)]
mod grwr_tests {
    use super::*;
    use crate::graph::{CumCSR,CSR};
    use crate::sampler::{Unweighted, Weighted};
    use crate::embeddings::{EmbeddingStore,Distance};
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
    fn test_grwr_unweighted() {
        let edges = build_edges();

        let csr = CSR::construct_from_edges(edges);
        let ccsr = CumCSR::convert(csr);
        let rwr = GuidedRWR {
            steps: Steps::Fixed(1),
            walks: 10_000,
            alpha: 0.,
            beta: 0.5,
            seed: 20222022
        };

        let es = EmbeddingStore::new(ccsr.len(), 1, Distance::Cosine);
        let candidates = rwr.sample(&ccsr, &Unweighted, &es, 0, 0);
        assert_eq!(candidates.len(), 1);
        assert!(candidates.contains_key(&1));
    }

    #[test]
    fn test_grwr_weighted() {
        let edges = build_edges();

        let csr = CSR::construct_from_edges(edges);
        let ccsr = CumCSR::convert(csr);
        let rwr = GuidedRWR {
            steps: Steps::Fixed(1),
            walks: 10_000,
            alpha: 0.,
            beta: 0.0,
            seed: 20222022
        };

        let es = EmbeddingStore::new(ccsr.len(), 1, Distance::Cosine);

        let mut candidates: Vec<_> = rwr.sample(&ccsr, &Weighted, &es, 1, 1)
            .into_iter().collect();
        candidates.sort_by_key(|(node_id, weight)| FloatOrd(-*weight));

        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[0].0, 0);
        assert_eq!(candidates[1].0, 1);
        assert_eq!(candidates[2].0, 2);
    }

}
