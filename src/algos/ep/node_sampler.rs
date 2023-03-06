use float_ord::FloatOrd;
use rand::prelude::*;

use crate::feature_store::FeatureStore;
use crate::graph::{Graph as CGraph,NodeID};

pub trait BatchSamplerStrategy {
    type Sampler: NodeSampler;

    fn initialize_batch<G: CGraph + Send + Sync>(&self,
        graph: &G,
        features: &FeatureStore
    ) -> Self::Sampler;

}

pub trait NodeSampler {
    fn sample_negatives<R: Rng>(
        &self, 
        anchor: NodeID, 
        negatives: &mut Vec<NodeID>,
        num_negs: usize,
        rng: &mut R); 
}

pub struct RandomSamplerStrategy {
    unigram_table: Vec<f32>
}

impl RandomSamplerStrategy {

    pub fn new<G: CGraph>(
        graph: &G
    ) -> Self {
        RandomSamplerStrategy {
            unigram_table: RandomSamplerStrategy::create_neg_sample_table(graph)
        }
    }

    pub fn create_neg_sample_table<G: CGraph>(
        graph: &G
    ) -> Vec<f32> {
        let mut fast_biased = vec![0f32; graph.len()];
        let denom = (0..graph.len())
            .map(|idx| (graph.degree(idx) as f64).powf(0.75))
            .sum::<f64>();

        let mut acc = 0f64;
        fast_biased.iter_mut().enumerate().for_each(|(i, w)| {
            *w = (acc / denom) as f32;
            acc += (graph.degree(i) as f64).powf(0.75);
        });
        fast_biased[graph.len()-1] = 1.;
        fast_biased
    }
}

impl <'s> BatchSamplerStrategy for &'s RandomSamplerStrategy {
    type Sampler = RandomSampler<'s>;

    fn initialize_batch<'b,G: CGraph + Send + Sync>(
        &self,
        _graph: &G,
        _features: &FeatureStore
    ) -> Self::Sampler {
        RandomSampler { unigram_table: &self.unigram_table }
    }
 
}

pub struct RandomSampler<'a> {
    unigram_table: &'a Vec<f32>
}

impl <'a> NodeSampler for RandomSampler<'a>  {
    fn sample_negatives<R: Rng>(
        &self, 
        anchor: NodeID, 
        negatives: &mut Vec<NodeID>,
        num_negs: usize,
        rng: &mut R) {

        // We make a good attempt to get the full negatives, but bail
        // if it's computationally too expensive
        for _ in 0..(num_negs*2) {
            let p: f32 = rng.gen();
            let neg_node = match self.unigram_table.binary_search_by_key(&FloatOrd(p), |w| FloatOrd(*w)) {
                Ok(idx) => idx,
                Err(idx) => idx
            };

            if neg_node != anchor { 
                negatives.push(neg_node) ;
                if negatives.len() == num_negs { break }
            }
        }
    }

}
