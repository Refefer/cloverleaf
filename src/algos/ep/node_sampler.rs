use float_ord::FloatOrd;
use rand::prelude::*;
use rand_distr::{Distribution,Uniform};

use crate::feature_store::FeatureStore;
use crate::graph::{Graph as CGraph,NodeID};

pub trait BatchSamplerStrategy {
    type Sampler: NodeSampler;

    fn initialize_batch<G: CGraph + Send + Sync>(
        &self,
        nodes: &[&NodeID],
        graph: &G,
        features: &FeatureStore
    ) -> Self::Sampler;

}

pub trait NodeSampler {
    fn sample_negatives<R: Rng>(
        &self, 
        graph: &impl CGraph,
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
        _nodes: &[&NodeID],
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
        _graph: &impl CGraph,
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

// Finds hard negatives through exploration of local graph walks
pub struct RandomWalkHardStrategy {
    num_hard_negatives: usize
}

impl RandomWalkHardStrategy {
   pub fn new(num_hard_negatives: usize) -> Self {
        RandomWalkHardStrategy { num_hard_negatives }
    }
}

impl BatchSamplerStrategy for &RandomWalkHardStrategy {
    type Sampler = RandomWalkHardSampler;

    fn initialize_batch<'b,G: CGraph + Send + Sync>(
        &self,
        _nodes: &[&NodeID],
        _graph: &G,
        _features: &FeatureStore
    ) -> Self::Sampler {
        RandomWalkHardSampler { 
            p: 0.25, 
            num_hard_negatives: self.num_hard_negatives 
        }
    }
}

pub struct RandomWalkHardSampler {
    p: f32,
    num_hard_negatives: usize
}

impl NodeSampler for RandomWalkHardSampler {
    fn sample_negatives<R: Rng>(
        &self, 
        graph: &impl CGraph,
        anchor: NodeID, 
        negatives: &mut Vec<NodeID>,
        num_negs: usize,
        rng: &mut R
    ) {
        let num_hard_negs = self.num_hard_negatives.min(num_negs);
        // Try filling with hard negs first
        for _ in 0..(num_hard_negs * 2) {
            if let Some(node) = random_walk(anchor, graph, rng, self.p, 10) {
                if !negatives.contains(&node) {
                    negatives.push(node);
                }
            }
            if negatives.len() >= num_hard_negs { break }
        }

        let dist = Uniform::new(0, graph.len());
        while negatives.len() < num_negs {
            negatives.push(dist.sample(rng));
        }
    }
}

fn random_walk<R: Rng, G: CGraph>(
    anchor: NodeID, 
    graph: &G,
    rng: &mut R,
    restart_p: f32,
    max_steps: usize
) -> Option<NodeID> {
    let anchor_edges = graph.get_edges(anchor).0;
    let mut node = anchor;
    let mut i = 0;
    
    // Random walk
    loop {
        i += 1;
        let edges = graph.get_edges(node).0;
        if edges.len() == 0 || i > max_steps {
            break
        }
        let dist = Uniform::new(0, edges.len());
        node = edges[dist.sample(rng)];
        // We want at least two steps in our walk
        // before exiting since 1 step guarantees an anchor
        // edge
        if i > 1 && rng.gen::<f32>() < restart_p && node != anchor { break }
    }

    if node != anchor && !anchor_edges.contains(&node) {
        Some(node)
    } else {
        None
    }
}

