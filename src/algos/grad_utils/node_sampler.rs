//! Defines Samplers for selecting negatives from the graph.  This is a big over-engineered right
//! now as the intent was to have richer samplers which ended up not being the limiting step.
use std::borrow::Borrow;
use rand::prelude::*;
use rand_distr::{Distribution,Uniform};

use crate::feature_store::FeatureStore;
use crate::graph::{Graph as CGraph,NodeID};

/// We initialize a new sampler for each batch.
pub trait BatchSamplerStrategy {
    type Sampler: NodeSampler;

    fn initialize_batch<
        G: CGraph + Send + Sync,
        T: Borrow<NodeID>>
    (
        &self,
        nodes: &[T],
        graph: &G,
        features: &FeatureStore
    ) -> Self::Sampler;

}

/// Creates a sampler for use within a batch.  If we needed to precompute something, we can do it
/// during the initialization of the sampler within the strategy.
pub trait NodeSampler {
    fn sample_negatives<R: Rng>(
        &self, 
        graph: &impl CGraph,
        anchor: NodeID, 
        negatives: &mut Vec<NodeID>,
        num_negs: usize,
        rng: &mut R); 
}

/// Finds hard negatives through exploration of local graph walks.  It will fill the negatives with
/// both easy negatives and hard negatives.  The take so far is random walks are perhaps too close
/// to being weak positives rather than hard negatives.
pub struct RandomWalkHardStrategy {
    /// Fills 
    num_hard_negatives: usize,
    train_idxs: Vec<NodeID>
}

impl RandomWalkHardStrategy {
   pub fn new(num_hard_negatives: usize, train_idxs: &[NodeID]) -> Self {
        RandomWalkHardStrategy { num_hard_negatives, train_idxs: train_idxs.to_vec() }
    }
}

impl <'a> BatchSamplerStrategy for &'a RandomWalkHardStrategy {
    type Sampler = RandomWalkHardSampler<'a>;

     fn initialize_batch<
        G: CGraph + Send + Sync,
        T: Borrow<NodeID>>
    (
        &self,
        _nodes: &[T],
        _graph: &G,
        _features: &FeatureStore
    ) -> Self::Sampler {
        RandomWalkHardSampler { 
            // Hard coded right now; should be parameterized
            p: 0.25, 
            num_hard_negatives: self.num_hard_negatives,
            train_idxs: self.train_idxs.as_slice()
        }
    }
}

pub struct RandomWalkHardSampler<'a> {
    // Restart probability
    p: f32,
    num_hard_negatives: usize,
    /// Only sample from the train IDs for obvious reasons.
    train_idxs: &'a [NodeID]
}

impl <'a> NodeSampler for RandomWalkHardSampler<'a> {
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

        let dist = Uniform::new(0, self.train_idxs.len());
        while negatives.len() < num_negs {
            negatives.push(self.train_idxs[dist.sample(rng)]);
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

