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
///
/// Borrows the node list easy negatives are drawn from instead of copying it: the list has an
/// entry per node, which is gigabytes on large graphs.
pub struct RandomWalkHardStrategy<'a> {
    /// Maximum number of negatives to find with random walks; the rest are drawn uniformly.
    num_hard_negatives: usize,
    train_idxs: &'a [NodeID]
}

impl <'a> RandomWalkHardStrategy<'a> {
   pub fn new(num_hard_negatives: usize, train_idxs: &'a [NodeID]) -> Self {
        RandomWalkHardStrategy { num_hard_negatives, train_idxs }
    }
}

impl <'a, 'b> BatchSamplerStrategy for &'b RandomWalkHardStrategy<'a> {
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
            train_idxs: self.train_idxs
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

        // The anchor is never its own negative, unless there is nothing else to draw.
        let dist = Uniform::new(0, self.train_idxs.len());
        let only_anchor = self.train_idxs.len() == 1 && self.train_idxs[0] == anchor;
        while negatives.len() < num_negs {
            let node = self.train_idxs[dist.sample(rng)];
            if node != anchor || only_anchor {
                negatives.push(node);
            }
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


#[cfg(test)]
mod node_sampler_tests {
    use super::*;

    /// The index lists hold one entry per node, so the strategy must draw from the caller's slice
    /// rather than keep its own copy.
    #[test]
    fn test_strategy_borrows_train_idxs() {
        let idxs: Vec<NodeID> = (0..1000).collect();
        let strategy = RandomWalkHardStrategy::new(0, &idxs);
        assert!(std::ptr::eq(strategy.train_idxs.as_ptr(), idxs.as_ptr()));
    }

    /// A node is never its own negative (the EP paper samples u from V \ {v}).
    #[test]
    fn test_anchor_is_never_its_own_negative() {
        use crate::graph::{CSR, CumCSR};
        use rand_xorshift::XorShiftRng;
        let edges = vec![(0, 1, 1f32), (1, 0, 1f32), (1, 2, 1f32), (2, 1, 1f32)];
        let graph = CumCSR::convert(CSR::construct_from_edges(edges, false));
        let fs = FeatureStore::new(3);
        let idxs: Vec<NodeID> = vec![0, 1, 2];
        let strategy = RandomWalkHardStrategy::new(0, &idxs);
        let sampler = (&strategy).initialize_batch::<_, NodeID>(&[], &graph, &fs);
        let mut rng = XorShiftRng::seed_from_u64(0);
        let mut negatives = Vec::new();
        for _ in 0..1000 {
            negatives.clear();
            sampler.sample_negatives(&graph, 0, &mut negatives, 1, &mut rng);
            assert_ne!(negatives, vec![0]);
        }
    }

    /// With nothing else to draw from, the anchor is still returned rather than looping forever.
    #[test]
    fn test_lone_node_can_be_its_own_negative() {
        use crate::graph::{CSR, CumCSR};
        use rand_xorshift::XorShiftRng;
        let graph = CumCSR::convert(CSR::construct_from_edges(vec![(0, 0, 1f32)], false));
        let fs = FeatureStore::new(1);
        let idxs: Vec<NodeID> = vec![0];
        let strategy = RandomWalkHardStrategy::new(0, &idxs);
        let sampler = (&strategy).initialize_batch::<_, NodeID>(&[], &graph, &fs);
        let mut negatives = Vec::new();
        sampler.sample_negatives(&graph, 0, &mut negatives, 1, &mut XorShiftRng::seed_from_u64(0));
        assert_eq!(negatives, vec![0]);
    }
}
