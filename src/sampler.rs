use rand::prelude::*;
use rand::distributions::WeightedIndex;
use rand_distr::{Distribution,Uniform};
use float_ord::FloatOrd;

use crate::graph::{CDFGraph,Graph,NodeID,CSR,NormalizedCSR};

pub trait Sampler<G>: Send + Sync {
    fn sample<R: Rng>(&self, g: &G, node_id: NodeID, rng: &mut R) -> Option<NodeID>;
}

pub struct Weighted; 

impl <S: CDFGraph> Sampler<S> for Weighted {
    fn sample<R: Rng>(&self, g: &S, node_id: NodeID, rng: &mut R) -> Option<NodeID> {
        let (edges, weights) = g.get_edges(node_id);
        if edges.len() == 0 {
            return None
        }

        let p: f32 = rng.gen();
        let idx = match weights.binary_search_by_key(&FloatOrd(p), |w| FloatOrd(*w)) {
            Ok(idx) => idx,
            Err(idx) => idx
        };
        Some(edges[idx])
 
    }
}

fn simple_weighted_sample<R: Rng>(g: &impl Graph, node_id: NodeID, rng: &mut R) -> Option<NodeID> {
    let (edges, weights) = g.get_edges(node_id);
    if edges.len() == 0 {
        return None
    }
    let dist = WeightedIndex::new(weights).unwrap();
    Some(edges[dist.sample(rng)])
}

impl Sampler<CSR> for Weighted {
    fn sample<R: Rng>(&self, g: &CSR, node_id: NodeID, rng: &mut R) -> Option<NodeID> {
        simple_weighted_sample(g, node_id, rng)
    }
}

impl Sampler<NormalizedCSR> for Weighted {
    fn sample<R: Rng>(&self, g: &NormalizedCSR, node_id: NodeID, rng: &mut R) -> Option<NodeID> {
        simple_weighted_sample(g, node_id, rng)
    }
}


pub struct Unweighted;

impl <G: Graph> Sampler<G> for Unweighted {
    fn sample<R:Rng>(&self, g: &G, node: NodeID, rng: &mut R) -> Option<NodeID> {
        let edges = g.get_edges(node).0;
        if edges.len() == 0 {
            return None
        }
        
        let dist = Uniform::new(0, edges.len());
        let idx = rng.sample(dist);
        Some(edges[idx])
    }
}

// Uses a blend of weighted versus unweighted sampling
pub struct GreedySampler(pub f32);

impl <G: CDFGraph> Sampler<G> for GreedySampler {
    fn sample<R:Rng>(&self, g: &G, node: NodeID, rng: &mut R) -> Option<NodeID> {
        if rng.gen::<f32>() < self.0 {
            Unweighted.sample(g, node, rng)
        } else {
            Weighted.sample(g, node, rng)
        }
    }
}
