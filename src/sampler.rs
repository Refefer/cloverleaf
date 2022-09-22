use rand::prelude::*;
use rand_distr::{Distribution,Uniform};
use rand_xorshift::XorShiftRng;
use float_ord::FloatOrd;

use crate::graph::{CDFGraph,Graph,NodeID};

pub struct WeightedSample; 

impl WeightedSample {
    pub fn sample<G:CDFGraph, R:Rng>(g: &G, node: NodeID, rng: &mut R) -> Option<NodeID> {
        let (edges, weights) = g.get_edges(node);
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

pub struct UniformSample;

impl UniformSample {
    pub fn sample<G: Graph, R:Rng>(g: &G, node: NodeID, rng: &mut R) -> Option<NodeID> {
        let (edges, weights) = g.get_edges(node);
        if edges.len() == 0 {
            return None
        }
        
        let dist = Uniform::new(0, edges.len());
        let idx = rng.sample(dist);
        Some(edges[idx])
    }
}
