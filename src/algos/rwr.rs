//! Classic Random walk with Restarts.  This uses the Rp3b algorithm to allow biasing toward/away
//! from popular nodes to rarer nodes.  
use hashbrown::HashMap;
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;

use crate::graph::{Graph,NodeID};
use crate::sampler::Sampler;

// Fixed step or random restarts
#[derive(Clone,Copy)]
pub enum Steps {
    /// Every walk is K steps long
    Fixed(usize),
    
    /// A walk ends random with p probability
    Probability(f32)
}

pub struct RWR {
    pub steps: Steps,
    pub walks: usize,
    pub beta: f32,
    pub seed: u64
}

impl RWR {

    pub fn sample<G: Graph + Send + Sync>(
        &self, 
        graph: &G, 
        sampler: &impl Sampler<G>,
        start_node: NodeID
    ) -> HashMap<NodeID, f32> {
       let mut ret = (0..self.walks).into_par_iter()
           .map(|idx| {
               let mut rng = XorShiftRng::seed_from_u64(self.seed + idx as u64);
               self.walk(graph, sampler, start_node, &mut rng) 
           }).fold(|| HashMap::new(), |mut acc, node_id| {
               *acc.entry(node_id).or_insert(0f32) += 1.; 
               acc
           }).reduce(|| HashMap::new(),|mut hm1, hm2| {
               hm2.into_iter().for_each(|(k, v)| {
                   let e = hm1.entry(k).or_insert(0f32); 
                   *e = *e + v;
               });
               hm1
           });

       ret.par_iter_mut()
           .for_each(|(k, v)| {
               let d = (graph.degree(*k) as f32).powf(self.beta);
               *v /= (self.walks as f32) * d;
           });
       ret
    }

    /// Runs a random walk, returning the terminal node.
    pub fn walk<G: Graph + Send + Sync>(
        &self, 
        graph: &G, 
        sampler: &impl Sampler<G>,
        start_node: NodeID,
        rng: &mut impl Rng
    ) -> NodeID {
       let mut cur_node = start_node;
       match self.steps {
           Steps::Probability(alpha) => loop {
               // Sample the next edge
               cur_node = sampler.sample(graph, cur_node, rng)
                   .unwrap_or(start_node);

               if rng.gen::<f32>() < alpha {
                   break
               }
           },
           Steps::Fixed(steps) => for _ in 0..steps {
               // Sample the next edge
               cur_node = sampler.sample(graph, cur_node, rng)
                   .unwrap_or(start_node);
           }
       }
       cur_node
    }

}

/// Creates a trajectory by randomly walking through the graph, recording it
/// in the output vector.  We can use this for a variety of other problems, such as SMCI.
pub fn rollout<G: Graph + Send + Sync>(
    graph: &G, 
    steps: Steps, 
    sampler: &impl Sampler<G>,
    start_node: NodeID,
    rng: &mut impl Rng,
    output: &mut Vec<NodeID>
) {
    output.clear();
    let mut cur_node = start_node;
    match steps {
       Steps::Probability(alpha) => loop {
           // Sample the next edge
           cur_node = sampler.sample(graph, cur_node, rng)
               .unwrap_or(start_node);

           output.push(cur_node);
           if rng.gen::<f32>() < alpha {
               break;
           }
       },
       Steps::Fixed(steps) => for _ in 0..steps {
           // Sample the next edge
           cur_node = sampler.sample(graph, cur_node, rng)
               .unwrap_or(start_node);
           output.push(cur_node);
       }
   };
}


#[cfg(test)]
mod rwr_tests {
    use super::*;
    use crate::graph::{CumCSR,CSR};
    use crate::sampler::Unweighted;
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
    fn test_rwr() {
        let edges = build_edges();

        let csr = CSR::construct_from_edges(edges);
        let ccsr = CumCSR::convert(csr);
        let rwr = RWR {
            steps: Steps::Probability(0.1),
            walks: 10_000,
            beta: 0.5,
            seed: 20222022
        };

        let map = rwr.sample(&ccsr, &Unweighted, 0);
        println!("{:?}", map);
        let mut v: Vec<_> = map.into_iter().collect();
        v.sort_by_key(|(_idx, w)| FloatOrd(*w));

        assert_eq!(v[0].0, 0);
        assert_eq!(v[1].0, 2);
        assert_eq!(v[2].0, 1);
    }

}
