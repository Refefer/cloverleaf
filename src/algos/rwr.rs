use hashbrown::HashMap;
use rand::prelude::*;
use rand_distr::{Distribution,Uniform,Normal};
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;

use crate::graph::{CDFGraph,CumCSR,NodeID};
use crate::sampler::{UniformSample};

pub struct RWR {
    pub alpha: f32,
    pub walks: usize,
    pub seed: u64
}

impl RWR {

    pub fn weighted<G: CDFGraph + Send + Sync>(
        &self, 
        graph: &G, 
        start_node: NodeID
    ) -> HashMap<NodeID, f32> {
       let mut ret = (0..self.walks).into_par_iter()
           .map(|idx| {
               let mut rng = XorShiftRng::seed_from_u64(self.seed + idx as u64);
               let mut cur_node = start_node;
               loop {
                   // Sample the next edge
                   cur_node = UniformSample::sample(graph, cur_node, &mut rng)
                       .unwrap_or(start_node);

                   if rng.gen::<f32>() < self.alpha {
                       break;
                   }
               }
               cur_node
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
               let d = (graph.degree(*k) as f32).powf(0.5);
               *v /= (self.walks as f32) * d;
           });
       ret
    }

}

#[cfg(test)]
mod rwr_tests {
    use super::*;
    use crate::graph::CSR;
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
            alpha: 0.1,
            walks: 10_000,
            seed: 20222022
        };

        let map = rwr.weighted(&ccsr, 0);
        println!("{:?}", map);
        let mut v: Vec<_> = map.into_iter().collect();
        v.sort_by_key(|(_idx, w)| FloatOrd(*w));

        assert_eq!(v[0].0, 0);
        assert_eq!(v[1].0, 2);
        assert_eq!(v[2].0, 1);
    }

}
