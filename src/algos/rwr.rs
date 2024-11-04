//! Classic Random walk with Restarts.  This uses the Rp3b algorithm to allow biasing toward/away
//! from popular nodes to rarer nodes.  
use hashbrown::{HashMap,HashSet};
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rand_distr::{Distribution,Uniform};
use rayon::prelude::*;

use crate::graph::{Graph,NodeID,CDFtoP,CDFGraph};
use crate::sampler::{Sampler, weighted_sample_cdf};

// Fixed step or random restarts
#[derive(Clone,Copy,Debug)]
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
    pub single_threaded: bool,
    pub seed: u64
}

impl RWR {

    pub fn sample<G: Graph + Send + Sync>(
        &self, 
        graph: &G, 
        sampler: &impl Sampler<G>,
        start_node: NodeID
    ) -> HashMap<NodeID, f32> {
        if self.single_threaded {
            self.sample_st(graph, sampler, start_node)
        } else {
            self.sample_mt(graph, sampler, start_node)
        }
    }

    fn sample_mt<G: Graph + Send + Sync>(
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

    fn sample_st<G: Graph + Send + Sync>(
        &self, 
        graph: &G, 
        sampler: &impl Sampler<G>,
        start_node: NodeID
    ) -> HashMap<NodeID, f32> {
        let mut counts = HashMap::new();
        let mut rng = XorShiftRng::seed_from_u64(self.seed);

        (0..self.walks).for_each(|_| {
            let node = self.walk(graph, sampler, start_node, &mut rng);
            *counts.entry(node).or_insert(0f32) += 1f32; 
        });

        counts.iter_mut()
            .for_each(|(k, v)| {
                let d = (graph.degree(*k) as f32).powf(self.beta);
                *v /= (self.walks as f32) * d;
            });
        counts 
    }

    fn sample_level<G: CDFGraph + Send + Sync>(
        &self, 
        graph: &G, 
        start_node: NodeID,
        walks: usize,
        rng: &mut impl Rng,
        entries: &mut HashMap<NodeID, usize>
    ) {

        let (edges, weights) = graph.get_edges(start_node);
        let probs = CDFtoP::new(weights);
        let mut cur_idx = weighted_sample_cdf(weights, rng);
        if edges.len() > 0 {
            let dist = Uniform::new(0, edges.len());
            for _ in 0..walks {
                let candidate = dist.sample(rng);
                let cur_p = probs.prob(cur_idx);
                let new_p = probs.prob(candidate);
                if rng.gen::<f32>() < new_p / cur_p {
                    cur_idx = candidate;
                }
                *entries.entry(edges[cur_idx]).or_insert(0) += 1;
            }
        } else {
            *entries.entry(start_node).or_insert(0) += 1;
        }
    }

    pub fn sample_bfs<G: CDFGraph + Send + Sync>(
        &self, 
        graph: &G, 
        start_node: NodeID
    ) -> HashMap<NodeID, f32> {
        let mut rng = XorShiftRng::seed_from_u64(self.seed as u64);
        let mut ret = HashMap::new();

        let mut counts = HashMap::new();
        let mut next_counts = HashMap::new();
        counts.insert(start_node, self.walks);
        let mut pass = 1;
        loop {
            if counts.len() == 0 { break }
            counts.drain().for_each(|(node_id, num_walks)| {
                self.sample_level(graph, node_id, num_walks, &mut rng, &mut next_counts);
            });

            match self.steps {
                Steps::Fixed(max_pass) => {
                    if max_pass == pass {
                        std::mem::swap(&mut ret, &mut next_counts);
                    } else {
                        std::mem::swap(&mut next_counts, &mut counts);
                    }
                },
                Steps::Probability(p) => {
                    next_counts.drain().map(|(node_id, count)| {
                        let discount = (count as f32 * p).ceil() as usize;
                        let rem = count - discount;
                        *ret.entry(node_id).or_insert(0usize) += discount;
                        (node_id, rem)
                    })
                    .filter(|(_, c)| *c > 0)
                    .for_each(|(n, c)| { counts.insert(n, c); });
                }
            }
            pass += 1;
        }
        ret.into_iter()
            .map(|(k, v)| {
                let d = (graph.degree(k) as f32).powf(self.beta);
                let v = v as f32 / ((self.walks as f32) * d);
                (k, v)
            })
            .collect()
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

pub fn ppr_estimate<G: Graph>(
    graph: &G,
    start_node: NodeID,
    alpha: f32,
    eps: f32,
) -> HashMap<NodeID, f32> {
    let mut r = HashMap::new();
    let mut pi = HashMap::new();
    let mut push_set = HashSet::new();
    let mut push = Vec::new();
    r.insert(start_node, 1f32);
    push_set.insert(start_node);
    push.push(start_node);
    while let Some(w) = push.pop() {
        push_set.remove(&w);

        let r_hat = r[&w];
        *pi.entry(w).or_insert(0f32) += alpha * r_hat;
        r.insert(w, (1f32 - alpha) * r_hat / 2f32);

        let (edges, weights) = graph.get_edges(w);
        if r[&w] > eps * edges.len() as f32 {
            push_set.insert(w);
            push.push(w);
        }

        edges.iter().zip( CDFtoP::new(weights) ).for_each(|(u, u_w)| {
            let r_u = *r.get(u).unwrap_or(&0f32) + u_w * (1f32 - alpha) * r_hat / 2f32;
            r.insert(*u, r_u);
            if r_u > eps * graph.degree(*u) as f32 && !push_set.contains(u) {
                push_set.insert(*u);
                push.push(*u);
            }
        });
    }
    pi
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
