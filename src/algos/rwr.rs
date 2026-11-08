//! Classic Random walk with Restarts.  This uses the Rp3b algorithm to allow biasing toward/away
//! from popular nodes to rarer nodes.  
use hashbrown::{HashMap,HashSet};
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rand_distr::{Binomial,Distribution,Uniform};
use rayon::prelude::*;

use crate::graph::{Graph,NodeID,CDFtoP,CDFGraph};
use crate::sampler::{Sampler, Weighted, Unweighted, weighted_sample_cdf};
use crate::algos::utils::Sample;

pub struct RWR {
    pub steps: Sample,
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
                // Walks can end on nodes without out edges; treat them as degree 1.
                let d = (graph.degree(*k).max(1) as f32).powf(self.beta);
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
                // Walks can end on nodes without out edges; treat them as degree 1.
                let d = (graph.degree(*k).max(1) as f32).powf(self.beta);
                *v /= (self.walks as f32) * d;
            });
        counts 
    }

    fn sample_level<G: CDFGraph + Send + Sync>(
        &self, 
        graph: &G, 
        start_node: NodeID,
        walks: usize,
        origin: NodeID,
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
            // Like `walk`, every walk at a dead end restarts from the origin.
            *entries.entry(origin).or_insert(0) += walks;
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

        // Zero-length walks terminate at the start node, matching `walk`.
        if matches!(self.steps, Sample::Fixed(0)) {
            std::mem::swap(&mut ret, &mut counts);
        }

        let mut pass = 1;
        loop {
            if counts.len() == 0 { break }
            counts.drain().for_each(|(node_id, num_walks)| {
                self.sample_level(graph, node_id, num_walks, start_node, &mut rng, &mut next_counts);
            });

            match self.steps {
                Sample::Fixed(max_pass) => {
                    if max_pass == pass {
                        std::mem::swap(&mut ret, &mut next_counts);
                    } else {
                        std::mem::swap(&mut next_counts, &mut counts);
                    }
                },
                Sample::Probability(p) => {
                    next_counts.drain().map(|(node_id, count)| {
                        // Each walk stops here independently with probability p.  Rounding the
                        // expected count instead stops every lone walk.
                        let discount = Binomial::new(count as u64, p as f64)
                            .expect("p is validated by Sample")
                            .sample(&mut rng) as usize;
                        let rem = count - discount;
                        *ret.entry(node_id).or_insert(0usize) += discount;
                        (node_id, rem)
                    })
                    .filter(|(_, c)| *c > 0)
                    .for_each(|(n, c)| { counts.insert(n, c); });
                },
               Sample::All => { panic!("Sample::All is illegal!") }
            }
            pass += 1;
        }
        ret.into_iter()
            .map(|(k, v)| {
                // Walks can end on nodes without out edges; treat them as degree 1.
                let d = (graph.degree(k).max(1) as f32).powf(self.beta);
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
           Sample::Probability(alpha) => loop {

               // Sample the next edge
               cur_node = sampler.sample(graph, cur_node, rng)
                   .unwrap_or(start_node);

               if rng.gen::<f32>() < alpha {
                   break
               }

           },
           Sample::Fixed(steps) => for _ in 0..steps {
               // Sample the next edge
               cur_node = sampler.sample(graph, cur_node, rng)
                   .unwrap_or(start_node);
           },
           Sample::All => { panic!("Sample::All is illegal!") }
       }
       cur_node
    }

}

/// Creates a trajectory by randomly walking through the graph, recording it
/// in the output vector.  We can use this for a variety of other problems, such as SMCI.
/// Runs one rollout per rng from `start_node`, returning each full path (start node included).
pub fn rollouts<G>(
    graph: &G,
    steps: Sample,
    weighted: bool,
    start_node: NodeID,
    rngs: Vec<XorShiftRng>
) -> Vec<Vec<NodeID>>
    where G: Graph + Send + Sync,
          Weighted: Sampler<G>
{
    rngs.into_par_iter().map(|mut rng| {
        let mut walk = vec![start_node];
        if weighted {
            rollout(graph, steps, &Weighted, start_node, &mut rng, &mut walk);
        } else {
            rollout(graph, steps, &Unweighted, start_node, &mut rng, &mut walk);
        };
        walk
    }).collect()
}

pub fn rollout<G: Graph + Send + Sync>(
    graph: &G, 
    steps: Sample, 
    sampler: &impl Sampler<G>,
    start_node: NodeID,
    rng: &mut impl Rng,
    output: &mut Vec<NodeID>
) {
    let mut cur_node = start_node;
    match steps {
       Sample::Probability(alpha) => loop {
           // Sample the next edge
           cur_node = sampler.sample(graph, cur_node, rng)
               .unwrap_or(start_node);

           output.push(cur_node);
           if rng.gen::<f32>() < alpha {
               break;
           }
       },
       Sample::Fixed(steps) => for _ in 0..steps {
           // Sample the next edge
           cur_node = sampler.sample(graph, cur_node, rng)
               .unwrap_or(start_node);
           output.push(cur_node);
       },
       Sample::All => { panic!("Sample::All is illegal!") }
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

        let csr = CSR::construct_from_edges(edges, false);
        let ccsr = CumCSR::convert(csr);
        let rwr = RWR {
            steps: Sample::Probability(0.1),
            walks: 10_000,
            beta: 0.5,
            single_threaded: false,
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

    /// A zero-length walk ends where it starts.  The BFS sampler must terminate (it used to spin
    /// forever) and agree with the per-walk sampler.
    #[test]
    fn test_sample_bfs_zero_steps_terminates() {
        use std::sync::mpsc;
        use std::time::Duration;

        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let ccsr = CumCSR::convert(CSR::construct_from_edges(build_edges(), false));
            let rwr = RWR {
                steps: Sample::Fixed(0),
                walks: 100,
                beta: 0.0,
                single_threaded: true,
                seed: 20222022
            };
            let bfs = rwr.sample_bfs(&ccsr, 0);
            let st = rwr.sample(&ccsr, &Unweighted, 0);
            tx.send((bfs, st)).unwrap();
        });

        let (bfs, st) = rx.recv_timeout(Duration::from_secs(10))
            .expect("sample_bfs did not terminate with Sample::Fixed(0)");
        assert_eq!(bfs.keys().collect::<Vec<_>>(), vec![&0]);
        assert_eq!(bfs, st);
    }

    /// `weighted=false` must sample neighbors uniformly regardless of edge weight.
    #[test]
    fn test_rollouts_unweighted_ignores_edge_weights() {
        let edges = vec![(0, 1, 1000.), (0, 2, 1.), (1, 0, 1.), (2, 0, 1.)];
        let ccsr = CumCSR::convert(CSR::construct_from_edges(edges, false));
        let n = 4000;
        let rngs = (0..n).map(|i| XorShiftRng::seed_from_u64(1234 + i)).collect();

        let walks = rollouts(&ccsr, Sample::Fixed(1), false, 0, rngs);
        assert_eq!(walks.len(), n as usize);
        let to_light_edge = walks.iter().filter(|w| w[1] == 2).count();
        let frac = to_light_edge as f32 / n as f32;
        assert!((0.45..0.55).contains(&frac),
            "unweighted rollouts took the light edge {} of {} times", to_light_edge, n);
    }


    /// Node 0 fans out to 100 children; child i leads to grandchild 100 + i, which loops on
    /// itself.  With restart probability p, a walk stops on a child with probability p, so 90%
    /// of walks should end past the children.
    fn fan_out_graph() -> CumCSR {
        let mut edges = Vec::new();
        for i in 1..=100 {
            edges.push((0, i, 1.));
            edges.push((i, 100 + i, 1.));
            edges.push((100 + i, 100 + i, 1.));
        }
        CumCSR::convert(CSR::construct_from_edges(edges, false))
    }

    #[test]
    fn test_sample_bfs_restarts_are_unbiased_when_walks_spread() {
        let graph = fan_out_graph();
        let rwr = RWR { steps: Sample::Probability(0.1), walks: 100, beta: 0.,
                        single_threaded: false, seed: 1234 };
        let scores = rwr.sample_bfs(&graph, 0);
        let past_children: f32 = scores.iter().filter(|(k, _)| **k > 100).map(|(_, v)| *v).sum();
        assert!((past_children - 0.9).abs() < 0.1, "fraction past children: {}", past_children);
    }

    /// Like `walk`, every walk at a node without out edges restarts from the start node.
    #[test]
    fn test_sample_bfs_restarts_all_walks_at_dead_ends() {
        let graph = CumCSR::convert(CSR::construct_from_edges(vec![(0, 1, 1.)], false));
        let rwr = RWR { steps: Sample::Fixed(2), walks: 100, beta: 0.,
                        single_threaded: false, seed: 1234 };
        let scores = rwr.sample_bfs(&graph, 0);
        assert_eq!(scores.len(), 1, "{:?}", scores);
        assert_eq!(scores.get(&0), Some(&1.0), "{:?}", scores);
    }

    /// A walk can end on a node with no out edges; its degree of zero must not make the score
    /// inf on any sampling path.
    #[test]
    fn test_scores_finite_at_zero_degree_nodes() {
        let graph = CumCSR::convert(CSR::construct_from_edges(vec![(0, 1, 1.)], false));
        for single_threaded in [false, true] {
            let rwr = RWR { steps: Sample::Fixed(1), walks: 100, beta: 0.5,
                            single_threaded, seed: 1234 };
            assert_eq!(rwr.sample_bfs(&graph, 0).get(&1), Some(&1.0));
            assert_eq!(rwr.sample(&graph, &Unweighted, 0).get(&1), Some(&1.0));
        }
    }
}
