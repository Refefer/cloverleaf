//! Defines the different losses for use within the Embedding Propagation framework.
//! Admitedly, the EP framework isn't parameterized on loss, so technically choosing a loss other
//! than Margin Loss is a different optimizer. 
use candle_core::{Device, Tensor};
use rand::prelude::*;
use rand_distr::{Distribution,Uniform};

use crate::EmbeddingStore;
use crate::FeatureStore;
use crate::graph::{Graph as CGraph,NodeID};
use super::model::*;
use super::attention::softmax;

#[derive(Copy,Clone,Debug)]
pub enum Loss {
    MarginLoss(f32, usize),
    Contrastive(f32, f32, usize),
    StarSpace(f32, usize),
    RankLoss(f32, usize),
    RankSpace(f32, usize),
    PPR(f32, usize, f32)
}

impl Loss {
    pub fn negatives(&self) -> usize {
        match self {
            Loss::Contrastive(_, _, negs) => *negs,
            Loss::MarginLoss(_, negs)  => *negs,
            Loss::StarSpace(_, negs) => *negs,
            Loss::RankLoss(_, negs) => *negs,
            Loss::RankSpace(_, negs) => *negs,
            Loss::PPR(_, negs, _) => *negs
        }
    }

    pub fn compute(&self, thv: &Tensor, hv: &Tensor, hus: &[Tensor], device: &Device) -> Tensor {
        match self {
            Loss::MarginLoss(gamma, _) | Loss::PPR(gamma, _, _) => {
                let d1 = crate::candle_utils::euclidean_distance(thv, hv).unwrap().add(&Tensor::from_slice(&[*gamma], 1usize, device).unwrap()).unwrap();
                let pos_losses: Vec<_> = hus.iter()
                    .map(|hu| {
                        let dist = crate::candle_utils::euclidean_distance(thv, hu).unwrap();
                        d1.sub(&dist).unwrap()
                    })
                    .filter(|loss| loss.to_vec1::<f32>().unwrap()[0] > 0f32)
                    .collect();

                if !pos_losses.is_empty() {
                    let n_losses = pos_losses.len() as f32;
                    let sum = pos_losses.iter().cloned().reduce(|a, b| a.add(&b).unwrap()).unwrap();
                    sum.div(&Tensor::from_slice(&[n_losses], 1usize, device).unwrap()).unwrap()
                } else {
                    Tensor::from_slice(&[0f32], 1usize, device).unwrap()
                }
            },

            Loss::RankSpace(gamma, n) => {
                let ss_loss = Loss::StarSpace(*gamma, *n).compute(thv, hv, hus, device);
                let rank_loss = Loss::RankLoss(*gamma, *n).compute(thv, hv, hus, device);
                ss_loss.add(&rank_loss).unwrap()
            }

            Loss::StarSpace(gamma, _)  => {
                let thv_norm = crate::candle_utils::il2norm(thv).unwrap();
                let hv_norm  = crate::candle_utils::il2norm(hv).unwrap();
                let reconstruction_dist = crate::candle_utils::cosine(&thv_norm, &hv_norm).unwrap();
                let gamma_t = Tensor::from_slice(&[*gamma], 1usize, device).unwrap();
                
                let losses: Vec<_> = hus.iter()
                    .map(|hu| {
                        let hu_norm = crate::candle_utils::il2norm(hu).unwrap();
                        let cos = crate::candle_utils::cosine(&hv_norm, &hu_norm).unwrap();
                        let diff = gamma_t.sub(&reconstruction_dist).unwrap().add(&cos).unwrap();
                        // Maximum with 0
                        let max_val = diff.maximum(&Tensor::from_slice(&[0f32], 1usize, device).unwrap()).unwrap();
                        max_val
                    })
                    .filter(|l| l.to_vec1::<f32>().unwrap()[0] > 0f32)
                    .collect();

                if !losses.is_empty() {
                    let n_losses = losses.len() as f32;
                    let sum = losses.iter().cloned().reduce(|a, b| a.add(&b).unwrap()).unwrap();
                    sum.div(&Tensor::from_slice(&[n_losses], 1usize, device).unwrap()).unwrap()
                } else {
                    Tensor::from_slice(&[0f32], 1usize, device).unwrap()
                }
            },

            Loss::Contrastive(pos_margin, neg_margin, _)  => {
                let thv_norm = crate::candle_utils::il2norm(thv).unwrap();
                let hv_norm  = crate::candle_utils::il2norm(hv).unwrap();
                let pos_margin_t = Tensor::from_slice(&[*pos_margin], 1usize, device).unwrap();
                let neg_margin_t = Tensor::from_slice(&[*neg_margin], 1usize, device).unwrap();

                let pos_cos = crate::candle_utils::cosine(&thv_norm, &hv_norm).unwrap();
                let pos_reconstruction = pos_margin_t.sub(&pos_cos).unwrap().maximum(&Tensor::from_slice(&[0f32], 1usize, device).unwrap()).unwrap();
                
                let mut margins: Vec<_> = hus.iter().map(|hu| {
                    let hu_norm = crate::candle_utils::il2norm(hu).unwrap();
                    let cs = crate::candle_utils::cosine(&hv_norm, &hu_norm).unwrap();
                    cs.sub(&neg_margin_t).unwrap().maximum(&Tensor::from_slice(&[0f32], 1usize, device).unwrap()).unwrap()
                })
                .filter(|v| v.to_vec1::<f32>().unwrap()[0] > 0f32)
                .collect();

                if pos_reconstruction.to_vec1::<f32>().unwrap()[0] > 0f32 {
                    margins.push(pos_reconstruction);
                }
                let n = margins.len();
                if n > 0 {
                    let sum = margins.iter().cloned().reduce(|a, b| a.add(&b).unwrap()).unwrap();
                    sum.div(&Tensor::from_slice(&[n as f32], 1usize, device).unwrap()).unwrap()
                } else {
                    Tensor::from_slice(&[0f32], 1usize, device).unwrap()
                }
            }

            Loss::RankLoss(tau, _)  => {
                let mut ds: Vec<_> = hus.iter().map(|hu| {
                    crate::candle_utils::dot(hv, hu).unwrap()
                }).collect();
                
                ds.push(crate::candle_utils::dot(hv, thv).unwrap());
                let len = ds.len();
                let dsc = Tensor::cat(&ds, 0).unwrap();
                let mut sm = softmax(&dsc, false);
                if sm.to_vec1::<f32>().unwrap()[len-1] <= 0f32 {
                    sm = softmax(&dsc, true);
                }
                let p = sm.narrow(0, len-1, 1).unwrap();
                let pi = p.to_vec1::<f32>().unwrap()[0];
                if pi < *tau {
                    p.log().unwrap().mul(&Tensor::from_slice(&[-1f32], 1usize, device).unwrap()).unwrap()
                } else {
                    Tensor::from_slice(&[0f32], 1usize, device).unwrap()
                }
            }
        }
    }

    pub fn construct_positive<G: CGraph, R: Rng, M: Model>(
        &self,
        graph: &G,
        node: NodeID,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        model: &M,
        rng: &mut R
    ) -> (NodeCounts,Tensor) {
        match self {
            Loss::PPR(_, num, restart_p) => {
                let mut nodes = Vec::with_capacity(*num);
                for _ in 0..(*num) {
                    if let Some(node) = random_walk(node, graph, rng, *restart_p, 10) {
                        nodes.push((node, 1f32));
                    }
                }
                if nodes.len() == 0 {
                    nodes.push((node, 1f32));
                }
                model.construct_from_multiple_nodes(nodes.into_iter(),
                        feature_store, feature_embeddings, rng)
            },
            _ => {
                model.reconstruct_node_embedding(
                    graph, node, feature_store, feature_embeddings, rng)
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
    
    loop {
        i += 1;
        let edges = graph.get_edges(node).0;
        if edges.len() == 0 || i > max_steps {
            break
        }
        let dist = Uniform::new(0, edges.len());
        node = edges[dist.sample(rng)];
        if i > 1 && rng.gen::<f32>() < restart_p && node != anchor { break }
    }

    if node != anchor {
        if !anchor_edges.is_empty() {
            Some(anchor_edges[Uniform::new(0, anchor_edges.len()).sample(rng)])
        } else {
            None
        }
    } else if !anchor_edges.is_empty() {
        Some(anchor_edges[Uniform::new(0, anchor_edges.len()).sample(rng)])
    } else {
        None
    }
}

fn l2norm(v: &Tensor, device: &Device) -> Tensor {
    v.powf(2.0).unwrap().sum_all().unwrap().sqrt().unwrap()
}

fn il2norm(v: &Tensor, device: &Device) -> Tensor {
    v.div(&l2norm(v, device)).unwrap()
}

fn cosine(x1: &Tensor, x2: &Tensor, device: &Device) -> Tensor {
    crate::candle_utils::dot(x1, x2).unwrap()
}

fn euclidean_distance(e1: &Tensor, e2: &Tensor, device: &Device) -> Tensor {
    crate::candle_utils::euclidean_distance(e1, e2).unwrap()
}

#[cfg(test)]
mod ep_loss_tests {
    use super::*;

    #[test]
    fn test_euclidean_dist() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1f32, 3f32], 2usize, &device).unwrap();
        let y = Tensor::from_slice(&[3f32, 5f32], 2usize, &device).unwrap();
        let dist = euclidean_distance(&x, &y, &device);
        assert!((dist.to_vec1::<f32>().unwrap()[0] - (8f32).powf(0.5)).abs() < 1e-5);
    }

    #[test]
    fn test_l2norm() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1f32, 3f32], 2usize, &device).unwrap();
        let norm = il2norm(&x, &device);
        let denom = 10f32.powf(0.5);
        let vals = norm.to_vec1::<f32>().unwrap();
        assert!((vals[0] - 1f32 / denom).abs() < 1e-5);
        assert!((vals[1] - 3f32 / denom).abs() < 1e-5);
    }
}
