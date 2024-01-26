//! Defines the different losses for use within the Embedding Propagation framework.
//! Admitedly, the EP framework isn't parameterized on loss, so technically choosing a loss other
//! than Margin Loss is a different optimizer. 
use simple_grad::*;
use rand::prelude::*;
use rand_distr::{Distribution,Uniform};

use crate::EmbeddingStore;
use crate::FeatureStore;
use crate::graph::{Graph as CGraph,NodeID};
use super::model::*;
use super::attention::softmax;

#[derive(Copy,Clone,Debug)]
pub enum Loss {
    /// This is the max margin loss with threshold that's common in embedding work.  FaceNet was
    /// one of the first to define it and a good starting point
    MarginLoss(f32, usize),

    /// Contrastive Loss is another embedding approach; two margins are provided: a positive and a
    /// negative margin, moderating how much each matters
    Contrastive(f32, f32, usize),

    /// This is loss used in the StarSpace paper.  It's basically max margin loss but using cosine
    /// rather than euclidean distances
    StarSpace(f32, usize),

    /// This use negative log likelihood to maximize a 1-of-N ranked list.
    RankLoss(f32, usize),
    
    /// This combines StarSpace and RankLoss, which for search purposes significantly outperforms
    /// the other losses
    RankSpace(f32, usize),

    /// This uses PPR to generate a set of candidates for optimize toward.  Should be broken out as
    /// it's fairly unique.
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

    // hv is the embedding constructed from its features
    // thv is the reconstruction of v from its neighbor nodes or 
    // a random positive, depending on the loss
    // hu is a random negative node constructed via its neighbors
    pub fn compute(&self, thv: ANode, hv: ANode, hus: &[ANode]) -> ANode {
        match self {

            Loss::MarginLoss(gamma, _) | Loss::PPR(gamma, _, _) => {
                let d1 = gamma + euclidean_distance(&thv, &hv);
                let pos_losses = hus.iter()
                    .map(|hu| &d1 - euclidean_distance(&thv, hu))
                    .filter(|loss| loss.value()[0] > 0f32)
                    .collect::<Vec<_>>();

                // Only return positive ones
                if pos_losses.len() > 0 {
                    let n_losses = pos_losses.len() as f32;
                    pos_losses.sum_all() / n_losses
                } else {
                    Constant::scalar(0f32)
                }
            },

            Loss::RankSpace(gamma, n) => {
                let ss_loss = Loss::StarSpace(*gamma, *n).compute(thv.clone(), hv.clone(), hus);
                let rank_loss = Loss::RankLoss(*gamma, *n).compute(thv, hv, hus);
                ss_loss + rank_loss
            }

            Loss::StarSpace(gamma, _)  => {
                let thv_norm = il2norm(&thv);
                let hv_norm  = il2norm(&hv);

                // margin between a positive node and its reconstruction
                // The more correlated
                let reconstruction_dist = cosine(thv_norm.clone(), hv_norm.clone());
                let losses = hus.iter()
                    .map(|hu| {
                        let hu_norm = il2norm(hu);
                        // Margin loss
                        (gamma - (&reconstruction_dist - cosine(hv_norm.clone(), hu_norm))).maximum(0f32)
                    })
                    // Only collect losses which are not zero
                    .filter(|l| l.value()[0] > 0f32)
                    .collect::<Vec<_>>();

                // Only return positive ones
                if losses.len() > 0 {
                    let n_losses = losses.len() as f32;
                    losses.sum_all() / n_losses
                } else {
                    Constant::scalar(0f32)
                }
            },

            Loss::Contrastive(pos_margin, neg_margin, _)  => {
                let thv_norm = il2norm(&thv);
                let hv_norm  = il2norm(&hv);

                let pos_reconstruction = (pos_margin - cosine(thv_norm, hv_norm.clone())).maximum(0f32);
                let mut margins: Vec<_> = hus.iter().map(|hu| {
                        let hu_norm = il2norm(hu);
                        let cs = cosine(hv_norm.clone(), hu_norm);
                        (cs - *neg_margin).maximum(0f32)
                    })
                    .filter(|v| v.value()[0] > 0f32)
                    .collect();

                if pos_reconstruction.value()[0] > 0f32 {
                    margins.push(pos_reconstruction);
                }
                let n = margins.len();
                if n > 0 {
                    margins.sum_all() / n as f32
                } else {
                    Constant::scalar(0f32)
                }
            }

            Loss::RankLoss(tau, _)  => {
                // Get the dot products
                let mut ds: Vec<_> = hus.iter().map(|hu| {
                    hu.dot(&hv)
                }).collect();
                
                // Add the positive example
                ds.push(hv.dot(&thv));
                let len = ds.len();
                let dsc = ds.concat();
                let sm = softmax(dsc);
                let p = sm.slice(len-1, 1);
                let pi = p.value()[0];
                if pi < *tau {
                    -p.ln()
                } else {
                    Constant::scalar(0f32)
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
    ) -> (NodeCounts,ANode) {
        match self {
            Loss::MarginLoss(_,_) | Loss::Contrastive(_,_,_) | Loss::RankLoss(_,_) | Loss::RankSpace (_,_) | Loss::StarSpace(_,_) => {
                model.reconstruct_node_embedding(
                    graph, node, feature_store, feature_embeddings, rng)
            },
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
        // We want at least one step in our walk
        // before exiting since zero-steps guarantees an anchor
        // edge
        if i > 1 && rng.gen::<f32>() < restart_p && node != anchor { break }
    }

    if node != anchor {
        Some(node)
    } else if anchor_edges.len() > 0 {
        Some(anchor_edges[Uniform::new(0, anchor_edges.len()).sample(rng)])
    } else {
        None
    }
}


fn l2norm(v: ANode) -> ANode {
    v.pow(2f32).sum().pow(0.5)
}

fn il2norm(v: &ANode) -> ANode {
    v / l2norm(v.clone())
}

fn cosine(x1: ANode, x2: ANode) -> ANode {
    x1.dot(&x2)
}

fn euclidean_distance(e1: &ANode, e2: &ANode) -> ANode {
    (e1 - e2).pow(2f32).sum().pow(0.5)
}

#[cfg(test)]
mod ep_loss_tests {
    use super::*;

    #[test]
    fn test_euclidean_dist() {
        let x = Variable::new(vec![1f32, 3f32]);
        let y = Variable::new(vec![3f32, 5f32]);
        let dist = euclidean_distance(&x, &y);
        assert_eq!(dist.value(), &[(8f32).powf(0.5)]);
    }

    #[test]
    fn test_l2norm() {
        let x = Variable::new(vec![1f32, 3f32]);
        let norm = il2norm(&x);
        let denom = 10f32.powf(0.5);
        assert_eq!(norm.value(), &[1f32 / denom, 3f32 / denom]);
    }

}
