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

#[derive(Copy,Clone)]
pub enum Loss {
    /// This is the max margin loss with threshold that's common in embedding work.  FaceNet was
    /// one of the first to define it and a good starting point
    MarginLoss(f32, usize),

    /// Contrastive Loss is another embedding approach; this one uses tau, or temperature, to
    /// moderate how much weight to give differences
    Contrastive(f32, usize),

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
            Loss::Contrastive(_, negs) => *negs,
            Loss::MarginLoss(_, negs)  => *negs,
            Loss::StarSpace(_, negs) => *negs,
            Loss::RankLoss(_, negs) => *negs,
            Loss::RankSpace(_, negs) => *negs,
            Loss::PPR(_, negs, _) => *negs
        }
    }

    // thv is the reconstruction of v from its neighbor nodes or 
    // a random positive, depending on the loss
    // hv is the embedding constructed from its features
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

            Loss::Contrastive(tau, _)  => {
                let thv_norm = il2norm(&thv);
                let hv_norm  = il2norm(&hv);

                let mut ds: Vec<_> = hus.iter().map(|hu| {
                    let hu_norm = il2norm(hu);
                    (cosine(thv_norm.clone(), hu_norm) / *tau).exp()
                }).collect();

                let d1 = cosine(thv_norm, hv_norm) / *tau;
                let d1_exp = d1.exp();
                ds.push(d1_exp.clone());
                -(d1_exp / ds.sum_all()).ln()
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
            Loss::MarginLoss(_,_) => {
                model.reconstruct_node_embedding(
                    graph, node, feature_store, feature_embeddings, rng)
            },
            Loss::StarSpace(_,_) | Loss::Contrastive(_,_) | Loss::RankLoss(_,_) | Loss::RankSpace (_,_) => {
                // Select random out edge
                let edges = graph.get_edges(node).0;

                // If it has no out edges, nothing to really do.  We can't build a comparison.
                let choice = *edges.choose(rng).unwrap_or(&node);
                model.construct_node_embedding(
                    choice, feature_store, feature_embeddings, rng)
            },
            Loss::PPR(_, num, restart_p) => {
                let mut nodes = Vec::with_capacity(*num);
                for _ in 0..(*num) {
                    if let Some(node) = random_walk(node, graph, rng, *restart_p, 10) {
                        nodes.push(node);
                    }
                }
                if nodes.len() == 0 {
                    nodes.push(node);
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
        // We want at least two steps in our walk
        // before exiting since 1 step guarantees an anchor
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
