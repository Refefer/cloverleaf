//! Defines the different losses for use within the Embedding Propagation framework.
//! Admitedly, the EP framework isn't parameterized on loss, so technically choosing a loss other
//! than Margin Loss is a different optimizer. 
use crate::autograd::*;
use rand::prelude::*;
use rand_distr::{Distribution,Uniform};

use crate::EmbeddingStore;
use crate::FeatureStore;
use crate::graph::{Graph as CGraph,NodeID};
use crate::geometry::Geometry;
use super::model::*;
use float_ord::FloatOrd;

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

/// Constructors; every loss uses at least one negative.
impl Loss {
    /// Max-margin loss over Euclidean distances, with margin `gamma`.
    pub fn margin(gamma: f32, negatives: usize) -> Self {
        Loss::MarginLoss(gamma, negatives.max(1))
    }

    /// Contrastive loss with a positive and a negative margin.
    pub fn contrastive(positive_margin: f32, negative_margin: f32, negatives: usize) -> Self {
        Loss::Contrastive(positive_margin, negative_margin, negatives.max(1))
    }

    /// StarSpace: max-margin loss over cosine similarities, with margin `gamma`.
    pub fn starspace(gamma: f32, negatives: usize) -> Self {
        Loss::StarSpace(gamma, negatives.max(1))
    }

    /// Softmax rank loss with temperature `tau`.
    pub fn rank(tau: f32, negatives: usize) -> Self {
        Loss::RankLoss(tau, negatives.max(1))
    }

    /// StarSpace and rank loss combined, with temperature `tau`.
    pub fn rankspace(tau: f32, negatives: usize) -> Self {
        Loss::RankSpace(tau, negatives.max(1))
    }

    /// Margin loss towards nodes reached by random walks restarting with probability `restart_p`.
    pub fn ppr(gamma: f32, negatives: usize, restart_p: f32) -> Self {
        Loss::PPR(gamma, negatives.max(1), restart_p)
    }
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

    /// Whether the loss can score embeddings in `geometry`: the rank and contrastive losses
    /// score by dot product and cosine, which the Poincaré ball doesn't have.
    pub fn supports(&self, geometry: &Geometry) -> bool {
        matches!(geometry, Geometry::Euclidean)
            || matches!(self, Loss::MarginLoss(..) | Loss::PPR(..) | Loss::StarSpace(..))
    }

    // hv is the embedding constructed from its features
    // thv is the reconstruction of v from its neighbor nodes or 
    // a random positive, depending on the loss
    // hu is a random negative node constructed via its neighbors
    // All three are already in `geometry`'s space (Geometry::to_space_node).  In the Poincaré ball
    // the margin losses use hyperbolic distance, and StarSpace negative hyperbolic distance in
    // place of cosine similarity.
    pub fn compute(&self, thv: ANode, hv: ANode, hus: &[ANode], geometry: &Geometry) -> ANode {
        if let (Geometry::Poincare(ball), Loss::StarSpace(gamma, _)) = (geometry, self) {
            // max(0, gamma - (s(thv, hv) - s(hv, hu))) with s = -d: gamma + d(thv, hv) - d(hv, hu).
            let d_pos = *gamma + ball.distance_node(&thv, &hv);
            let losses = hus.iter()
                .map(|hu| (&d_pos - ball.distance_node(hu, &hv)).maximum(0f32))
                .filter(|l| l.value()[0] > 0f32)
                .collect::<Vec<_>>();
            return if losses.len() > 0 {
                let n_losses = losses.len() as f32;
                losses.sum_all() / n_losses
            } else {
                Constant::scalar(0f32)
            }
        }
        let distance = |a: &ANode, b: &ANode| match geometry {
            Geometry::Euclidean => euclidean_distance(a, b),
            Geometry::Poincare(ball) => ball.distance_node(a, b)
        };
        match self {

            Loss::MarginLoss(gamma, _) | Loss::PPR(gamma, _, _) => {
                let d1 = *gamma + distance(&thv, &hv);
                let pos_losses = hus.iter()
                    .map(|hu| &d1 - distance(&thv, hu))
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
                let ss_loss = Loss::StarSpace(*gamma, *n).compute(thv.clone(), hv.clone(), hus, geometry);
                let rank_loss = Loss::RankLoss(*gamma, *n).compute(thv, hv, hus, geometry);
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
                        (*gamma - (&reconstruction_dist - cosine(hv_norm.clone(), hu_norm))).maximum(0f32)
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

                let pos_reconstruction = (*pos_margin - cosine(thv_norm, hv_norm.clone())).maximum(0f32);
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
                let mut sm = softmax(dsc.clone(), false);
                // Approximate softmax has some underflow issues in the taylor series
                // so, we test if running the natural log causes a NaN, and if so,
                // backtrack to the exact version at the expense of speed
                if sm.value()[len-1] <= 0f32 {
                    sm = softmax(dsc, true);
                }
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
        params: &M::Params,
        graph: &G,
        node: NodeID,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        model: &M,
        rng: &mut R
    ) -> (NodeCounts,ANode) {
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
                model.construct_from_multiple_nodes(params, nodes.into_iter(),
                        feature_store, feature_embeddings, rng)
            },
            _ => {
                model.reconstruct_node_embedding(
                    params, graph, node, feature_store, feature_embeddings, rng)
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

// Simple softmax.
pub fn softmax(numers: ANode, exact: bool) -> ANode {
    // Doesn't need to be part of the graph
    let numer_values = numers.value();
    let max_value = numer_values.iter()
        .max_by_key(|v| FloatOrd(**v))
        .expect("Shouldn't be non-zero!");

    let mv = Constant::scalar(*max_value);
    let n = if exact {
        (numers - &mv).exp()
    } else {
        (numers - &mv).exp_approx()
    };
    &n / n.sum()
}

#[cfg(test)]
mod ep_loss_tests {

    /// Every constructor uses at least one negative.
    #[test]
    fn test_constructors_clamp_negatives() {
        assert_eq!(Loss::margin(1., 0).negatives(), 1);
        assert_eq!(Loss::starspace(0.5, 0).negatives(), 1);
        assert_eq!(Loss::rankspace(0.5, 3).negatives(), 3);
        assert_eq!(Loss::ppr(1., 0, 0.5).negatives(), 1);
    }

    use super::*;

    #[test]
    fn test_euclidean_dist() {
        let x = Variable::new(vec![1f32, 3f32]);
        let y = Variable::new(vec![3f32, 5f32]);
        let dist = euclidean_distance(&x, &y);
        assert_eq!(dist.value(), &[(8f32).powf(0.5)]);
    }

    /// The Poincaré losses match the GPU trainer's (gpu/ep_gpu/losses.py) on the same tangent
    /// vectors: values and the gradient with respect to h(v)'s tangent vector, through exp0 with
    /// clipping (the third negative is past max_tangent_norm).
    #[test]
    fn test_poincare_losses_match_the_gpu_trainer() {
        use crate::geometry::{Curvature, PoincareBall};
        let geometry = Geometry::Poincare(PoincareBall::new(Curvature::new(0.7).unwrap(), Some(2.0)).unwrap());
        let cases = [
            (Loss::StarSpace(0.5, 3), 0.95257360f32, [0.20113260f32, 0.01119233, 0.33772844, 0.26259622]),
            (Loss::MarginLoss(1.0, 3), 1.62045932, [1.09414995, 0.59685302, -1.69100356, 1.61992657]),
        ];
        for (loss, expected, expected_grad) in cases {
            let t = Variable::new(vec![0.3, -0.8, 0.5, 0.1]);
            let h = Variable::new(vec![0.4, -0.6, 0.2, 0.3]);
            let hus: Vec<ANode> = [vec![-0.5f32, 0.9, 0.3, -0.2], vec![0.35, -0.7, 0.4, 0.2], vec![2.5, 1.0, -1.5, 0.5]]
                .into_iter().map(|v| geometry.to_space_node(Variable::new(v))).collect();
            let value = loss.compute(geometry.to_space_node(t), geometry.to_space_node(h.clone()), &hus, &geometry);
            assert!((value.value()[0] - expected).abs() < 1e-4, "{:?}: {} vs {}", loss, value.value()[0], expected);
            let mut graph = Graph::new();
            graph.backward(&value);
            let grad = graph.get_grad(&h).unwrap();
            for (g, e) in grad.iter().zip(expected_grad.iter()) {
                assert!((g - e).abs() < 1e-3, "{:?}: grad {:?} vs {:?}", loss, grad, expected_grad);
            }
        }
    }

    #[test]
    fn test_only_distance_losses_support_the_poincare_ball() {
        use crate::geometry::{Curvature, PoincareBall};
        let ball = Geometry::Poincare(PoincareBall::new(Curvature::new(1.).unwrap(), None).unwrap());
        assert!(Loss::StarSpace(0.1, 1).supports(&ball) && Loss::MarginLoss(1., 1).supports(&ball));
        assert!(!Loss::RankLoss(0.1, 1).supports(&ball) && !Loss::RankSpace(0.1, 1).supports(&ball));
        assert!(!Loss::Contrastive(0.5, 0.1, 1).supports(&ball));
        assert!(Loss::RankLoss(0.1, 1).supports(&Geometry::Euclidean));
    }

    #[test]
    fn test_l2norm() {
        let x = Variable::new(vec![1f32, 3f32]);
        let norm = il2norm(&x);
        let denom = 10f32.powf(0.5);
        assert_eq!(norm.value(), &[1f32 / denom, 3f32 / denom]);
    }

}
