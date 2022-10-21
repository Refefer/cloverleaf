use std::fmt::Write;

use rayon::prelude::*;
use hashbrown::HashMap;
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rand_distr::{Distribution,Uniform};
use simple_grad::*;

use crate::graph::{Graph as CGraph,NodeID};
use crate::embeddings::{EmbeddingStore,Distance};
use crate::vocab::Vocab;
use crate::progress::CLProgressBar;
use crate::algos::utils::FeatureStore;

pub struct EmbeddingPropagation {
    pub alpha: f32,
    pub gamma: f32,
    pub wd: f32,
    pub loss: Loss,
    pub batch_size: usize,
    pub dims: usize,
    pub passes: usize,
    pub seed: u64,
    pub max_features: Option<usize>,
    pub max_nodes: Option<usize>,
    pub indicator: bool
}

impl EmbeddingPropagation {

    pub fn learn<G: CGraph + Send + Sync>(
        &self, 
        graph: &G, 
        features: &FeatureStore
    ) -> EmbeddingStore {
        let mut agraph = Graph::new();
        let feat_embeds = self.learn_feature_embeddings(graph, &mut agraph, features);
        feat_embeds
    }

    
    // The uber expensive function
    fn learn_feature_embeddings<G: CGraph + Send + Sync>(
        &self,
        graph: &G,
        agraph: &mut Graph,
        features: &FeatureStore,
    ) -> EmbeddingStore {

        // We create separate embeddings for momentum and feature_embeddings.
        let mut feature_embeddings = EmbeddingStore::new(features.num_features(), self.dims, Distance::Cosine);
        let momentum = EmbeddingStore::new(features.num_features(), self.dims, Distance::Cosine);

        let mut rng = XorShiftRng::seed_from_u64(self.seed);
        
        // Initialize embeddings as random
        randomize_embedding_store(&mut feature_embeddings, &mut rng);

        let mut node_idxs: Vec<_> = (0..graph.len()).into_iter().collect();
        let pb = CLProgressBar::new((self.passes * graph.len()) as u64, self.indicator);
        
        // Enable/disable shared memory pool
        use_shared_pool(true);

        let mut current_error = std::f32::INFINITY;
        let mut alpha = self.alpha;
        for pass in 0..self.passes {
            pb.update_message(|msg| {
                msg.clear();
                write!(msg, "Pass {}/{}, Error: {:.5}, alpha: {:.5}", pass + 1, self.passes, current_error, alpha)
                    .expect("Error writing out indicator message!");
            });

            // Shuffle for SGD
            node_idxs.shuffle(&mut rng);
            let err: Vec<_> = node_idxs.par_iter().chunks(self.batch_size).enumerate().map(|(i, nodes)| {
                let mut grads = Vec::with_capacity(self.batch_size);
                let mut all_grads = HashMap::new();
                
                // Compute grads for batch
                nodes.par_iter().map(|node_id| {
                    let mut rng = XorShiftRng::seed_from_u64(self.seed + (i + **node_id) as u64);
                    let (loss, grads) = self.run_pass(graph, **node_id, &features, &feature_embeddings, &mut rng) ;
                    (loss, grads)
                }).collect_into_vec(&mut grads);

                let mut error = 0f32;
                let mut cnt = 0f32;
                // Since we're dealing with multiple reconstructions with likely shared features,
                // we aggregate all the gradients
                for (err, grad_set) in grads.drain(..nodes.len()) {
                    for (feat, grad) in grad_set.into_iter() {
                        let e = all_grads.entry(feat).or_insert_with(|| vec![0.; grad.len()]);
                        e.iter_mut().zip(grad.iter()).for_each(|(ei, gi)| *ei += *gi);
                    }
                    error += err;
                    cnt += 1f32;
                }
                
                // Backpropagate embeddings
                sgd(&feature_embeddings, &momentum, self.gamma, &mut all_grads, alpha);

                // Update progress bar
                pb.inc(nodes.len() as u64);
                error / cnt
            }).collect();

            let error = err.iter().sum::<f32>() / err.len() as f32;
            // Decay alpha when we plateau
            if error > current_error {
                alpha = (alpha * 0.99).max(self.alpha / 100.);
            }
            current_error = error;
        }
        pb.finish();
        feature_embeddings
    }

    fn sample_negatives<R: Rng>(&self, anchor: NodeID, n_nodes: usize, rng: &mut R) -> Vec<NodeID> {
        let dist = Uniform::new(0, n_nodes);

        // Get negative v
        let num_negs = self.loss.negatives();
        let mut negatives = Vec::with_capacity(num_negs);
        // We make a good attempt to get the full negatives, but bail
        // if it's computationally too expensive
        for _ in 0..(num_negs*2) {
            let neg_node = dist.sample(rng);
            if neg_node != anchor { 
                negatives.push(neg_node) ;
                if negatives.len() == num_negs { break }
            }
        }
        negatives
    }

    fn run_pass<G: CGraph + Send + Sync, R: Rng>(
        &self, 
        graph: &G,
        node: NodeID,
        features: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut R
    ) -> (f32, HashMap<usize, Vec<f32>>) {

        // h(v)
        let (hv_vars, hv) = construct_node_embedding(
            node, features, &feature_embeddings, self.max_features, rng);
        
        // ~h(v)
        let (thv_vars, thv) = self.loss.construct_positive(
            graph, node, features, &feature_embeddings, self.max_nodes, self.max_features, rng);
        
        // h(u)
        let negatives = self.sample_negatives(node, graph.len(), rng);
        let mut hu_vars = Vec::with_capacity(negatives.len());
        let mut hus = Vec::with_capacity(negatives.len());
        negatives.into_iter().for_each(|neg_node| {
            let (hu_var, hu) = construct_node_embedding(neg_node, features, &feature_embeddings, self.max_features, rng);
            hu_vars.push(hu_var);
            hus.push(hu);
        });

        // Compute error
        let mut loss = self.loss.compute(thv, hv.clone(), hus.clone());
        
        // Extract it so we can add weight decay
        let err = loss.value()[0];
        
        // Weight decay
        if self.wd > 0f32 {
            hus.push(hv);
            let norms = hus.into_iter()
                .map(|hu| (1f32 - l2norm(hu)).pow(2f32))
                .collect::<Vec<_>>().sum_all();
            loss = loss + self.wd * norms;
        }

        // Compute gradients
        let mut agraph = Graph::new();
        agraph.backward(&loss);

        let mut grads = HashMap::new();
        extract_grads(&agraph, &mut grads, hv_vars.into_iter());
        extract_grads(&agraph, &mut grads, thv_vars.into_iter());
        hu_vars.into_iter().for_each(|hu_var| {
            extract_grads(&agraph, &mut grads, hu_var.into_iter());
        });

        (err, grads)

    }

}

fn extract_grads(
    graph: &Graph, 
    grads: &mut HashMap<usize, Vec<f32>>, 
    vars: impl Iterator<Item=(usize, (ANode, usize))>
) {
    for (feat_id, (var, _)) in vars {
        if grads.contains_key(&feat_id) { continue }

        if let Some(grad) = graph.get_grad(&var) {
            if grad.iter().all(|gi| !gi.is_nan()) {
                // Can get some nans in weird cases, such as the distance between
                // a node and it's reconstruction when it shares all features.
                // Since that's not all that helpful anyways, we simply ignore it and move on
                grads.insert(feat_id, grad.to_vec());
            }
        }
    }
}

// We use SGD with momentum as it's fast, cheap, and easy to implement.
fn sgd(
    feature_embeddings: &EmbeddingStore,
    momentum: &EmbeddingStore,
    gamma: f32,
    grads: &mut HashMap<usize, Vec<f32>>,
    alpha: f32
) {
    for (feat_id, grad) in grads.drain() {

        let emb = feature_embeddings.get_embedding_mut_hogwild(feat_id);
        let mom = momentum.get_embedding_mut_hogwild(feat_id);

        if grad.iter().all(|gi| !gi.is_nan()) {
            // Can get some nans in weird cases, such as the distance between
            // a node and it's reconstruction when it shares all features.
            // We just skip over those weird ones.
            emb.iter_mut().zip(grad.iter().zip(mom.iter_mut())).for_each(|(ei, (gi, mi))| {
                *mi = gamma * *mi + *gi;
                *ei -= alpha * *mi;
            });
        }
    }

}

type NodeCounts = HashMap<usize, (ANode, usize)>;

fn collect_embeddings_from_node<R: Rng>(
    node: NodeID,
    feature_store: &FeatureStore,
    feature_embeddings: &EmbeddingStore,
    feat_map: &mut NodeCounts,
    max_features: Option<usize>,
    rng: &mut R
) {
    let feats = feature_store.get_features(node);
    let max_features = max_features.unwrap_or(feats.len());
    for feat in feats.choose_multiple(rng, max_features) {
        if let Some((_emb, count)) = feat_map.get_mut(feat) {
            *count += 1;
        } else {
            let emb = feature_embeddings.get_embedding(*feat);
            let v = Variable::pooled(emb);
            feat_map.insert(*feat, (v, 1));
        }
    }
}

// H(n)
// Average the features associated with a node
// to create the node embedding
fn construct_node_embedding<R: Rng>(
    node: NodeID,
    feature_store: &FeatureStore,
    feature_embeddings: &EmbeddingStore,
    max_features: Option<usize>,
    rng: &mut R
) -> (NodeCounts, ANode) {
    let mut feature_map = HashMap::new();
    collect_embeddings_from_node(node, feature_store, 
                                 feature_embeddings, 
                                 &mut feature_map,
                                 max_features,
                                 rng);

    let mean = mean_embeddings(feature_map.values());
    (feature_map, mean)
}

// ~H(n)
// The Expensive function.  We grab a nodes neighbors
// and use the average of their features to construct
// an estimate of H(n)
fn reconstruct_node_embedding<G: CGraph, R: Rng>(
    graph: &G,
    node: NodeID,
    feature_store: &FeatureStore,
    feature_embeddings: &EmbeddingStore,
    max_nodes: Option<usize>,
    max_features: Option<usize>,
    rng: &mut R
) -> (NodeCounts, ANode) {
    let edges = &graph.get_edges(node).0;
    let mut feature_map = HashMap::new();
    
    if edges.len() <= max_nodes.unwrap_or(edges.len()) {
        for out_node in edges.iter() {
            collect_embeddings_from_node(*out_node, feature_store, 
                                      feature_embeddings, 
                                      &mut feature_map,
                                      max_features,
                                      rng);
        }

    } else {
        for out_node in edges.choose_multiple(rng, max_nodes.unwrap()) {
            collect_embeddings_from_node(*out_node, feature_store, 
                                         feature_embeddings, 
                                         &mut feature_map,
                                         max_features,
                                         rng);
        }
    }
    let mean = mean_embeddings(feature_map.values());
    (feature_map, mean)
}

fn mean_embeddings<'a,I: Iterator<Item=&'a (ANode, usize)>>(items: I) -> ANode {
    let mut vs = Vec::new();
    let mut n = 0;
    items.for_each(|(emb, count)| {
        if *count > 1 {
            vs.push(emb * *count as f32);
        } else {
            vs.push(emb.clone());
        }
        n += *count;
    });
    vs.sum_all() / n as f32
}

fn euclidean_distance(e1: ANode, e2: ANode) -> ANode {
    (e1 - e2).pow(2f32).sum().pow(0.5)
}

#[derive(Copy,Clone)]
pub enum Loss {
    MarginLoss(f32, usize),
    Contrastive(f32, usize),
    StarSpace(f32, usize)
}



impl Loss {
    fn negatives(&self) -> usize {
        match self {
            Loss::Contrastive(_, negs) => *negs,
            Loss::MarginLoss(_, negs)  => *negs,
            Loss::StarSpace(_, negs) => *negs
        }
    }

    fn compute(&self, thv: ANode, hv: ANode, hus: Vec<ANode>) -> ANode {
        match self {

            Loss::MarginLoss(gamma, _) => {
                let d1 = gamma + euclidean_distance(thv.clone(), hv);
                let pos_losses = hus.into_iter()
                    .map(|hu| &d1 - euclidean_distance(thv.clone(), hu))
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

            Loss::StarSpace(gamma, _) => {
                let thv_norm = il2norm(thv);
                let hv_norm  = il2norm(hv);

                let d1 = gamma - cosine(thv_norm.clone(), hv_norm);
                let negs = hus.len();
                let pos_losses = hus.into_iter()
                    .map(|hu| {
                        let hu_norm = il2norm(hu);
                        (d1.clone() + cosine(thv_norm.clone(), hu_norm)).maximum(0f32)
                    })
                    // Only collect losses which are not zero
                    .filter(|l| l.value()[0] > 0f32)
                    .collect::<Vec<_>>();

                // Only return positive ones
                if pos_losses.len() > 0 {
                    let n_losses = pos_losses.len() as f32;
                    pos_losses.sum_all() / n_losses
                } else {
                    Constant::scalar(0f32)
                }
            },

            Loss::Contrastive(tau, _) => {
                let thv_norm = il2norm(thv);
                let hv_norm  = il2norm(hv);

                let mut ds: Vec<_> = hus.into_iter().map(|hu| {
                    let hu_norm = il2norm(hu);
                    (cosine(thv_norm.clone(), hu_norm) / *tau).exp()
                }).collect();

                let d1 = cosine(thv_norm, hv_norm) / *tau;
                let d1_exp = d1.exp();
                ds.push(d1_exp.clone());
                -(d1_exp / ds.sum_all()).ln()
            }
        }
    }

    fn construct_positive<G: CGraph, R: Rng>(
        &self,
        graph: &G,
        node: NodeID,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        max_nodes: Option<usize>,
        max_features: Option<usize>,
        rng: &mut R
    ) -> (NodeCounts,ANode) {
        match self {
            Loss::Contrastive(_,_) | Loss::MarginLoss(_,_) => {
                reconstruct_node_embedding(
                    graph, node, feature_store, feature_embeddings, max_nodes, max_features, rng)
            },
            Loss::StarSpace(_,_) => {
                // Select random out edge
                let edges = graph.get_edges(node).0;
                // If it has no out edges, nothing to really do.  We can't build a comparison.
                let choice = *edges.choose(rng).unwrap_or(&node);
                construct_node_embedding(
                    node, feature_store, feature_embeddings, max_features, rng)
            }
        }

    }

}

fn l2norm(v: ANode) -> ANode {
    v.pow(2f32).sum().pow(0.5)
}

fn il2norm(v: ANode) -> ANode {
    &v / l2norm(v.clone())
}

fn cosine(x1: ANode, x2: ANode) -> ANode {
    x1.dot(&x2)
}

fn randomize_embedding_store(es: &mut EmbeddingStore, rng: &mut impl Rng) {
    for idx in 0..es.len() {
        let e = es.get_embedding_mut(idx);
        let mut norm = 0f32;
        e.iter_mut().for_each(|ei| {
            *ei = 2f32 * rng.gen::<f32>() - 1f32;
            norm += ei.powf(2f32);
        });
        norm = norm.sqrt();
        e.iter_mut().for_each(|ei| *ei /= norm);
    }
}

#[cfg(test)]
mod ep_tests {
    use super::*;
    use crate::graph::{CumCSR,CSR};

    fn build_star_edges() -> Vec<(usize, usize, f32)> {
        let mut edges = Vec::new();
        let max = 100;
        for ni in 0..max {
            for no in (ni+1)..max {
                edges.push((ni, no, 1f32));
                edges.push((no, ni, 1f32));
            }
        }
        edges
    }

    #[test]
    fn test_euclidean_dist() {
        let x = Variable::new(vec![1f32, 3f32]);
        let y = Variable::new(vec![3f32, 5f32]);
        let dist = euclidean_distance(x, y);
        assert_eq!(dist.value(), &[(8f32).powf(0.5)]);
    }

    #[test]
    fn test_l2norm() {
        let x = Variable::new(vec![1f32, 3f32]);
        let norm = il2norm(x);
        let denom = 10f32.powf(0.5);
        assert_eq!(norm.value(), &[1f32 / denom, 3f32 / denom]);
    }

    #[test]
    fn test_simple_learn_dist() {
        let edges = build_star_edges();
        let csr = CSR::construct_from_edges(edges);
        let ccsr = CumCSR::convert(csr);
        
        let mut feature_store = FeatureStore::new(ccsr.len(), "feat".to_string());
        feature_store.fill_missing_nodes();

        let mut agraph = Graph::new();

        let ep = EmbeddingPropagation {
            alpha: 1e-2,
            gamma: 0.1,
            wd: 0f32,
            loss: Loss::MarginLoss(1f32, 1usize),
            batch_size: 32,
            dims: 5,
            passes: 50,
            max_features: None,
            max_nodes: None,
            seed: 202220222,
            indicator: false
        };

        let embeddings = ep.learn_feature_embeddings(&ccsr, &mut agraph, &feature_store);
        for idx in 0..embeddings.len() {
            let e = embeddings.get_embedding(idx);
            println!("{:?} -> {:?}", idx, e);
        }
    }

}
