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


#[derive(Debug)]
pub struct FeatureStore {
    features: Vec<Vec<usize>>,
    feature_vocab: Vocab,
    empty_nodes: usize
}

impl FeatureStore {

    pub fn new(size: usize) -> Self {
        FeatureStore {
            features: vec![Vec::with_capacity(0); size],
            feature_vocab: Vocab::new(),
            empty_nodes: 0
        }
    }

    pub fn set_features(&mut self, node: NodeID, node_features: Vec<String>) {
        self.features[node] = node_features.into_iter()
            .map(|f| self.feature_vocab.get_or_insert("feat".to_string(), f))
            .collect()
    }

    pub fn get_features(&self, node: NodeID) -> &[usize] {
        &self.features[node]
    }

    pub fn len(&self) -> usize {
        self.feature_vocab.len() + self.empty_nodes
    }

    pub fn fill_missing_nodes(&mut self) {
        let mut idxs = self.feature_vocab.len();
        self.features.iter_mut().for_each(|f| {
            if f.len() == 0 {
                *f = vec![idxs];
                idxs += 1;
                self.empty_nodes += 1;
            }
        });
    }

    pub fn get_vocab(self) -> Vocab {
        self.feature_vocab
    }
}

pub struct EmbeddingPropagation {
    pub alpha: f32,
    pub gamma: f32,
    pub loss: Loss,
    pub batch_size: usize,
    pub dims: usize,
    pub passes: usize,
    pub seed: u64,
    pub indicator: bool
}

impl EmbeddingPropagation {

    pub fn learn<G: CGraph + Send + Sync>(
        &self, 
        graph: &G, 
        features: &FeatureStore
    ) -> (EmbeddingStore, EmbeddingStore) {
        let mut agraph = Graph::new();
        let feat_embeds = self.learn_feature_embeddings(graph, &mut agraph, features);
        let es = self.construct_node_embeddings(graph.len(), features, &feat_embeds);
        (es, feat_embeds)
    }

    fn construct_node_embeddings(
        &self, 
        num_nodes: usize, 
        features: &FeatureStore, 
        feat_embeds: &EmbeddingStore
    ) -> EmbeddingStore {
        let mut es = EmbeddingStore::new(num_nodes, self.dims, Distance::Cosine);
        (0..num_nodes).into_par_iter().for_each(|node| {
            let node_embedding = construct_node_embedding(node, features, &feat_embeds).1;
            // Safe to access in parallel
            let embedding = es.get_embedding_mut_hogwild(node);
            embedding.clone_from_slice(node_embedding.value());
        });
        es
    }

    fn learn_feature_embeddings<G: CGraph + Send + Sync>(
        &self,
        graph: &G,
        agraph: &mut Graph,
        features: &FeatureStore,
    ) -> EmbeddingStore {

        let mut feature_embeddings = EmbeddingStore::new(features.len(), self.dims, Distance::Cosine);
        let momentum = EmbeddingStore::new(features.len(), self.dims, Distance::Cosine);
        let mut rng = XorShiftRng::seed_from_u64(self.seed);
        randomize_embedding_store(&mut feature_embeddings, &mut rng);

        let mut node_idxs: Vec<_> = (0..graph.len()).into_iter().collect();
        let dist = Uniform::new(0, node_idxs.len());
        let pb = CLProgressBar::new((self.passes * graph.len()) as u64, self.indicator);
        
        // Enable/disable shared memory pool
        use_shared_pool(true);

        let mut current_error = std::f32::INFINITY;
        for pass in 0..self.passes {
            pb.update_message(|msg| {
                msg.clear();
                write!(msg, "Pass {}/{}, Error: {}", pass + 1, self.passes, current_error)
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

                // Back propagate and SGD
                let mut error = 0f32;
                let mut cnt = 0f32;
                for (err, grad_set) in grads.drain(..nodes.len()) {
                    for (feat, grad) in grad_set.into_iter() {
                        let e = all_grads.entry(feat).or_insert_with(|| vec![0.; grad.len()]);
                        e.iter_mut().zip(grad.iter()).for_each(|(ei, gi)| *ei += *gi);
                    }
                    error += err;
                    cnt += 1f32;
                }
                
                // Backpropagate embeddings
                sgd(&feature_embeddings, &momentum, self.gamma, &mut all_grads, self.alpha);
                // Update progress bar
                pb.inc(nodes.len() as u64);
                error / cnt
            }).collect();

            current_error = err.iter().sum::<f32>() / err.len() as f32;
        }
        pb.finish();
        feature_embeddings
    }

    fn run_pass<G: CGraph + Send + Sync, R: Rng>(
        &self, 
        graph: &G,
        node: NodeID,
        features: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut R
    ) -> (f32, HashMap<usize, Vec<f32>>) {

        let dist = Uniform::new(0, graph.len());

        // Get negative v
        let num_negs = self.loss.negatives();
        let mut negatives = Vec::with_capacity(num_negs);
        loop {
            let neg_node = dist.sample(rng);
            if neg_node != node { 
                negatives.push(neg_node) ;
                if negatives.len() == num_negs { break }
            }
        }

        // h(v)
        let (hv_vars, hv) = construct_node_embedding(node, features, &feature_embeddings);
        
        // ~h(v)
        let (thv_vars, thv) = reconstruct_node_embedding(graph, node, features, &feature_embeddings, Some(10));
        
        // h(u)
        let mut hu_vars = Vec::with_capacity(num_negs);
        let mut hus = Vec::with_capacity(num_negs);
        negatives.into_iter().for_each(|neg_node| {
            let (hu_var, hu) = construct_node_embedding(neg_node, features, &feature_embeddings);
            hu_vars.push(hu_var);
            hus.push(hu);
        });

        // Compute error
        let loss = self.loss.compute(thv, hv, hus);

        let mut agraph = Graph::new();
        agraph.backward(&loss);

        let mut grads = HashMap::new();
        extract_grads(&agraph, &mut grads, hv_vars.into_iter());
        extract_grads(&agraph, &mut grads, thv_vars.into_iter());
        hu_vars.into_iter().for_each(|hu_var| {
            extract_grads(&agraph, &mut grads, hu_var.into_iter());
        });

        (loss.value()[0], grads)

    }

}

fn extract_grads(
    graph: &Graph, 
    grads: &mut HashMap<usize, Vec<f32>>, 
    vars: impl Iterator<Item=(usize, (ANode, usize))>
) {
    for (feat_id, (var, _)) in vars {
        if grads.contains_key(&feat_id) { continue }

        let grad = graph.get_grad(&var)
            .expect("Should have a gradient!");

        if grad.iter().all(|gi| !gi.is_nan()) {
            // Can get some nans in weird cases, such as the distance between
            // a node and it's reconstruction when it shares all features.
            // SGD
            grads.insert(feat_id, grad.to_vec());
        }
    }
}

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
            // SGD
            emb.iter_mut().zip(grad.iter().zip(mom.iter_mut())).for_each(|(ei, (gi, mi))| {
                *mi = gamma * *mi + *gi;
                *ei -= alpha * *mi;
            });
        }
    }

}

type NodeCounts = HashMap<usize, (ANode, usize)>;

fn collect_embeddings_from_node(
    node: NodeID,
    feature_store: &FeatureStore,
    feature_embeddings: &EmbeddingStore,
    feat_map: &mut NodeCounts  
) {
   for feat in feature_store.get_features(node).iter() {
        if let Some((_emb, count)) = feat_map.get_mut(feat) {
            *count += 1;
        } else {
            let emb = feature_embeddings.get_embedding(*feat);
            let v = Variable::new(emb.to_vec());
            feat_map.insert(*feat, (v, 1));
        }
    }
}

// H(n)
fn construct_node_embedding(
    node: NodeID,
    feature_store: &FeatureStore,
    feature_embeddings: &EmbeddingStore,
) -> (NodeCounts, ANode) {
    let mut feature_map = HashMap::new();
    collect_embeddings_from_node(node, feature_store, 
                                 feature_embeddings, &mut feature_map);
    let mean = mean_embeddings(feature_map.values());
    (feature_map, mean)
}

// ~H(n)
fn reconstruct_node_embedding<G: CGraph>(
    graph: &G,
    node: NodeID,
    feature_store: &FeatureStore,
    feature_embeddings: &EmbeddingStore,
    max_nodes: Option<usize>
) -> (NodeCounts, ANode) {
    let edges = &graph.get_edges(node).0;
    let mut feature_map = HashMap::new();
    for out_node in edges.iter().take(max_nodes.unwrap_or(edges.len())) {
        collect_embeddings_from_node(*out_node, feature_store, 
                                     feature_embeddings, &mut feature_map);
    }
    let mean = mean_embeddings(feature_map.values());
    (feature_map, mean)
}

fn mean_embeddings<'a,I: Iterator<Item=&'a (ANode, usize)>>(items: I) -> ANode {
    let mut vs = Vec::new();
    let mut n = 0;
    items.for_each(|(emb, count)| {
        vs.push(emb * *count as f32);
        n += *count;
    });
    vs.sum_all() / n as f32
}

fn euclidean_distance(e1: ANode, e2: ANode) -> ANode {
    (e1 - e2).pow(2f32).sum().pow(0.5)
}

#[derive(Copy,Clone)]
pub enum Loss {
    MarginLoss(f32),
    Contrastive(f32, usize)
}

impl Loss {
    fn negatives(&self) -> usize {
        if let Loss::Contrastive(_, negs) = self {
            *negs
        } else {
            1
        }
    }

    fn compute(&self, thv: ANode, hv: ANode, mut hus: Vec<ANode>) -> ANode {
        match self {
            Loss::MarginLoss(gamma) => {
                let d1 = euclidean_distance(thv.clone(), hv);
                let d2 = euclidean_distance(thv, hus.pop().unwrap());
                (gamma + d1 - d2).maximum(0f32)
            },
            Loss::Contrastive(tau, _) => {
                let thv_norm = l2norm(thv);
                let hv_norm  = l2norm(hv);

                let mut ds: Vec<_> = hus.into_iter().map(|hu| {
                    let hu_norm = l2norm(hu);
                    (cosine(thv_norm.clone(), hu_norm) / *tau).exp()
                }).collect();

                let d1 = cosine(thv_norm, hv_norm) / *tau;
                let d1_exp = d1.exp();
                ds.push(d1_exp.clone());
                -(d1_exp / ds.sum_all()).ln()
            }
        }
    }
}

fn l2norm(v: ANode) -> ANode {
    &v / (&v).pow(2f32).sum().pow(0.5)
}

fn cosine(x1: ANode, x2: ANode) -> ANode {
    x1.dot(&x2)
}

fn margin_loss(thv: ANode, hv: ANode, hu: ANode, gamma: f32) -> ANode {
    let d1 = euclidean_distance(thv.clone(), hv);
    let d2 = euclidean_distance(thv, hu);
    (gamma + d1 - d2).maximum(0f32)
}

fn randomize_embedding_store(es: &mut EmbeddingStore, rng: &mut impl Rng) {
    for idx in 0..es.len() {
        let e = es.get_embedding_mut(idx);
        e.iter_mut().for_each(|ei| *ei = 2f32 * rng.gen::<f32>() - 1f32);
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
        let norm = l2norm(x);
        let denom = 10f32.powf(0.5);
        assert_eq!(norm.value(), &[1f32 / denom, 3f32 / denom]);
    }

    #[test]
    fn test_simple_learn_dist() {
        let edges = build_star_edges();
        let csr = CSR::construct_from_edges(edges);
        let ccsr = CumCSR::convert(csr);
        
        let mut feature_store = FeatureStore::new(ccsr.len());
        feature_store.fill_missing_nodes();

        let mut rng = XorShiftRng::seed_from_u64(202220222);
        let mut agraph = Graph::new();

        let ep = EmbeddingPropagation {
            alpha: 1e-2,
            gamma: 0.1,
            loss: Loss::MarginLoss(1f32),
            batch_size: 32,
            dims: 5,
            passes: 50,
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
