pub mod optimizer;
pub mod node_sampler;
pub mod loss;
pub mod model;

use std::fmt::Write;

use rayon::prelude::*;
use hashbrown::HashMap;
use std::collections::{HashMap as CHashMap};
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use simple_grad::*;

use crate::graph::{Graph as CGraph,NodeID};
use crate::embeddings::{EmbeddingStore,Distance};
use crate::progress::CLProgressBar;
use crate::feature_store::FeatureStore;

use self::optimizer::{Optimizer,AdamOptimizer};
use self::node_sampler::*;
use self::loss::*;
use self::model::Model;

pub struct EmbeddingPropagation {
    pub alpha: f32,
    pub loss: Loss,
    pub batch_size: usize,
    pub dims: usize,
    pub passes: usize,
    pub seed: u64,
    pub indicator: bool
}

impl EmbeddingPropagation {

    pub fn learn<G: CGraph + Send + Sync, M: Model>(
        &self, 
        graph: &G, 
        features: &FeatureStore,
        feature_embeddings: Option<EmbeddingStore>,
        model: &M
    ) -> EmbeddingStore {
        let feat_embeds = self.learn_feature_embeddings(graph, features, feature_embeddings, model);
        feat_embeds
    }
    
    // The uber expensive function
    fn learn_feature_embeddings<G: CGraph + Send + Sync, M: Model>(
        &self,
        graph: &G,
        features: &FeatureStore,
        feature_embeddings: Option<EmbeddingStore>,
        model: &M
    ) -> EmbeddingStore {

        let mut rng = XorShiftRng::seed_from_u64(self.seed);

        let feature_embeddings = if let Some(embs) = feature_embeddings {
            embs
        } else {
            let mut fe = EmbeddingStore::new(features.num_features(), self.dims, Distance::Cosine);
            // Initialize embeddings as random
            randomize_embedding_store(&mut fe, &mut rng);
            fe
        };

        // Initializer SGD optimizer
        let optimizer = AdamOptimizer::new(0.9, 0.999,
            feature_embeddings.dims(), 
            feature_embeddings.len()); 

        let random_sampler = node_sampler::RandomSamplerStrategy::new(graph);

        let pb = CLProgressBar::new((self.passes * graph.len()) as u64, self.indicator);
        
        // Enable/disable shared memory pool
        use_shared_pool(true);

        let mut lr_scheduler = AttentionLRScheduler::new(self.alpha / 100f32, self.alpha, 5);
        let mut alpha = lr_scheduler.update(std::f32::INFINITY, 0);
        let mut node_idxs: Vec<_> = (0..graph.len()).into_iter().collect();
        let mut last_error = std::f32::INFINITY;
        for pass in 1..(self.passes + 1) {
            pb.update_message(|msg| {
                msg.clear();
                write!(msg, "Pass {}/{}, Error: {:.5}, alpha: {:.5}", pass + 1, self.passes, last_error, alpha)
                    .expect("Error writing out indicator message!");
            });

            // Shuffle for SGD
            node_idxs.shuffle(&mut rng);
            let err: Vec<_> = node_idxs.par_iter().chunks(self.batch_size).enumerate().map(|(i, nodes)| {
                let mut grads = Vec::with_capacity(self.batch_size);
                // We are using std Hashmap instead of hashbrown due to a weird bug
                // where the optimizer, for whatever reason, has troubles draining it
                // on 0.13.  We'll keep testing it on subsequent fixes but until then
                // std is the way to go.
                let mut all_grads = CHashMap::new();

                let sampler = (&random_sampler).initialize_batch(
                    graph,
                    features);
                
                // Compute grads for batch
                nodes.par_iter().map(|node_id| {
                    let mut rng = XorShiftRng::seed_from_u64(self.seed + (i + **node_id) as u64);
                    let (loss, grads) = self.run_pass(
                        graph, 
                        **node_id, 
                        &features, 
                        &feature_embeddings, 
                        model,
                        &sampler, 
                        &mut rng) ;
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
                optimizer.update(&feature_embeddings, all_grads, alpha, pass as f32);

                // Update progress bar
                pb.inc(nodes.len() as u64);
                error / cnt
            }).collect();

            let error = err.iter().sum::<f32>() / err.len() as f32;
            alpha = lr_scheduler.update(error, pass);
            last_error = error;
        }
        pb.finish();
        feature_embeddings
    }
    
    fn run_pass<G: CGraph + Send + Sync, R: Rng, S: NodeSampler, M: Model>(
        &self, 
        graph: &G,
        node: NodeID,
        features: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        model: &M,
        sampler: &S,
        rng: &mut R
    ) -> (f32, HashMap<usize, Vec<f32>>) {

        // h(v)
        let (hv_vars, hv) = model.construct_node_embedding(
            node, features, &feature_embeddings, rng);
        
        // ~h(v)
        let (thv_vars, thv) = self.loss.construct_positive(
            graph, node, features, &feature_embeddings, model, rng);
        
        // h(u)
        let num_negs = self.loss.negatives();
        let mut negatives = Vec::with_capacity(num_negs);
        
        // Sample random negatives
        sampler.sample_negatives(node, &mut negatives, num_negs, rng);
        
        let mut hu_vars = Vec::with_capacity(negatives.len());
        let mut hus = Vec::with_capacity(negatives.len());
        negatives.into_iter().for_each(|neg_node| {
            let (hu_var, hu) = model.construct_node_embedding(neg_node, features, &feature_embeddings, rng);
            hu_vars.push(hu_var);
            hus.push(hu);
        });

        // Compute error
        let loss = self.loss.compute(thv, hv.clone(), hus.clone());
        
        // Compute gradients
        let mut agraph = Graph::new();
        agraph.backward(&loss);

        let mut grads = HashMap::new();
        extract_grads(&agraph, &mut grads, hv_vars.into_iter());
        extract_grads(&agraph, &mut grads, thv_vars.into_iter());
        hu_vars.into_iter().for_each(|hu_var| {
            extract_grads(&agraph, &mut grads, hu_var.into_iter());
        });

        let err = loss.value()[0];
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
            if grad.iter().all(|gi| !(gi.is_nan() || gi.is_infinite())) {
                // Can get some nans in weird cases, such as the distance between
                // a node and it's reconstruction when it shares all features.
                // Since that's not all that helpful anyways, we simply ignore it and move on
                grads.insert(feat_id, grad.to_vec());
            }
        }
    }
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

struct LearningRateTracker {
    pass: usize,
    best: f32,
    last: f32,
    delta: usize
}

impl LearningRateTracker {
    fn new(delta: usize) -> Self {
        LearningRateTracker {pass: 0, best: std::f32::MAX, delta, last: std::f32::INFINITY}
    }

    fn update(&mut self, next: f32, cur_pass: usize) {
        if next < self.best {
            self.best = next;
            self.pass = cur_pass;
        }
        self.last = next;
    }

    fn stagnated(&self, cur_pass: usize) -> bool {
        cur_pass - self.pass >= self.delta
    }

    fn reset(&mut self, error: f32, pass: usize) {
        self.best = error;
        self.pass = pass;
    }

    fn last(&self) -> f32 {
        self.last
    }
}

trait LRScheduler {
    fn update(&mut self, cur_error: f32, cur_pass: usize) -> f32;
}

struct LinearDecay {
    tracker: LearningRateTracker,
    alpha: f32,
    decay: f32
}

impl LinearDecay {
    fn new(alpha: f32, decay: f32, stagnation_epochs: usize) -> Self {
        LinearDecay {
            tracker: LearningRateTracker::new(stagnation_epochs),
            alpha: alpha,
            decay: decay
        }
    }
}

impl LRScheduler for LinearDecay {
    fn update(&mut self, cur_error: f32, cur_pass: usize) -> f32 {
        self.tracker.update(cur_error, cur_pass);
        if self.tracker.stagnated(cur_pass) {
            self.alpha = self.alpha * self.decay;
            self.tracker.reset(cur_error, cur_pass);
        }
        self.alpha
    }
}

struct AttentionLRScheduler {
    init_alpha: f32,
    alpha: f32,
    warmup: usize
}

impl AttentionLRScheduler {
    fn new(init_alpha: f32, alpha: f32, warmup: usize) -> Self {
        AttentionLRScheduler { init_alpha, alpha, warmup }
    }
}

impl LRScheduler for AttentionLRScheduler {
    fn update(&mut self, _cur_error: f32, cur_pass: usize) -> f32 {
        if cur_pass <= self.warmup {
            self.init_alpha
        } else {
            self.alpha / (1f32 + (cur_pass - self.warmup) as f32).sqrt()
        }
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
