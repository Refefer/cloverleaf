//! Defines the embedding propagation algorithm.  It needs to be rebranded since the actual paper
//! defines it around a loss which we instead parameterize.  Maybe EmbeddingLearner or something
//! generic?
mod optimizer;
mod node_sampler;
pub mod loss;
pub mod model;
pub mod attention;
mod scheduler;

use std::fmt::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use hashbrown::HashMap;
use std::collections::{HashMap as CHashMap};
use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_xorshift::XorShiftRng;
use simple_grad::*;

use crate::graph::{Graph as CGraph,NodeID};
use crate::embeddings::{EmbeddingStore,Distance};
use crate::progress::CLProgressBar;
use crate::feature_store::FeatureStore;

use self::optimizer::{Optimizer,AdamOptimizer};
use self::node_sampler::*;
use self::loss::*;
use self::model::{Model,NodeCounts};
use self::scheduler::LRScheduler;

/// Defines the propagator
pub struct EmbeddingPropagation {
    /// Learning rate for updating feature embeddings
    pub alpha: f32,

    /// Loss to minimize.
    pub loss: Loss,

    /// Batch size to use.  Larger batches have fewer updates, but also lower variance
    pub batch_size: usize,

    /// Sice of the node embeddings.  Feature embeddings can be larger if they use attention
    pub d_model: usize,

    /// Number of passes to optimize for
    pub passes: usize,

    /// Whether we use hard negatives or not.  We might strip this out since I've had difficulty
    /// using it to improve test loss
    pub hard_negs: usize,

    /// Random seed
    pub seed: u64,

    /// We split out valid_pct of nodes to use for validation.
    pub valid_pct: f32,

    /// If added, we add noise to the gradients as a way to regularize the results; this can be
    /// useful when the model overfits and validation start to diverge.
    pub noise: f32,

    /// Whether to show a pretty indicator
    pub indicator: bool
}

impl EmbeddingPropagation {

    /// Learns the feature embeddings.
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

        let dims = model.feature_dims(self.d_model);
        let feature_embeddings = if let Some(embs) = feature_embeddings {
            embs
        } else {
            let mut fe = EmbeddingStore::new(features.num_features(), dims, Distance::Cosine);
            // Initialize embeddings as random
            randomize_embedding_store(&mut fe, &mut rng);
            fe
        };

        // Initializer SGD optimizer.  Right now we hard code the parameters for the optimizer but
        // in the future we could allow for this to be parameterized.
        let optimizer = AdamOptimizer::new(0.9, 0.999,
            feature_embeddings.dims(), 
            feature_embeddings.len()); 

        // Pull out validation idxs;
        let mut node_idxs: Vec<_> = (0..graph.len()).into_iter().collect();
        node_idxs.shuffle(&mut rng);
        let valid_idx = (graph.len() as f32 * self.valid_pct) as usize;
        let valid_idxs = node_idxs.split_off(graph.len() - valid_idx);

        // Number of update stpes
        let steps_per_pass = (node_idxs.len() as f32 / self.batch_size as f32) as usize;

        let pb = CLProgressBar::new((self.passes * steps_per_pass) as u64, self.indicator);
        
        // Enable/disable shared memory pool
        use_shared_pool(true);

        let total_updates = steps_per_pass * self.passes;
        let lr_scheduler = {
            let warm_up_steps = (total_updates as f32 / 5f32) as usize;
            let max_steps = self.passes * steps_per_pass;
            LRScheduler::cos_decay(self.alpha / 100f32, self.alpha, warm_up_steps, max_steps)
        };

        // Noise 
        let noise_scheduler = if self.noise > 1e-9 {
            let min_noise = 1e-9f32;
            let decay = (((min_noise.ln()) - self.noise.ln()) / (total_updates as f32)).exp();
            LRScheduler::exp_decay(0.0, self.noise, decay)
        } else {
            LRScheduler::noop()
        };

        // Initialize samplers for negatives.
        let random_sampler = node_sampler::RandomWalkHardStrategy::new(self.hard_negs, &node_idxs);
        let valid_random_sampler = node_sampler::RandomWalkHardStrategy::new(self.hard_negs, &valid_idxs);

        let mut last_error = std::f32::INFINITY;
        let step = AtomicUsize::new(1);
        let mut valid_error = std::f32::INFINITY;
        
        for pass in 1..(self.passes + 1) {

            pb.update_message(|msg| {
                msg.clear();
                let cur_step = step.load(Ordering::Relaxed);
                let alpha = lr_scheduler.compute(cur_step);
                let noise = noise_scheduler.compute(cur_step);
                write!(msg, "Pass {}/{}, Train: {:.5}, Valid: {:.5}, LR: {:.5}, Noise: {:.5}", pass, self.passes, 
                       last_error, valid_error, alpha, noise)
                    .expect("Error writing out indicator message!");
            });

            if pass % 10 == 0 {
                println!();
            }

            // Shuffle for SGD
            node_idxs.shuffle(&mut rng);
            let err: Vec<_> = node_idxs.par_iter().chunks(self.batch_size).enumerate().map(|(i, nodes)| {

                let mut grads = Vec::with_capacity(self.batch_size);
                
                let sampler = (&random_sampler).initialize_batch(
                    &nodes,
                    graph,
                    features);
                
                // Compute grads for batch
                nodes.par_iter().map(|node_id| {
                    let mut rng = XorShiftRng::seed_from_u64(self.seed + (i + **node_id) as u64);
                    let (loss, hv_vars, thv_vars, hu_vars) = self.run_forward_pass(
                        graph, **node_id, &features, &feature_embeddings, 
                        model, &sampler, &mut rng);

                    let grads = self.extract_gradients(&loss, hv_vars, thv_vars, hu_vars);
                    (loss.value()[0], grads)
                }).collect_into_vec(&mut grads);

                let mut error = 0f32;
                let mut cnt = 0f32;
                
                // We are using std Hashmap instead of hashbrown due to a weird bug
                // where the optimizer, for whatever reason, has trouble draining it
                // on 0.13.  We'll keep testing it on subsequent fixes but until then
                // std is the way to go.
                let mut all_grads = CHashMap::new();

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

                let cur_step = step.fetch_add(1, Ordering::Relaxed);

                // Add gaussian noise to help regulate embeddings
                if cnt > 0f32 {
                    if self.noise > 0.0 {
                        let noise = noise_scheduler.compute(cur_step);
                        all_grads.par_iter_mut().for_each(|(feat, emb)| {
                            let mut rng = XorShiftRng::seed_from_u64(self.seed + (i + feat) as u64);
                            emb.iter_mut().for_each(|ei| {
                                *ei += noise * rng.sample::<f32,StandardNormal>(StandardNormal);
                            });
                        });
                    }

                    // Backpropagate embeddings
                    let alpha = lr_scheduler.compute(cur_step);
                    optimizer.update(&feature_embeddings, all_grads, alpha, pass as f32);
                }

                // Update progress bar
                pb.inc(1);
                if cnt > 0f32 {
                    error / cnt
                } else {
                    0f32
                }
            }).collect();

            // Some losses go toward infinity.  This is a bug we should fix.
            last_error = err.iter()
                .filter(|x| !x.is_infinite() )
                .sum::<f32>() / err.len() as f32;
            
            if valid_idxs.len() > 0 {
                // Validate.  We use the same random seed for consistency across iterations.
                let valid_errors = valid_idxs.par_iter().chunks(self.batch_size).map(|nodes| {
                    let sampler = (&valid_random_sampler).initialize_batch(&nodes, graph, features);

                    nodes.par_iter().map(|node_id| {
                        let mut rng = XorShiftRng::seed_from_u64(self.seed - 1);
                        let loss = self.run_forward_pass(
                            graph, **node_id, &features, &feature_embeddings, 
                            model, &sampler, &mut rng).0;

                        loss.value()[0]
                    }).sum::<f32>()
                }).sum::<f32>();
                
                valid_error = valid_errors / valid_idxs.len() as f32;
            }
        }
        pb.finish();
        feature_embeddings
    }

    fn run_forward_pass<G: CGraph + Send + Sync, R: Rng, S: NodeSampler, M: Model>(
        &self, 
        graph: &G,
        node: NodeID,
        features: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        model: &M,
        sampler: &S,
        rng: &mut R
    ) -> (ANode, NodeCounts, NodeCounts, Vec<NodeCounts>) {
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
        sampler.sample_negatives(graph, node, &mut negatives, num_negs, rng);
        
        let mut hu_vars = Vec::with_capacity(negatives.len());
        let mut hus = Vec::with_capacity(negatives.len());
        negatives.into_iter().for_each(|neg_node| {
            let (hu_var, hu) = model.construct_node_embedding(neg_node, features, &feature_embeddings, rng);
            hu_vars.push(hu_var);
            hus.push(hu);
        });

        // Compute error
        let loss = self.loss.compute(thv, hv.clone(), &hus);

        (loss, hv_vars, thv_vars, hu_vars)

    }

    fn extract_gradients(
        &self, 
        loss: &ANode,
        hv_vars: NodeCounts,
        thv_vars: NodeCounts,
        hu_vars: Vec<NodeCounts>
    ) -> HashMap<usize, Vec<f32>> {

        // Compute gradients
        let mut agraph = Graph::new();

        // This call is about 90% of the entire cost of the algorithm.  The compute graph is
        // incredibly deep when using attention and requires the computation of several gigabytes
        // of gradients.
        // Non-attention methods are substantially simpler to compute the gradients for and are two
        // orders of a magnitude lower.
        agraph.backward(&loss);

        let mut grads = HashMap::new();
        extract_grads(&agraph, &mut grads, hv_vars.into_iter());
        extract_grads(&agraph, &mut grads, thv_vars.into_iter());
        hu_vars.into_iter().for_each(|hu_var| {
            extract_grads(&agraph, &mut grads, hu_var.into_iter());
        });

        grads

    }

}

/// We extract the gradients for each unique feature
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

// Randomize embeddings.  Might make more sense to have in the embedding file.
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
    fn test_simple_learn_dist() {
        let edges = build_star_edges();
        let csr = CSR::construct_from_edges(edges);
        let ccsr = CumCSR::convert(csr);
        
        let mut feature_store = FeatureStore::new(ccsr.len(), "feat".to_string());
        feature_store.fill_missing_nodes();

        let model = super::model::AveragedFeatureModel::new(None, None);
        let ep = EmbeddingPropagation {
            alpha: 1e-2,
            loss: Loss::MarginLoss(1f32, 1usize),
            batch_size: 32,
            hard_negs: 0,
            d_model: 5,
            valid_pct: 0.0,
            passes: 50,
            noise: 0.0,
            seed: 202220222,
            indicator: false
        };

        let embeddings = ep.learn_feature_embeddings(&ccsr, &feature_store, None, &model);
        for idx in 0..embeddings.len() {
            let e = embeddings.get_embedding(idx);
            println!("{:?} -> {:?}", idx, e);
        }
    }

}
