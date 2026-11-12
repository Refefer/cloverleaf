//! Defines the embedding propagation algorithm.  It needs to be rebranded since the actual paper
//! defines it around a loss which we instead parameterize.  Maybe EmbeddingLearner or something
//! generic?
pub mod loss;
pub mod model;
pub mod attention;

use std::fmt::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use hashbrown::HashMap;
use std::collections::{HashMap as CHashMap};
use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_xorshift::XorShiftRng;
use crate::autograd::*;

use crate::graph::{Graph as CGraph,NodeID};
use crate::embeddings::EmbeddingStore;
use crate::distance::Distance;
use crate::progress::CLProgressBar;
use crate::feature_store::FeatureStore;
use crate::algos::utils::node_seed;
use crate::algos::grad_utils::scheduler::LRScheduler;
use crate::algos::grad_utils::optimizer::{Optimizer,AdamOptimizer,OptimizerKind};
use crate::algos::grad_utils::node_sampler::*;

use self::loss::*;
use self::model::{Model,NodeCounts};

/// How the learning rate changes over training.
#[derive(Clone,Copy,Debug,PartialEq)]
pub enum LearningRateSchedule {
    /// Linear warmup from alpha / 100 over the first 20% of updates, then cosine decay back to
    /// alpha / 100 by the last update.
    CosineWarmup,

    /// alpha throughout, as in the EP paper.
    Constant
}

/// How feature embeddings are initialised when no warm start is given.
#[derive(Clone,Copy,Debug,PartialEq)]
pub enum EmbeddingInit {
    /// Each coordinate uniform in [-1, 1), then the embedding is scaled to unit length.
    UnitNorm,

    /// Glorot (Xavier) uniform, as in the EP paper: each coordinate from U(-l, l) with
    /// l = sqrt(6 / (num_features + dims)).
    GlorotUniform
}

#[derive(Clone,Copy,Debug)]
pub enum LossWeighting {
    DegreeLog,
    DegreeExponential(f32),
    None
}

/// Defines the propagator
#[derive(Debug)]
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

    /// Reweights losses according to the number of outbound edges they have
    pub loss_weighting: LossWeighting,

    /// Random seed
    pub seed: u64,

    /// We split out valid_pct of nodes to use for validation.
    pub valid_pct: f32,

    /// If added, we add noise to the gradients as a way to regularize the results; this can be
    /// useful when the model overfits and validation start to diverge.
    pub noise: f32,

    /// Whether to show a pretty indicator
    pub indicator: bool,

    /// Optimizer for the feature embeddings
    pub optimizer: OptimizerKind,

    /// Learning rate schedule
    pub lr_schedule: LearningRateSchedule,

    /// Initialisation of the feature embeddings when no warm start is given
    pub init: EmbeddingInit
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
            initialize_embedding_store(&mut fe, self.init, &mut rng);
            fe
        };

        let optimizer = AdamOptimizer::new(self.optimizer,
            feature_embeddings.dims(),
            feature_embeddings.len());

        // Pull out validation idxs;
        let mut node_idxs: Vec<_> = (0..graph.len()).into_iter().collect();
        node_idxs.shuffle(&mut rng);
        let valid_idx = (graph.len() as f32 * self.valid_pct) as usize;
        let valid_idxs = node_idxs.split_off(graph.len() - valid_idx);

        // Number of update stpes
        let steps_per_pass = (node_idxs.len() as f32 / self.batch_size as f32).ceil() as usize;

        let pb = CLProgressBar::new((self.passes * steps_per_pass) as u64, self.indicator);
        
        // Enable/disable shared memory pool
        use_shared_pool(false);

        let total_updates = steps_per_pass * self.passes;
        let lr_scheduler = match self.lr_schedule {
            LearningRateSchedule::CosineWarmup => {
                let warm_up_steps = (total_updates as f32 / 5f32) as usize;
                let max_steps = self.passes * steps_per_pass;
                LRScheduler::cos_decay(self.alpha / 100f32, self.alpha, warm_up_steps, max_steps)
            },
            LearningRateSchedule::Constant => LRScheduler::constant(self.alpha)
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
        let valid_random_sampler = RandomWalkHardStrategy::new(self.hard_negs, &valid_idxs);

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
            // Borrows the shuffled node list for easy negatives, so it's rebuilt every pass.
            let random_sampler = RandomWalkHardStrategy::new(self.hard_negs, &node_idxs);
            let err_cnt: (f32, usize) = node_idxs.par_iter().chunks(self.batch_size).enumerate().map(|(i, nodes)| {

                let sampler = (&random_sampler).initialize_batch(
                    &nodes,
                    graph,
                    features);
                
                let n_nodes = nodes.len();
                // Compute grads for batch
                let grads: Vec<_> = nodes.par_iter().filter_map(|node_id| {
                    let n_id = **node_id;
                    let mut rng = XorShiftRng::seed_from_u64(node_seed(self.seed, pass, i, n_id));
                    let (mut loss, hv_vars, thv_vars, hu_vars) = self.run_forward_pass(
                        graph, n_id, &features, &feature_embeddings, 
                        model, &sampler, &mut rng);

                    loss = match self.loss_weighting {
                        LossWeighting::DegreeLog => {
                            let decrease = (1f32 + graph.degree(n_id) as f32).ln();
                            loss / decrease
                        },
                        LossWeighting::DegreeExponential(weight) => {
                            let decrease = (graph.degree(n_id) as f32).powf(weight);
                            loss / decrease
                        },
                        LossWeighting::None => { loss }
                    };

                    let loss_value = loss.value()[0];

                    // Sometimes there are weird errors due to underflows in softmax
                    // In this case, just print the graph and don't return
                    if loss_value.is_nan() {
                        Graph::print_graph(&loss);
                        None
                    } else if loss_value > 0f32 {
                        let grads = self.extract_gradients(&loss, hv_vars, thv_vars, hu_vars);
                        Some((loss_value, grads))
                    } else {
                        None
                    }
                }).collect();

                let cnt = grads.len();
                
                // We are using std Hashmap instead of hashbrown due to a weird bug
                // where the optimizer, for whatever reason, has trouble draining it
                // on 0.13.  We'll keep testing it on subsequent fixes but until then
                // std is the way to go.
                let mut all_grads = CHashMap::new();

                // Since we're dealing with multiple reconstructions with likely shared features,
                // we aggregate all the gradients
                let error = grads.into_iter().map(|(err, grad_set)| {
                    for (feat, grad) in grad_set.into_iter() {
                        let e = all_grads.entry(feat).or_insert_with(|| vec![0.; grad.len()]);
                        e.iter_mut().zip(grad.iter()).for_each(|(ei, gi)| *ei += *gi);
                    }
                    err
                }).sum::<f32>();

                let cur_step = step.fetch_add(1, Ordering::Relaxed);

                if cnt > 0 {
                    // Add gaussian noise to help regulate embeddings
                    if self.noise > 0.0 {
                        let noise = noise_scheduler.compute(cur_step);
                        all_grads.par_iter_mut().for_each(|(feat, emb)| {
                            let mut rng = XorShiftRng::seed_from_u64(node_seed(self.seed, pass, i, *feat));
                            emb.iter_mut().for_each(|ei| {
                                *ei += noise * rng.sample::<f32,StandardNormal>(StandardNormal);
                            });
                        });
                    }

                    // Backpropagate embeddings
                    let alpha = lr_scheduler.compute(cur_step);
                    optimizer.update(&feature_embeddings, all_grads, alpha, cur_step as f32);
                }

                // Update progress bar
                pb.inc(1);
                if cnt > 0 {
                    error / n_nodes as f32
                } else {
                    0f32
                }
            })
            .map(|x| { if x.is_infinite() { (0f32, 1usize) } else { (x, 1usize) } })
            .reduce(|| (0f32, 0usize), |a, b| (a.0 + b.0, a.1 + b.1));

            last_error = err_cnt.0 / if err_cnt.1 > 0 { err_cnt.1 as f32} else { 1f32 };
            
            if valid_idxs.len() > 0 {
                // Validate.  We use the same random seed for consistency across iterations.
                let valid_errors = valid_idxs.par_iter().chunks(self.batch_size).map(|nodes| {
                    let sampler = (&valid_random_sampler).initialize_batch(&nodes, graph, features);

                    nodes.par_iter().map(|node_id| {
                        let mut rng = XorShiftRng::seed_from_u64(self.seed.wrapping_sub(1));
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
            node, 1f32, features, &feature_embeddings, rng);
        
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
            let (hu_var, hu) = model.construct_node_embedding(neg_node, 1f32, features, &feature_embeddings, rng);
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

/// We extract the gradients for each unique feature.  The same feature can appear as a separate
/// leaf in h(v), ~h(v) and each h(u), so its total gradient is the sum over every leaf.
pub fn extract_grads(
    graph: &Graph, 
    grads: &mut HashMap<usize, Vec<f32>>, 
    vars: impl Iterator<Item=(usize, (ANode, f32))>
) {
    for (feat_id, (var, _)) in vars {
        if let Some(grad) = graph.get_grad(&var) {
            if grad.iter().all(|gi| !(gi.is_nan() || gi.is_infinite())) {
                // Can get some nans in weird cases, such as the distance between
                // a node and it's reconstruction when it shares all features.
                // Since that's not all that helpful anyways, we simply ignore it and move on
                match grads.get_mut(&feat_id) {
                    Some(acc) => acc.iter_mut().zip(grad.iter()).for_each(|(a, g)| *a += *g),
                    None => { grads.insert(feat_id, grad); }
                }
            }
        }
    }
}

// Randomize embeddings.  Might make more sense to have in the embedding file.
fn initialize_embedding_store(es: &mut EmbeddingStore, init: EmbeddingInit, rng: &mut impl Rng) {
    match init {
        EmbeddingInit::UnitNorm => randomize_embedding_store(es, rng),
        EmbeddingInit::GlorotUniform => {
            let limit = (6f32 / (es.len() + es.dims()) as f32).sqrt();
            for idx in 0..es.len() {
                es.get_embedding_mut(idx).iter_mut().for_each(|ei| {
                    *ei = limit * (2f32 * rng.gen::<f32>() - 1f32);
                });
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


#[cfg(test)]
mod ep_tests {
    use super::*;
    use crate::algos::utils::Sample;

    /// A feature can appear as a separate autograd leaf in h(v), ~h(v) and each h(u); its
    /// gradient is the sum over all of them.
    #[test]
    fn test_extract_grads_sums_shared_features() {
        let a = Variable::new(vec![1., 2.]);
        let b = Variable::new(vec![1., 2.]);
        let c = Variable::new(vec![5., 5.]);
        let loss = a.sum() + b.sum() * 3f32 + c.sum() * 2f32;

        let mut graph = Graph::new();
        graph.backward(&loss);

        let mut grads = HashMap::new();
        let first: Vec<(usize, (ANode, f32))> = vec![(7, (a, 1.))];
        let second: Vec<(usize, (ANode, f32))> = vec![(7, (b, 1.)), (9, (c, 1.))];
        extract_grads(&graph, &mut grads, first.into_iter());
        extract_grads(&graph, &mut grads, second.into_iter());

        assert_eq!(grads[&7], vec![4., 4.]);
        assert_eq!(grads[&9], vec![2., 2.]);
    }
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
        let csr = CSR::construct_from_edges(edges, false);
        let ccsr = CumCSR::convert(csr);
        
        let mut feature_store = FeatureStore::new(ccsr.len());
        feature_store.fill_missing_nodes();

        let model = super::model::AveragedFeatureModel::new(Sample::All, Sample::All, false, false);
        let ep = EmbeddingPropagation {
            alpha: 1e-2,
            loss: Loss::MarginLoss(1f32, 1usize),
            batch_size: 32,
            hard_negs: 0,
            loss_weighting: LossWeighting::None,
            d_model: 5,
            valid_pct: 0.0,
            passes: 50,
            noise: 0.0,
            seed: 202220222,
            indicator: false,
            optimizer: OptimizerKind::Adam,
            lr_schedule: LearningRateSchedule::CosineWarmup,
            init: EmbeddingInit::UnitNorm
        };

        let embeddings = ep.learn_feature_embeddings(&ccsr, &feature_store, None, &model);
        for idx in 0..embeddings.len() {
            let e = embeddings.get_embedding(idx);
            println!("{:?} -> {:?}", idx, e);
        }
    }

    /// Glorot uniform draws every coordinate from U(-l, l) with l = sqrt(6 / (rows + dims)).
    #[test]
    fn test_glorot_uniform_init() {
        let (rows, dims) = (1000, 128);
        let mut es = EmbeddingStore::new(rows, dims, Distance::Cosine);
        let mut rng = XorShiftRng::seed_from_u64(0);
        initialize_embedding_store(&mut es, EmbeddingInit::GlorotUniform, &mut rng);
        let limit = (6f32 / (rows + dims) as f32).sqrt();
        let mut max_abs = 0f32;
        for idx in 0..rows {
            for v in es.get_embedding(idx) {
                assert!(v.abs() <= limit, "{} outside +-{}", v, limit);
                max_abs = max_abs.max(v.abs());
            }
        }
        // The draws should fill the range, not collapse near zero.
        assert!(max_abs > 0.99 * limit, "max |v| {} for limit {}", max_abs, limit);
    }

    /// The default initialisation gives every embedding unit length.
    #[test]
    fn test_unit_norm_init() {
        let mut es = EmbeddingStore::new(10, 16, Distance::Cosine);
        let mut rng = XorShiftRng::seed_from_u64(0);
        initialize_embedding_store(&mut es, EmbeddingInit::UnitNorm, &mut rng);
        for idx in 0..10 {
            let norm = es.get_embedding(idx).iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1f32).abs() < 1e-5, "row {} has norm {}", idx, norm);
        }
    }

}
