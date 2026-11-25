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
use crate::geometry::Geometry;
use crate::progress::CLProgressBar;
use crate::feature_store::{FeatureStore, FeatureSet};
use crate::embeddings::NodeEmbeddings;
use crate::error::Error;
use crate::named_graph::NamedGraph;
use std::sync::Arc;
use crate::algos::utils::node_seed;
use crate::algos::grad_utils::scheduler::LRScheduler;
use crate::algos::grad_utils::optimizer::{Optimizer,AdamOptimizer,OptimizerKind};
use crate::algos::grad_utils::node_sampler::*;
use crate::algos::grad_utils::accumulator::GradientAccumulator;

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

/// How often the feature embeddings are updated.
#[derive(Clone,Copy,Debug,PartialEq)]
pub enum UpdateFrequency {
    /// After every batch: batches run in parallel and each applies its own update as it
    /// finishes.
    Batch,

    /// Once per epoch (pass): every batch computes its gradients against the same embeddings,
    /// they are summed over the whole pass, and one update applies the sum at the end.  Lower
    /// variance, but only `passes` updates in total, so it generally needs a larger learning rate
    /// or more passes.  Keeps an extra gradient table the size of the feature embeddings.
    Epoch
}

fn parse_choice<T: Copy>(what: &str, value: &str, choices: &[(&str, T)]) -> Result<T, Error> {
    choices.iter().find(|(name, _)| *name == value).map(|(_, v)| *v).ok_or_else(|| {
        let names: Vec<String> = choices.iter().map(|(n, _)| format!("{:?}", n)).collect();
        Error::InvalidArgument(format!("{} must be {}, got {:?}", what, names.join(" or "), value))
    })
}

/// "cosine" or "constant".
impl std::str::FromStr for LearningRateSchedule {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        parse_choice("lr_schedule", s, &[("cosine", LearningRateSchedule::CosineWarmup),
                                         ("constant", LearningRateSchedule::Constant)])
    }
}

/// "unit" or "glorot".
impl std::str::FromStr for EmbeddingInit {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        parse_choice("init", s, &[("unit", EmbeddingInit::UnitNorm), ("glorot", EmbeddingInit::GlorotUniform)])
    }
}

/// "batch" or "epoch".
impl std::str::FromStr for UpdateFrequency {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        parse_choice("update_frequency", s, &[("batch", UpdateFrequency::Batch), ("epoch", UpdateFrequency::Epoch)])
    }
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
    pub init: EmbeddingInit,

    /// Whether updates are applied after every batch or once per epoch
    pub update_frequency: UpdateFrequency,

    /// Learning rate of the model's own dense weights (attention's projections), on the same
    /// schedule as `alpha`.  Dense matrices every node uses need far smaller steps than sparse
    /// feature embeddings.
    pub dense_alpha: f32,

    /// The space node embeddings live in: pooled vectors are mapped into it before the loss
    /// (see [`crate::geometry`]).  Node embeddings are then built with this geometry too:
    /// [`FeatureAggregator::with_geometry`](crate::algos::aggregator::FeatureAggregator::with_geometry).
    pub geometry: Geometry
}

impl EmbeddingPropagation {
    /// Checks the settings can train: positive dims, batch size and learning rate, a
    /// non-negative dense learning rate and noise, all finite, and a validation share in [0, 1).
    /// Learning checks this first.
    pub fn validate(&self) -> Result<(), Error> {
        let bad = |what: &str, v: f32| Error::InvalidArgument(format!("{} must be finite and {}, got {}", what,
            if what == "alpha" { "positive" } else { "non-negative" }, v));
        if !(self.alpha.is_finite() && self.alpha > 0.) { return Err(bad("alpha", self.alpha)) }
        if !(self.dense_alpha.is_finite() && self.dense_alpha >= 0.) { return Err(bad("dense_alpha", self.dense_alpha)) }
        if !(self.noise.is_finite() && self.noise >= 0.) { return Err(bad("noise", self.noise)) }
        if self.d_model == 0 || self.batch_size == 0 {
            return Err(Error::InvalidArgument(format!(
                "dims and batch_size must be positive, got {} and {}", self.d_model, self.batch_size)))
        }
        if !(0. ..1.).contains(&self.valid_pct) {
            return Err(Error::InvalidArgument(format!("valid_pct must be in [0, 1), got {}", self.valid_pct)))
        }
        if !self.loss.supports(&self.geometry) {
            return Err(Error::InvalidArgument(format!(
                "{:?} scores by dot product or cosine, so it needs geometry \"euclidean\", not {}",
                self.loss, self.geometry)))
        }
        Ok(())
    }

    /// Learns feature embeddings over a named graph and returns them named by feature.  Nodes
    /// without features get one of their own; `warm_start` continues from earlier feature
    /// embeddings, which must be indexed by the features.
    pub fn learn_features<M: Model>(
        &self,
        graph: &NamedGraph,
        features: &mut FeatureSet,
        warm_start: Option<NodeEmbeddings>,
        model: &M
    ) -> Result<NodeEmbeddings, Error> {
        features.prepare_for_learning(graph, warm_start.as_ref())?;
        let embeddings = self.learn(graph.graph(), &features.features,
                                    warm_start.map(|w| w.embeddings), model)?;
        Ok(NodeEmbeddings { vocab: Arc::new(features.features.clone_vocab()), embeddings })
    }
}

/// The defaults the Python bindings use.
impl Default for EmbeddingPropagation {
    fn default() -> Self {
        EmbeddingPropagation {
            alpha: 0.9,
            loss: Loss::MarginLoss(1f32, 1),
            batch_size: 50,
            d_model: 100,
            passes: 100,
            hard_negs: 0,
            loss_weighting: LossWeighting::None,
            seed: crate::DEFAULT_SEED,
            valid_pct: 0.1,
            noise: 0.0,
            indicator: true,
            optimizer: OptimizerKind::Adam,
            lr_schedule: LearningRateSchedule::CosineWarmup,
            init: EmbeddingInit::UnitNorm,
            update_frequency: UpdateFrequency::Batch,
            dense_alpha: 1e-3,
            geometry: Geometry::Euclidean
        }
    }
}

impl EmbeddingPropagation {

    /// Learns the feature embeddings, and a model's own weights (attention's) in the model.
    pub fn learn<G: CGraph + Send + Sync, M: Model>(
        &self, 
        graph: &G, 
        features: &FeatureStore,
        feature_embeddings: Option<EmbeddingStore>,
        model: &M
    ) -> Result<EmbeddingStore, Error> {
        self.learn_feature_embeddings(graph, features, feature_embeddings, model)
    }
    
    // The uber expensive function
    fn learn_feature_embeddings<G: CGraph + Send + Sync, M: Model>(
        &self,
        graph: &G,
        features: &FeatureStore,
        feature_embeddings: Option<EmbeddingStore>,
        model: &M
    ) -> Result<EmbeddingStore, Error> {
        self.validate()?;

        let mut rng = XorShiftRng::seed_from_u64(self.seed);

        let dims = self.d_model;
        let feature_embeddings = if let Some(embs) = feature_embeddings {
            embs
        } else {
            let mut fe = EmbeddingStore::new(features.num_features(), dims, Distance::Cosine);
            // Initialize embeddings as random
            initialize_embedding_store(&mut fe, self.init, &mut rng);
            fe
        };
        model.prepare(feature_embeddings.dims(), &mut rng)?;

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

        // Epoch updates sum every batch's gradients here and apply them at the end of the pass.
        let epoch_grads = match self.update_frequency {
            UpdateFrequency::Batch => None,
            UpdateFrequency::Epoch => Some(GradientAccumulator::new(
                feature_embeddings.len(), feature_embeddings.dims()))
        };

        // And the model's dense gradients, when it has weights of its own.
        let epoch_dense: std::sync::Mutex<Option<Vec<f32>>> = std::sync::Mutex::new(None);

        let updates_per_pass = if epoch_grads.is_some() { 1 } else { steps_per_pass };
        let total_updates = updates_per_pass * self.passes;
        let lr_scheduler = match self.lr_schedule {
            LearningRateSchedule::CosineWarmup => {
                let warm_up_steps = (total_updates as f32 / 5f32) as usize;
                LRScheduler::cos_decay(self.alpha / 100f32, self.alpha, warm_up_steps, total_updates)
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
                    let params = model.params();
                    let (mut loss, hv_vars, thv_vars, hu_vars) = self.run_forward_pass(
                        graph, n_id, &features, &feature_embeddings, 
                        model, &params, &sampler, &mut rng);

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
                        let grads = self.extract_gradients(&loss, hv_vars, thv_vars, hu_vars, model, &params);
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
                let mut dense: Option<Vec<f32>> = None;
                let error = grads.into_iter().map(|(err, (grad_set, dense_grad))| {
                    if let Some(g) = dense_grad {
                        match dense.as_mut() {
                            Some(sum) => sum.iter_mut().zip(g.iter()).for_each(|(s, gi)| *s += gi),
                            None => dense = Some(g)
                        }
                    }
                    for (feat, grad) in grad_set.into_iter() {
                        let e = all_grads.entry(feat).or_insert_with(|| vec![0.; grad.len()]);
                        e.iter_mut().zip(grad.iter()).for_each(|(ei, gi)| *ei += *gi);
                    }
                    err
                }).sum::<f32>();

                if let Some(acc) = &epoch_grads {
                    // Applied once the whole pass is done.
                    all_grads.par_iter().for_each(|(feat, grad)| acc.add(*feat, grad));
                    if let Some(g) = dense {
                        let mut sum = epoch_dense.lock().expect("epoch dense gradients lock");
                        match sum.as_mut() {
                            Some(s) => s.iter_mut().zip(g.iter()).for_each(|(si, gi)| *si += gi),
                            None => *sum = Some(g)
                        }
                    }
                } else if cnt > 0 {
                    let cur_step = step.fetch_add(1, Ordering::Relaxed);
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
                    if let Some(g) = dense {
                        model.update_dense(&g, self.dense_alpha * lr_scheduler.multiplier(cur_step));
                    }
                } else {
                    step.fetch_add(1, Ordering::Relaxed);
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

            if let Some(acc) = &epoch_grads {
                let cur_step = step.fetch_add(1, Ordering::Relaxed);
                let alpha = lr_scheduler.compute(cur_step);
                let noise = noise_scheduler.compute(cur_step);
                (0..acc.rows()).into_par_iter().for_each_init(|| vec![0f32; acc.dims()], |grad, feat| {
                    // Features no node in the pass touched aren't updated, as with batch updates.
                    if acc.take_row(feat, grad) {
                        if self.noise > 0.0 {
                            let mut rng = XorShiftRng::seed_from_u64(node_seed(self.seed, pass, 0, feat));
                            grad.iter_mut().for_each(|gi| {
                                *gi += noise * rng.sample::<f32,StandardNormal>(StandardNormal);
                            });
                        }
                        optimizer.update_row(&feature_embeddings, feat, grad, alpha, cur_step as f32);
                    }
                });
                if let Some(g) = epoch_dense.lock().expect("epoch dense gradients lock").take() {
                    model.update_dense(&g, self.dense_alpha * lr_scheduler.multiplier(cur_step));
                }
            }

            // Once-per-pass regularisation (AdamW's weight decay), at the schedule's position
            // after the pass's last update.
            let last_step = step.load(Ordering::Relaxed).saturating_sub(1).max(1);
            optimizer.end_pass(&feature_embeddings, lr_scheduler.multiplier(last_step));
            
            if valid_idxs.len() > 0 {
                // Validate.  We use the same random seed for consistency across iterations.
                let valid_errors = valid_idxs.par_iter().chunks(self.batch_size).map(|nodes| {
                    let sampler = (&valid_random_sampler).initialize_batch(&nodes, graph, features);

                    nodes.par_iter().map(|node_id| {
                        let mut rng = XorShiftRng::seed_from_u64(self.seed.wrapping_sub(1));
                        let loss = self.run_forward_pass(
                            graph, **node_id, &features, &feature_embeddings, 
                            model, &model.params(), &sampler, &mut rng).0;

                        loss.value()[0]
                    }).sum::<f32>()
                }).sum::<f32>();
                
                valid_error = valid_errors / valid_idxs.len() as f32;
            }
        }
        pb.finish();
        Ok(feature_embeddings)
    }

    fn run_forward_pass<G: CGraph + Send + Sync, R: Rng, S: NodeSampler, M: Model>(
        &self, 
        graph: &G,
        node: NodeID,
        features: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        model: &M,
        params: &M::Params,
        sampler: &S,
        rng: &mut R
    ) -> (ANode, NodeCounts, NodeCounts, Vec<NodeCounts>) {
        // h(v)
        let (hv_vars, hv) = model.construct_node_embedding(
            params, node, 1f32, features, &feature_embeddings, rng);
        
        // ~h(v)
        let (thv_vars, thv) = self.loss.construct_positive(
            params, graph, node, features, &feature_embeddings, model, rng);
        
        // h(u)
        let num_negs = self.loss.negatives();
        let mut negatives = Vec::with_capacity(num_negs);
        
        // Sample random negatives
        sampler.sample_negatives(graph, node, &mut negatives, num_negs, rng);
        
        let mut hu_vars = Vec::with_capacity(negatives.len());
        let mut hus = Vec::with_capacity(negatives.len());
        negatives.into_iter().for_each(|neg_node| {
            let (hu_var, hu) = model.construct_node_embedding(params, neg_node, 1f32, features, &feature_embeddings, rng);
            hu_vars.push(hu_var);
            hus.push(hu);
        });

        // Into the embedding space (the identity in Euclidean space), then the error.
        let g = &self.geometry;
        let hus: Vec<ANode> = hus.into_iter().map(|hu| g.to_space_node(hu)).collect();
        let loss = self.loss.compute(g.to_space_node(thv), g.to_space_node(hv), &hus, g);

        (loss, hv_vars, thv_vars, hu_vars)

    }

    /// The feature gradients, summed per feature, and the model's dense gradients (None without
    /// dense weights, or when any is non-finite, which skips this node's contribution as
    /// `extract_grads` skips non-finite feature gradients).
    fn extract_gradients<M: Model>(
        &self, 
        loss: &ANode,
        hv_vars: NodeCounts,
        thv_vars: NodeCounts,
        hu_vars: Vec<NodeCounts>,
        model: &M,
        params: &M::Params
    ) -> (HashMap<usize, Vec<f32>>, Option<Vec<f32>>) {

        // Compute gradients
        let mut agraph = Graph::new();

        // This call is about 90% of the entire cost of the algorithm, and with attention the
        // graph is far deeper than with averaging.
        agraph.backward(&loss);

        let mut grads = HashMap::new();
        extract_grads(&agraph, &mut grads, hv_vars.into_iter());
        extract_grads(&agraph, &mut grads, thv_vars.into_iter());
        hu_vars.into_iter().for_each(|hu_var| {
            extract_grads(&agraph, &mut grads, hu_var.into_iter());
        });

        let dense = model.dense_gradients(params, &agraph)
            .filter(|g| g.iter().all(|gi| gi.is_finite()));
        (grads, dense)
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
            init: EmbeddingInit::UnitNorm,
            update_frequency: UpdateFrequency::Batch,
            dense_alpha: 1e-3,
            geometry: Geometry::Euclidean
        };

        let embeddings = ep.learn_feature_embeddings(&ccsr, &feature_store, None, &model).unwrap();
        for idx in 0..embeddings.len() {
            let e = embeddings.get_embedding(idx);
            println!("{:?} -> {:?}", idx, e);
        }
    }

    /// Two 20-node cliques joined by a single edge.
    fn two_cliques() -> CumCSR {
        let mut edges = Vec::new();
        for c in 0..2 {
            for a in 0..20 {
                for b in 0..20 {
                    if a != b { edges.push((c * 20 + a, c * 20 + b, 1f32)); }
                }
            }
        }
        edges.push((0, 20, 1.));
        edges.push((20, 0, 1.));
        CumCSR::convert(CSR::construct_from_edges(edges, false))
    }

    fn quick_ep(update_frequency: UpdateFrequency, alpha: f32, passes: usize) -> EmbeddingPropagation {
        EmbeddingPropagation {
            alpha, passes, batch_size: 8, d_model: 8, valid_pct: 0.0, indicator: false,
            lr_schedule: LearningRateSchedule::Constant, init: EmbeddingInit::GlorotUniform,
            update_frequency, ..EmbeddingPropagation::default()
        }
    }

    /// The initial and trained feature embeddings, one feature per node.
    fn train(update_frequency: UpdateFrequency, alpha: f32, passes: usize) -> (EmbeddingStore, EmbeddingStore) {
        let graph = two_cliques();
        let mut features = FeatureStore::new(graph.len());
        features.fill_missing_nodes();
        let model = super::model::AveragedFeatureModel::new(Sample::All, Sample::All, false, false);
        // Same seed, so zero passes gives the starting point.
        let init = quick_ep(update_frequency, alpha, 0).learn_feature_embeddings(&graph, &features, None, &model).unwrap();
        let trained = quick_ep(update_frequency, alpha, passes).learn_feature_embeddings(&graph, &features, None, &model).unwrap();
        (init, trained)
    }

    /// The largest change to any coordinate between two stores.
    fn max_change(a: &EmbeddingStore, b: &EmbeddingStore) -> f32 {
        (0..a.len()).flat_map(|i| a.get_embedding(i).iter().zip(b.get_embedding(i))
            .map(|(x, y)| (x - y).abs()).collect::<Vec<_>>()).fold(0f32, f32::max)
    }

    /// One pass of epoch updates is a single Adam step, which moves each coordinate by at most
    /// alpha; batch updates take one step per batch.
    #[test]
    fn test_epoch_updates_apply_once_per_pass() {
        let alpha = 0.01;
        let (init, epoch) = train(UpdateFrequency::Epoch, alpha, 1);
        let moved = max_change(&init, &epoch);
        assert!(moved <= alpha * 1.0001 && moved > alpha * 0.5, "moved {} for alpha {}", moved, alpha);

        let (init, batch) = train(UpdateFrequency::Batch, alpha, 1);
        let moved = max_change(&init, &batch);
        assert!(moved > alpha * 1.5, "batch updates moved only {}", moved);
    }

    /// Epoch updates still learn: each clique ends up closer to itself than to the other.
    #[test]
    fn test_epoch_updates_learn() {
        let (_, embs) = train(UpdateFrequency::Epoch, 0.02, 100);
        let dist = |a: usize, b: usize| Distance::Cosine.compute(embs.get_embedding(a), embs.get_embedding(b));
        let within = (2..20).map(|i| dist(1, i)).sum::<f32>() / 18.;
        let across = (21..40).map(|i| dist(1, i)).sum::<f32>() / 19.;
        // Untrained, both are near 1.
        assert!(across - within > 1.0, "within-clique distance {}, across {}", within, across);
    }

    /// EP in the Poincaré ball learns too: each clique ends up closer to itself, by hyperbolic
    /// distance, than to the other, with either distance loss.
    #[test]
    fn test_poincare_ep_separates_the_cliques() {
        use crate::geometry::{Curvature, PoincareBall};
        let graph = two_cliques();
        let mut features = FeatureStore::new(graph.len());
        features.fill_missing_nodes();
        let model = super::model::AveragedFeatureModel::new(Sample::All, Sample::All, false, false);
        let ball = PoincareBall::new(Curvature::new(0.5).unwrap(), None).unwrap();
        for loss in [Loss::MarginLoss(1., 3), Loss::StarSpace(1., 3)] {
            let ep = EmbeddingPropagation { loss, geometry: Geometry::Poincare(ball),
                                            ..quick_ep(UpdateFrequency::Batch, 0.05, 30) };
            let embs = ep.learn_feature_embeddings(&graph, &features, None, &model).unwrap();
            let point = |i: usize| { let mut v = embs.get_embedding(i).to_vec(); ep.geometry.to_space(&mut v); v };
            let dist = |a: usize, b: usize| ball.curvature().distance(&point(a), &point(b));
            let within = (2..20).map(|i| dist(1, i)).sum::<f32>() / 18.;
            let across = (21..40).map(|i| dist(1, i)).sum::<f32>() / 19.;
            // The GPU trainer's clique test's criterion: StarSpace separates id-only cliques only
            // weakly in the ball (all distances grow together), the margin loss clearly.
            assert!(across > within, "{:?}: within-clique distance {}, across {}", loss, within, across);
        }
    }

    /// AdamW's weight decay is once per pass, so every feature decays by the same factor with
    /// batch and epoch updates alike, however many updates touched it.  A tiny learning rate
    /// makes the gradient steps negligible next to the decay.
    #[test]
    fn test_adamw_decay_is_per_pass_in_both_modes() {
        use crate::algos::grad_utils::optimizer::WeightDecay;
        let graph = two_cliques();
        let mut features = FeatureStore::new(graph.len());
        features.fill_missing_nodes();
        let model = super::model::AveragedFeatureModel::new(Sample::All, Sample::All, false, false);
        let (rate, passes) = (1f32, 3);
        let expected = (-rate * passes as f32).exp();
        for mode in [UpdateFrequency::Batch, UpdateFrequency::Epoch] {
            let ep = |passes| EmbeddingPropagation {
                optimizer: OptimizerKind::AdamW(WeightDecay::new(rate).unwrap()),
                ..quick_ep(mode, 1e-6, passes)
            };
            let init = ep(0).learn_feature_embeddings(&graph, &features, None, &model).unwrap();
            let trained = ep(passes).learn_feature_embeddings(&graph, &features, None, &model).unwrap();
            for i in 0..init.len() {
                for (a, b) in init.get_embedding(i).iter().zip(trained.get_embedding(i)) {
                    assert!((a * expected - b).abs() < 1e-4,
                        "{:?}: feature {} went from {} to {}, expected {}", mode, i, a, b, a * expected);
                }
            }
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


    /// Two cliques, each node with its own feature and one it shares with its pair (2k, 2k+1),
    /// so attention has several tokens per node to pool.
    fn two_cliques_with_pairs() -> (CumCSR, FeatureStore) {
        let graph = two_cliques();
        let mut features = FeatureStore::new(graph.len());
        for node in 0..graph.len() {
            features.set_features(node, [("n", node.to_string()), ("p", (node / 2).to_string())].into_iter());
        }
        (graph, features)
    }

    /// Five sampled neighbours per reconstruction keeps the debug-build test quick.
    fn attention_model() -> super::model::AttentionFeatureModel {
        super::model::AttentionFeatureModel::new(2, Sample::All, Sample::Fixed(5), false, false).unwrap()
    }

    fn attention_ep(update_frequency: UpdateFrequency, passes: usize) -> EmbeddingPropagation {
        EmbeddingPropagation { dense_alpha: 0.01, loss: Loss::StarSpace(0.5, 3), ..quick_ep(update_frequency, 0.05, passes) }
    }

    /// Node embeddings from the trained feature embeddings and the model's aggregator.
    fn embed_nodes(features: &FeatureStore, fe: &EmbeddingStore, model: &super::model::AttentionFeatureModel) -> Vec<Vec<f32>> {
        let weights = model.weights().unwrap();
        (0..features.num_nodes()).map(|node| {
            let x: Vec<&[f32]> = features.get_features(node).iter().map(|f| fe.get_embedding(*f)).collect();
            let mut out = vec![0f32; fe.dims()];
            weights.pool(&x, &vec![1f32; x.len()], &mut out);
            out
        }).collect()
    }

    /// CPU attention EP learns: each clique ends up closer to itself than to the other, and the
    /// projections train, the output projection leaving zero.
    #[test]
    fn test_attention_learns_the_cliques() {
        let (graph, features) = two_cliques_with_pairs();
        let model = attention_model();
        let fe = attention_ep(UpdateFrequency::Batch, 30).learn_feature_embeddings(&graph, &features, None, &model).unwrap();
        let weights = model.weights().unwrap();
        assert_eq!((weights.dims(), weights.heads(), weights.head_dims()), (8, 2, 8));
        assert!(weights.out().iter().any(|v| v.abs() > 1e-3), "the output projection didn't train");
        let embs = embed_nodes(&features, &fe, &model);
        let dist = |a: usize, b: usize| Distance::Cosine.compute(&embs[a], &embs[b]);
        let within = (2..20).map(|i| dist(1, i)).sum::<f32>() / 18.;
        let across = (21..40).map(|i| dist(1, i)).sum::<f32>() / 19.;
        assert!(across - within > 0.35, "within-clique distance {}, across {}", within, across);
    }

    /// One pass of epoch updates is one dense Adam step, so no attention weight moves by more
    /// than dense_alpha; batch updates take a step per batch.
    #[test]
    fn test_attention_epoch_updates_take_one_dense_step_per_pass() {
        let (graph, features) = two_cliques_with_pairs();
        let moved = |mode| {
            let start = attention_model();
            attention_ep(mode, 0).learn_feature_embeddings(&graph, &features, None, &start).unwrap();
            let model = attention_model();
            attention_ep(mode, 1).learn_feature_embeddings(&graph, &features, None, &model).unwrap();
            start.weights().unwrap().parameters().iter().zip(model.weights().unwrap().parameters())
                .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max)
        };
        let epoch = moved(UpdateFrequency::Epoch);
        assert!(epoch > 0.005 && epoch <= 0.01 * 1.0001, "epoch mode moved {}", epoch);
        assert!(moved(UpdateFrequency::Batch) > 0.015);
    }

    /// Warm-start weights must fit the embeddings being trained.
    #[test]
    fn test_attention_warm_start_must_match_dims() {
        let (graph, features) = two_cliques_with_pairs();
        let model = attention_model();
        let mut rng = XorShiftRng::seed_from_u64(0);
        model.warm_start(super::attention::AttentionWeights::init(6, 2, &mut rng).unwrap()).unwrap();
        assert!(attention_ep(UpdateFrequency::Batch, 1).learn_feature_embeddings(&graph, &features, None, &model).is_err());
        assert!(model.warm_start(super::attention::AttentionWeights::init(8, 3, &mut rng).unwrap()).is_err());
        let fitting = super::attention::AttentionWeights::init(8, 2, &mut rng).unwrap();
        model.warm_start(fitting.clone()).unwrap();
        attention_ep(UpdateFrequency::Batch, 0).learn_feature_embeddings(&graph, &features, None, &model).unwrap();
        assert_eq!(*model.weights().unwrap(), fitting);
    }


    #[test]
    fn test_settings_parse_from_strings() {
        assert_eq!("constant".parse::<LearningRateSchedule>().unwrap(), LearningRateSchedule::Constant);
        assert_eq!("glorot".parse::<EmbeddingInit>().unwrap(), EmbeddingInit::GlorotUniform);
        assert_eq!("epoch".parse::<UpdateFrequency>().unwrap(), UpdateFrequency::Epoch);
        for bad in ["", "Cosine", "linear"] {
            assert!(matches!(bad.parse::<LearningRateSchedule>(), Err(Error::InvalidArgument(_))), "{}", bad);
        }
        assert!("step".parse::<UpdateFrequency>().is_err() && "xavier".parse::<EmbeddingInit>().is_err());
    }

    /// Settings that can't train are rejected before training starts.
    #[test]
    fn test_validate_rejects_untrainable_settings() {
        assert!(EmbeddingPropagation::default().validate().is_ok());
        let bad = [
            EmbeddingPropagation { alpha: 0., ..EmbeddingPropagation::default() },
            EmbeddingPropagation { dense_alpha: f32::NAN, ..EmbeddingPropagation::default() },
            EmbeddingPropagation { batch_size: 0, ..EmbeddingPropagation::default() },
            EmbeddingPropagation { d_model: 0, ..EmbeddingPropagation::default() },
            EmbeddingPropagation { valid_pct: 1., ..EmbeddingPropagation::default() },
            EmbeddingPropagation { noise: -1., ..EmbeddingPropagation::default() },
            EmbeddingPropagation { loss: Loss::RankLoss(0.5, 3),
                geometry: Geometry::new("poincare", Some(1.), None).unwrap(), ..EmbeddingPropagation::default() },
        ];
        let graph = two_cliques();
        let mut features = FeatureStore::new(graph.len());
        features.fill_missing_nodes();
        let model = super::model::AveragedFeatureModel::new(Sample::All, Sample::All, false, false);
        for ep in bad.iter() {
            assert!(ep.validate().is_err());
            assert!(ep.learn(&graph, &features, None, &model).is_err());
        }
    }

    /// FeatureModel picks the model, warm-starts attention only, and gives each model's aggregator.
    #[test]
    fn test_feature_model() {
        use super::model::FeatureModel;
        use crate::algos::aggregator::FeatureAggregator;
        let averaged = FeatureModel::new(None, Sample::All, Sample::All, false, false).unwrap();
        assert_eq!(averaged.aggregator().unwrap().to_string(), "Averaged");
        let mut rng = XorShiftRng::seed_from_u64(1);
        let weights = FeatureAggregator::attention(super::attention::AttentionWeights::init(8, 2, &mut rng).unwrap());
        assert!(averaged.warm_start_from(&weights, 8).is_err());
        let attention = FeatureModel::new(Some(2), Sample::All, Sample::All, false, false).unwrap();
        assert!(attention.aggregator().is_none(), "no weights before training or a warm start");
        assert!(attention.warm_start_from(&FeatureAggregator::averaged(), 8).is_err());
        assert!(attention.warm_start_from(&weights, 6).is_err());
        attention.warm_start_from(&weights, 8).unwrap();
        assert_eq!(attention.aggregator().unwrap().attention_weights(), weights.attention_weights());
        assert!(FeatureModel::new(Some(0), Sample::All, Sample::All, false, false).is_err());
    }

}
