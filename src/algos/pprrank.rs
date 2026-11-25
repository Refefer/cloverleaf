//! 
use std::fmt::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use float_ord::FloatOrd;
use rayon::prelude::*;
use hashbrown::HashMap;
use std::collections::{HashMap as CHashMap};
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use crate::autograd::*;

use crate::algos::rwr::RWR;
use crate::algos::utils::{Sample,node_seed};
use crate::sampler::Weighted;
use crate::graph::{Graph as CGraph, CDFGraph, NodeID};
use crate::embeddings::{EmbeddingStore,randomize_embedding_store};
use crate::distance::Distance;
use crate::progress::CLProgressBar;
use crate::feature_store::{FeatureStore, FeatureSet};
use crate::embeddings::NodeEmbeddings;
use crate::error::Error;
use crate::named_graph::NamedGraph;
use crate::algos::ep::model::{construct_node_embedding,NodeCounts};
use crate::algos::ep::extract_grads;
use crate::algos::grad_utils::node_sampler::{RandomWalkHardStrategy,NodeSampler,BatchSamplerStrategy};
use crate::algos::grad_utils::optimizer::{Optimizer,AdamOptimizer,OptimizerKind};
use crate::algos::grad_utils::scheduler::LRScheduler;

#[derive(Copy,Clone,Debug,PartialEq)]
pub enum Loss {
    ListNet { passive:bool, weight_decay: f32 },
    ListMLE { weight_decay: f32 },
    /// ApproxNDCG (Qin, Liu and Li, 2010): one minus NDCG over the ranked list, with each node's
    /// rank replaced by a smooth estimate, `1 + sum_j sigmoid((s_j - s_i) / temperature)`, so
    /// it can be optimised directly.  The gains are the PPR weights.
    ApproxNdcg { temperature: f32, weight_decay: f32 }
}

impl Loss {
    /// ApproxNDCG's default temperature: smaller is closer to the true rank but has sharper
    /// gradients.
    pub const DEFAULT_APPROX_NDCG_TEMPERATURE: f32 = 0.1;

    /// "listnet" (in passive mode), "listmle" or "approxndcg" (at the default temperature),
    /// with L2 weight decay `weight_decay`.
    pub fn from_name(name: &str, weight_decay: f32) -> Result<Self, Error> {
        match name {
            "listnet" => Ok(Loss::ListNet { passive: true, weight_decay }),
            "listmle" => Ok(Loss::ListMLE { weight_decay }),
            "approxndcg" => Ok(Loss::ApproxNdcg { temperature: Self::DEFAULT_APPROX_NDCG_TEMPERATURE, weight_decay }),
            other => Err(Error::InvalidArgument(format!(
                "loss must be \"listnet\", \"listmle\" or \"approxndcg\", got {:?}", other)))
        }
    }

    /// ApproxNDCG at `temperature`, which must be finite and positive.
    pub fn approx_ndcg(temperature: f32, weight_decay: f32) -> Result<Self, Error> {
        if !(temperature.is_finite() && temperature > 0f32) {
            return Err(Error::InvalidArgument(format!(
                "the ApproxNDCG temperature must be finite and positive, got {}", temperature)))
        }
        Ok(Loss::ApproxNdcg { temperature, weight_decay })
    }
}

/// Defines PPR Rank
#[derive(Clone,Debug)]
pub struct PprRank {
    /// Optimization to use for computing loss
    pub loss: Loss,

    /// Learning rate for updating feature embeddings
    pub alpha: f32,

    /// Batch size to use.  Larger batches have fewer updates, but also lower variance
    pub batch_size: usize,

    /// Size of the node embeddings.  Feature embeddings can be larger if they use attention
    pub dims: usize,

    /// Number of passes to optimize for
    pub passes: usize,

    /// Whether we use hard negatives or not.  We might strip this out since I've had difficulty
    /// using it to improve test loss
    pub negatives: usize,
    
    /// Whether we use hard negatives or not.  We might strip this out since I've had difficulty
    /// using it to improve test loss
    pub num_walks: usize,

    /// Whether we use hard negatives or not.  We might strip this out since I've had difficulty
    /// using it to improve test loss
    pub steps: Sample,

    /// Beta to use for the RWR algorithm
    pub beta: f32,

    /// Number of items to select from the RWR algorithm for optimization
    pub k: usize,

    /// Randomly selects K features
    pub num_features: Sample,

    /// Compresses or flattens distribution
    pub compression: f32,

    /// Random seed
    pub seed: u64,

    /// We split out valid_pct of nodes to use for validation.
    pub valid_pct: f32,

    /// Whether to show a pretty indicator
    pub indicator: bool,

    /// Optimizer for the feature embeddings
    pub optimizer: OptimizerKind
}

impl PprRank {
    /// A learner with the defaults the Python bindings use: passive ListNet without weight
    /// decay, compression 1, beta 0.8, 10% of nodes for validation, every feature, Adam, the
    /// default seed and a progress bar.
    pub fn new(
        alpha: f32,
        batch_size: usize,
        dims: usize,
        passes: usize,
        steps: Sample,
        num_walks: usize,
        k: usize,
        negatives: usize
    ) -> Self {
        PprRank {
            loss: Loss::ListNet { passive: true, weight_decay: 0. },
            alpha, batch_size, dims, passes, negatives, num_walks, steps,
            beta: 0.8, k, num_features: Sample::All, compression: 1., seed: crate::DEFAULT_SEED,
            valid_pct: 0.1, indicator: true, optimizer: OptimizerKind::Adam
        }
    }

    /// Learns the feature embeddings.
    /// Learns feature embeddings over a named graph and returns them named by feature.  Nodes
    /// without features get one of their own; `warm_start` continues from earlier feature
    /// embeddings, which must be indexed by the features.
    pub fn learn_features(
        &self,
        graph: &NamedGraph,
        features: &mut FeatureSet,
        warm_start: Option<NodeEmbeddings>
    ) -> Result<NodeEmbeddings, Error> {
        features.prepare_for_learning(graph, warm_start.as_ref())?;
        let embeddings = self.learn(graph.graph(), &features.features, warm_start.map(|w| w.embeddings));
        Ok(NodeEmbeddings { vocab: std::sync::Arc::new(features.features.clone_vocab()), embeddings })
    }

    pub fn learn<G: CGraph + CDFGraph + Send + Sync>(
        &self, 
        graph: &G, 
        features: &FeatureStore,
        feature_embeddings: Option<EmbeddingStore>
    ) -> EmbeddingStore {
        let feat_embeds = self.learn_feature_embeddings(graph, features, feature_embeddings);
        feat_embeds
    }
    
    fn learn_feature_embeddings<G: CGraph + CDFGraph + Send + Sync>(
        &self,
        graph: &G,
        features: &FeatureStore,
        feature_embeddings: Option<EmbeddingStore>,
    ) -> EmbeddingStore {

        // 
        // Initialization
        //
        let mut rng = XorShiftRng::seed_from_u64(self.seed);

        let feature_embeddings = if let Some(embs) = feature_embeddings {
            embs
        } else {
            let mut fe = EmbeddingStore::new(features.num_features(), self.dims, Distance::Cosine);
            // Initialize embeddings as random
            randomize_embedding_store(&mut fe, &mut rng);
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

        // Enable/disable shared memory pool
        use_shared_pool(true);

        let total_updates = steps_per_pass * self.passes;
        let lr_scheduler = {
            let warm_up_steps = (total_updates as f32 / 5f32) as usize;
            let max_steps = self.passes * steps_per_pass;
            LRScheduler::cos_decay(self.alpha / 100f32, self.alpha, warm_up_steps, max_steps)
            //let min_alpha = self.alpha / 100f32; 
            //let gamma = (min_alpha.ln() / max_steps as f32).exp();
            //LRScheduler::exp_decay(min_alpha, self.alpha, gamma)
        };

        // Initialize samplers for negatives.
        let valid_random_sampler = RandomWalkHardStrategy::new(0, &valid_idxs);

        let mut last_error = std::f32::INFINITY;
        let step = AtomicUsize::new(1);
        let mut valid_error = std::f32::INFINITY;

        // Generate the top K for each node once
        let walk_lib = self.generate_random_walks(graph, self.seed+1,);

        //println!("{} -> {:?}", 0, walk_lib.get(0));
        //println!("");
        let pb = CLProgressBar::new((self.passes * steps_per_pass) as u64, self.indicator);
        
        for pass in 1..(self.passes + 1) {

            pb.update_message(|msg| {
                msg.clear();
                let cur_step = step.load(Ordering::Relaxed);
                let alpha = lr_scheduler.compute(cur_step);
                write!(msg, "Pass {}/{}, Train: {:.5}, Valid: {:.5}, LR: {:.5}", pass, self.passes, 
                       last_error, valid_error, alpha)
                    .expect("Error writing out indicator message!");
            });

            if pass % 10 == 0 && self.indicator {
                println!();
            }

            // Shuffle for SGD
            node_idxs.shuffle(&mut rng);
            // Borrows the shuffled node list for easy negatives, so it's rebuilt every pass.
            let random_sampler = RandomWalkHardStrategy::new(0, &node_idxs);
            let err: Vec<_> = node_idxs.par_iter().chunks(self.batch_size).enumerate().map(|(i, nodes)| {

                let mut grads = Vec::with_capacity(self.batch_size);
                
                let sampler = (&random_sampler).initialize_batch(
                    &nodes,
                    graph,
                    features);
                
                // Compute grads for batch
                nodes.par_iter().map(|node_id| {
                    let mut rng = XorShiftRng::seed_from_u64(node_seed(self.seed, pass, i, **node_id));
                    let (loss, feat_maps) = self.run_forward_pass(
                        graph, **node_id, &walk_lib, &features, &feature_embeddings, &sampler, &mut rng);

                    let grads = self.extract_gradients(&loss, feat_maps);
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

                // Backpropagate embeddings
                let alpha = lr_scheduler.compute(cur_step);
                optimizer.update(&feature_embeddings, all_grads, alpha, cur_step as f32);

                // Update progress bar
                pb.inc(1);
                if cnt > 0f32 { error / cnt } else { 0f32 }
            }).collect();

            // Once-per-pass regularisation (AdamW's weight decay).
            let last_step = step.load(Ordering::Relaxed).saturating_sub(1).max(1);
            optimizer.end_pass(&feature_embeddings, lr_scheduler.multiplier(last_step));

            // Some losses go toward infinity.  This is a bug we should fix.
            last_error = err.iter()
                .filter(|x| !x.is_infinite() )
                .sum::<f32>() / err.len() as f32;
            
            if valid_idxs.len() > 0 {
                // Validate.  We use the same random seed for consistency across iterations.
                let valid_errors = valid_idxs.par_iter().chunks(self.batch_size).map(|nodes| {
                    let sampler = (&valid_random_sampler).initialize_batch(&nodes, graph, features);

                    nodes.par_iter().map(|node_id| {
                        let mut rng = XorShiftRng::seed_from_u64(self.seed + **node_id as u64);
                        let loss = self.run_forward_pass(
                            graph, **node_id, &walk_lib, &features, &feature_embeddings, 
                            &sampler, &mut rng).0;

                        loss.value()[0]
                    })
                    .filter(|l| !l.is_infinite())
                    .sum::<f32>()
                }).sum::<f32>();
                
                valid_error = valid_errors / valid_idxs.len() as f32;
            }
        }
        pb.finish();
        feature_embeddings
    }

    fn construct_avg_node(
        &self,
        node_id: NodeID,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut impl Rng
    ) -> (NodeCounts, ANode) {
        construct_node_embedding(
            node_id, 
            1f32,
            feature_store,
            feature_embeddings,
            self.num_features,
            rng)
    }

    fn generate_random_walks<G: CGraph + CDFGraph + Send + Sync>(
        &self, 
        graph: &G,
        seed: u64
    ) -> WalkLibrary {
        ppr_lists(graph, self.steps.clone(), self.num_walks, self.beta, self.k, self.compression,
                  seed, |_node, _other| true, self.indicator)
    }

    fn run_forward_pass<G: CGraph + Send + Sync, R: Rng, S: NodeSampler>(
        &self, 
        graph: &G,
        node: NodeID,
        walk_lib: &WalkLibrary,
        features: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        sampler: &S,
        rng: &mut R
    ) -> (ANode, Vec<NodeCounts>) {
        let mut ranked_ids = Vec::with_capacity(self.k + self.negatives);
        let mut ranked_scores = Vec::with_capacity(self.k + self.negatives);
        let mut neg_ids = Vec::with_capacity(self.negatives);

        // Add the positives
        let (nodes, scores) = walk_lib.get(node);
        ranked_ids.extend_from_slice(nodes);
        ranked_scores.extend_from_slice(scores);

        // Sample random negatives
        sampler.sample_negatives(graph, node, &mut neg_ids, self.negatives, rng);
        neg_ids.into_iter().for_each(|neg_id| {
            ranked_ids.push(neg_id);
            ranked_scores.push(0.);
        });

        // Create the embeddings
        let mut ranked_embeddings = Vec::with_capacity(ranked_ids.len());
        let mut feat_maps = Vec::with_capacity(ranked_ids.len());
        ranked_ids.iter().for_each(|node_id| {
            let (fm, emb) = self.construct_avg_node(*node_id, features, feature_embeddings, rng);
            feat_maps.push(fm);
            ranked_embeddings.push(emb);
        });

        let (fm, query_node) = self.construct_avg_node(node, features, feature_embeddings, rng);
        feat_maps.push(fm);
        
        // Compute error
        let loss = match self.loss {
            Loss::ListNet { passive, weight_decay } => {
                let mut list_loss = listnet_loss(&query_node, &ranked_embeddings, &ranked_scores, passive);
                if weight_decay > 0f32 {
                    list_loss = list_loss + weight_decay * comp_weight_decay(&query_node, &ranked_embeddings, 0.1f32)
                }
                list_loss
            },
            Loss::ListMLE { weight_decay } => {
                let mut list_loss = list_mle_loss(&query_node, &ranked_embeddings, nodes.len());
                if weight_decay > 0f32 {
                    list_loss = list_loss + weight_decay * comp_weight_decay(&query_node, &ranked_embeddings, 0.1f32)
                }
                list_loss
            },
            Loss::ApproxNdcg { temperature, weight_decay } => {
                let logits = compute_logits(&query_node, &ranked_embeddings, false);
                let mut list_loss = approx_ndcg_loss(&logits, &ranked_scores, temperature);
                if weight_decay > 0f32 {
                    list_loss = list_loss + weight_decay * comp_weight_decay(&query_node, &ranked_embeddings, 0.1f32)
                }
                list_loss
            }
        };

        (loss, feat_maps)
    }

    fn extract_gradients(
        &self, 
        loss: &ANode,
        feat_maps: Vec<NodeCounts>
    ) -> HashMap<usize, Vec<f32>> {

        // Compute gradients
        let mut agraph = Graph::new();

        agraph.backward(&loss);

        let mut grads = HashMap::new();
        feat_maps.into_iter().for_each(|fm| {
            extract_grads(&agraph, &mut grads, fm.into_iter());
        });

        grads

    }

}

/// Each node's top-k personalised PageRank list, estimated with `walks` random walks with
/// restart (see [`RWR`]) that follow edges in proportion to their weight.  The node itself, and
/// any node `keep(node, other)` rejects, are left out; the rest are weighted score^compression
/// and normalised to sum to 1.
/// Settings for [`ppr_lists_named`].
#[derive(Clone,Copy,Debug)]
pub struct PprListOptions {
    /// List length.
    pub k: usize,
    /// Random walks per node.
    pub walks: usize,
    /// A restart probability (below 1) or a fixed walk length.
    pub steps: Sample,
    /// The popularity discount: scores / degree^beta.
    pub beta: f32,
    /// Weights are score^compression, normalised.
    pub compression: f32,
    /// Keep only nodes of a different type from the start node.
    pub other_type_only: bool,
    pub seed: u64,
    pub indicator: bool
}

impl PprListOptions {
    /// `k`-long lists from `walks` walks per node of `steps` steps, with the defaults: beta 0.8,
    /// compression 1, every node type, the default seed and a progress bar.
    pub fn new(k: usize, walks: usize, steps: Sample) -> Self {
        PprListOptions { k, walks, steps, beta: 0.8, compression: 1., other_type_only: false,
                         seed: crate::DEFAULT_SEED, indicator: true }
    }
}

/// [`ppr_lists`] over a named graph: every node's top-k PPR list, in node id order.
pub fn ppr_lists_named(graph: &NamedGraph, opts: &PprListOptions) -> Result<WalkLibrary, Error> {
    if opts.walks == 0 || opts.k == 0 {
        return Err(Error::InvalidArgument(format!(
            "walks and k must be positive, got {} and {}", opts.walks, opts.k)))
    }
    let vocab = graph.vocab();
    let keep = |node: NodeID, other: NodeID| {
        !opts.other_type_only || vocab.get_node_type(node) != vocab.get_node_type(other)
    };
    Ok(ppr_lists(graph.graph(), opts.steps, opts.walks, opts.beta, opts.k, opts.compression,
                 opts.seed, keep, opts.indicator))
}

pub fn ppr_lists<G: CGraph + CDFGraph + Send + Sync>(
    graph: &G,
    steps: Sample,
    walks: usize,
    beta: f32,
    k: usize,
    compression: f32,
    seed: u64,
    keep: impl Fn(NodeID, NodeID) -> bool + Sync,
    indicator: bool
) -> WalkLibrary {
    let mut walk_lib = WalkLibrary::new(graph.len(), k);
    let rwr = RWR { steps, walks, beta, single_threaded: false, seed: seed + 13 };

    let pb = CLProgressBar::new(graph.len() as u64, indicator);
    pb.update_message(|msg| write!(msg, "Computing random walks...").unwrap());
    let idxs = (0..graph.len()).collect::<Vec<_>>();
    idxs.chunks(1024).for_each(|node_ids| {
        let groups: Vec<_> = node_ids.par_iter().map(|node_id| {
            let mut nodes = Vec::with_capacity(k);
            let mut weights = Vec::with_capacity(k);

            let mut scores: Vec<_> = rwr.sample(graph, &Weighted, *node_id).into_iter().collect();
            // Ties broken by node id, so the lists don't depend on hash map order.
            scores.sort_by_key(|(n, v)| (FloatOrd(-*v), *n));
            scores.into_iter()
                .filter(|(n, _v)| n != node_id && keep(*node_id, *n))
                .map(|(n, v)| (n, v.powf(compression)))
                .take(k)
                .for_each(|(n, v)| {
                    nodes.push(n);
                    weights.push(v);
                });

            let sum = weights.iter().sum::<f32>();
            weights.iter_mut().for_each(|v| *v /= sum);
            pb.inc(1);
            (*node_id, nodes, weights)
        }).collect();

        groups.into_iter().for_each(|(node_id, nodes, weights)| {
            walk_lib.set(node_id, &nodes, &weights);
        });
    });
    pb.finish();
    walk_lib
}

/// ListNet: cross entropy between the PPR distribution over the ranked nodes (zero for the
/// negatives) and a softmax over the query's dot products with them.  With `passive`, positives
/// already predicted at least as likely as their PPR weight contribute nothing.
fn listnet_loss(
    query_node: &ANode,
    ranked_nodes: &[ANode],
    node_weights: &[f32],
    passive: bool
) -> ANode {
    let log_p = log_softmax(&compute_logits(query_node, ranked_nodes, false));
    let log_p_values = log_p.value();
    let terms = node_weights.iter().enumerate()
        .filter(|(i, s)| **s > 0f32 && !(passive && log_p_values[*i].exp() > **s))
        .map(|(i, s)| log_p.slice(i, 1) * *s)
        .collect::<Vec<ANode>>();

    if terms.is_empty() {
        Constant::scalar(0f32)
    } else {
        -terms.sum_all()
    }
}

/// Top-k ListMLE: the Plackett-Luce likelihood of the positives coming first, in PPR order, with
/// the negatives competing in every denominator.  The negatives' own order is arbitrary, so it
/// isn't scored.
fn list_mle_loss(query_node: &ANode, ranked_nodes: &[ANode], num_positives: usize) -> ANode {
    let logits = compute_logits(query_node, ranked_nodes, false);
    let loss = plackett_luce_nll(&logits, num_positives);
    if loss.value()[0].is_finite() { loss } else { Constant::scalar(0f32) }
}

/// ApproxNDCG: `1 - DCG / IDCG`, with DCG's ranks the smooth estimates
/// `pi_i = 1 + sum_{j != i} sigmoid((s_j - s_i) / temperature)` and gains `2^y - 1` from the PPR
/// weights `y` (zero for the negatives, which only push the positives' ranks down).  In [0, 1];
/// 0 for a list without positives.
fn approx_ndcg_loss(logits: &ANode, gains: &[f32], temperature: f32) -> ANode {
    let gain = |y: f32| 2f32.powf(y) - 1f32;
    let mut ideal: Vec<f32> = gains.iter().filter(|y| **y > 0f32).map(|y| gain(*y)).collect();
    ideal.sort_by(|a, b| b.total_cmp(a));
    let idcg: f32 = ideal.iter().enumerate().map(|(r, g)| g / (2f32 + r as f32).log2()).sum();
    if !(idcg > 0f32) {
        return Constant::scalar(0f32)
    }
    let terms: Vec<ANode> = gains.iter().enumerate().filter(|(_, y)| **y > 0f32).map(|(i, y)| {
        // sigmoid(x) = (1 + tanh(x / 2)) / 2, which can't overflow.  Summed over every j, the
        // j = i term adds sigmoid(0) = 1/2, so pi_i = 1/2 + the sum.
        let x = (logits - &logits.slice(i, 1)) * (0.5f32 / temperature);
        let rank = (x.tanh() + 1f32).sum() * 0.5f32 + 0.5f32;
        (gain(*y) * std::f32::consts::LN_2 / idcg) / (rank + 1f32).ln()
    }).collect();
    1f32 - terms.sum_all()
}

/// The query's scores for each ranked node: dot products, or cosine similarities times 5.  These
/// are logits; the losses apply the softmax.
fn compute_logits(query_node: &ANode, ranked_nodes: &[ANode], cosine: bool) -> ANode {
    if cosine {
        let qn = il2norm(query_node);
        ranked_nodes.iter().map(|n| qn.dot(&il2norm(n))).collect::<Vec<_>>().concat() * 5f32
    } else {
        ranked_nodes.iter().map(|n| query_node.dot(n)).collect::<Vec<_>>().concat()
    }
}

/// log softmax, shifted by the max so exp can't overflow.
fn log_softmax(logits: &ANode) -> ANode {
    let shift = Constant::scalar(
        logits.value().iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    let shifted = logits - &shift;
    let log_sum = shifted.exp().sum().ln();
    shifted - &log_sum
}

/// Negative log likelihood of the Plackett-Luce model for the first `top` items, already in
/// ranked order, each chosen from itself and everything after it.  Sums log probabilities rather
/// than multiplying probabilities, which underflows for long lists.
fn plackett_luce_nll(logits: &ANode, top: usize) -> ANode {
    let values = logits.value();
    let n = values.len();
    let top = top.min(n);
    if top == 0 {
        return Constant::scalar(0f32)
    }
    let terms: Vec<_> = (0..top).map(|i| {
        // Shift by the tail's max so the largest exp is 1: exp can't overflow and the tail sum is
        // at least 1, so its log can't be -inf.  The shift cancels out of the term.
        let shift = Constant::scalar(
            values[i..].iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        let log_tail_sum = (logits.slice(i, n - i) - &shift).exp().sum().ln();
        log_tail_sum - (logits.slice(i, 1) - &shift)
    }).collect();
    terms.sum_all()
}

fn il2norm(v: &ANode) -> ANode {
    v / l2norm(v)
}

fn l2norm(v: &ANode) -> ANode {
    v.pow(2f32).sum().pow(0.5)
}

fn comp_weight_decay(query_node: &ANode, ranked_nodes: &[ANode], threshold: f32) -> ANode {
    let t = Constant::scalar(threshold);
    let mut mag = Vec::with_capacity(ranked_nodes.len() + 1);


    let qn = query_node.pow(2f32).sum() - &t;
    if qn.value()[0] > 0f32 {
        mag.push(qn);
    }

    ranked_nodes.iter().for_each(|n| {
        let nn = n.pow(2f32).sum() - &t;
        if nn.value()[0] > 0f32 {
            mag.push(nn)
        }
    });
    mag.concat().sum()
}

/// Each node's top-k personalised PageRank list, as [`ppr_lists`] builds it.
pub struct WalkLibrary {
    k: usize,
    nodes: Vec<NodeID>,
    weights: Vec<f32>,
    lens: Vec<usize>
}

impl WalkLibrary {
    fn new(num_nodes: usize, k: usize) -> Self {
        let nodes = vec![0; num_nodes * k];
        let weights = vec![0f32; num_nodes * k];
        let lens = vec![0; num_nodes];
        WalkLibrary { k, nodes, weights, lens }
    }

    fn set(&mut self, node_id: NodeID, nodes: &[NodeID], weights: &[f32]) {
        let offset = node_id * self.k;
        self.lens[node_id] = weights.len().min(self.k); 
        nodes.iter().zip(weights.iter()).take(self.k).enumerate()
            .for_each(|(i, (ni, wi))| {
                let idx = offset + i;
                self.nodes[idx] = *ni;
                self.weights[idx] = *wi;
            });
    }

    /// The list length.
    pub fn k(&self) -> usize { self.k }

    /// Every node's list, flattened: node `v`'s is entries `v * k .. (v + 1) * k`, padded with
    /// node 0 at weight 0.
    pub fn into_parts(self) -> (Vec<NodeID>, Vec<f32>) { (self.nodes, self.weights) }

    pub fn get(&self, node_id: NodeID) -> (&[NodeID], &[f32]) {
        let offset: usize = node_id * self.k;
        let len = self.lens[node_id];
        let ns = &self.nodes[offset..(offset+len)];
        let ws = &self.weights[offset..(offset+len)];
        (ns, ws)
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
        let csr = CSR::construct_from_edges(edges, false);
        let ccsr = CumCSR::convert(csr);
        
        let mut feature_store = FeatureStore::new(ccsr.len());
        feature_store.fill_missing_nodes();

    }

    #[test]
    fn test_walk_library() {
        let mut walk_lib = WalkLibrary::new(10, 5);
        let (ns, ws) = walk_lib.get(3);
        assert_eq!(ns, vec![0;0]);
        assert_eq!(ws, vec![0.;0]);

        walk_lib.set(3, &[1, 2], &[1., 2.]);
        let (ns, ws) = walk_lib.get(3);
        assert_eq!(ns, &[1, 2]);
        assert_eq!(ws, &[1., 2.]);
    }


    fn plackett_luce_reference(logits: &[f64]) -> f64 {
        (0..logits.len()).map(|i| {
            let tail = &logits[i..];
            let m = tail.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lse = m + tail.iter().map(|l| (l - m).exp()).sum::<f64>().ln();
            lse - logits[i]
        }).sum()
    }

    /// A badly ordered list has a tiny likelihood; the loss must stay finite, not underflow.
    #[test]
    fn test_plackett_luce_nll_no_underflow() {
        let logits: Vec<f32> = (0..10).map(|i| i as f32 * 5.).collect();
        let reference = plackett_luce_reference(&logits.iter().map(|l| *l as f64).collect::<Vec<_>>());
        let loss = plackett_luce_nll(&Variable::new(logits), 10).value()[0];
        assert!(((loss as f64 - reference) / reference).abs() < 1e-4,
            "loss {} reference {}", loss, reference);
    }

    /// Large logits must not overflow exp.
    #[test]
    fn test_plackett_luce_nll_no_overflow() {
        let logits = vec![100f32, 0.];
        let reference = plackett_luce_reference(&[100., 0.]);
        let loss = plackett_luce_nll(&Variable::new(logits), 2).value()[0];
        assert!((loss as f64 - reference).abs() < 1e-5, "loss {} reference {}", loss, reference);
    }

    fn log_softmax_reference(logits: &[f64]) -> Vec<f64> {
        let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lse = m + logits.iter().map(|l| (l - m).exp()).sum::<f64>().ln();
        logits.iter().map(|l| l - lse).collect()
    }

    fn query_and_ranked() -> (ANode, Vec<ANode>, Vec<f64>) {
        let query = Variable::new(vec![1f32, -2., 0.5]);
        let ranked: Vec<Vec<f32>> = vec![vec![0.5, 1., 2.], vec![3., -1., 0.], vec![-1., -1., 1.],
                                          vec![0., 2., -4.]];
        let logits = ranked.iter()
            .map(|r| r.iter().zip([1f64, -2., 0.5]).map(|(a, b)| *a as f64 * b).sum())
            .collect();
        (query, ranked.into_iter().map(Variable::new).collect(), logits)
    }

    /// ListNet is the cross entropy against one softmax over the dot products, not exp(exp(.)).
    #[test]
    fn test_listnet_is_cross_entropy_over_dot_products() {
        let (query, ranked, logits) = query_and_ranked();
        let weights = [0.7f32, 0.3, 0., 0.];
        let log_p = log_softmax_reference(&logits);
        let reference = -(0.7 * log_p[0] + 0.3 * log_p[1]);
        let loss = listnet_loss(&query, &ranked, &weights, false).value()[0] as f64;
        assert!((loss - reference).abs() < 1e-5, "loss {} reference {}", loss, reference);
    }

    /// Passive ListNet skips positives already predicted at least as likely as their weight.
    #[test]
    fn test_listnet_passive_skips_satisfied_positives() {
        let (query, ranked, logits) = query_and_ranked();
        let log_p = log_softmax_reference(&logits);
        // logits are [-0.5, 5, 1.5, -6]: node 1 takes nearly all the mass, node 0 almost none.
        let weights = [0.5f32, 0.5, 0., 0.];
        let reference = -(0.5 * log_p[0]);
        let loss = listnet_loss(&query, &ranked, &weights, true).value()[0] as f64;
        assert!((loss - reference).abs() < 1e-5, "loss {} reference {}", loss, reference);
    }

    /// Large dot products must not overflow.
    #[test]
    fn test_listnet_stable_for_large_logits() {
        let query = Variable::new(vec![100f32]);
        let ranked = vec![Variable::new(vec![1f32]), Variable::new(vec![-1f32])];
        let loss = listnet_loss(&query, &ranked, &[0f32, 1.], false).value()[0];
        assert!((loss - 200.).abs() < 1e-3, "loss {}", loss);
    }

    /// Top-k ListMLE scores only the positives' positions: reordering the negatives changes nothing.
    #[test]
    fn test_list_mle_ignores_negative_order() {
        let (query, ranked, logits) = query_and_ranked();
        let loss = list_mle_loss(&query, &ranked, 2).value()[0] as f64;
        let reference = plackett_luce_reference(&logits) - plackett_luce_reference(&logits[2..]);
        assert!((loss - reference).abs() < 1e-5, "loss {} reference {}", loss, reference);

        let swapped = vec![ranked[0].clone(), ranked[1].clone(), ranked[3].clone(), ranked[2].clone()];
        let swapped_loss = list_mle_loss(&query, &swapped, 2).value()[0] as f64;
        assert!((loss - swapped_loss).abs() < 1e-6);
    }

    #[test]
    fn test_loss_names_are_strict() {
        assert_eq!(Loss::from_name("listnet", 0.1).unwrap(), Loss::ListNet { passive: true, weight_decay: 0.1 });
        assert_eq!(Loss::from_name("listmle", 0.).unwrap(), Loss::ListMLE { weight_decay: 0. });
        assert_eq!(Loss::from_name("approxndcg", 0.).unwrap(),
                   Loss::ApproxNdcg { temperature: Loss::DEFAULT_APPROX_NDCG_TEMPERATURE, weight_decay: 0. });
        for bad in ["ListNet", "list_mle", "approx_ndcg", ""] {
            assert!(Loss::from_name(bad, 0.).is_err(), "{}", bad);
        }
        for bad in [0f32, -1., f32::NAN, f32::INFINITY] {
            assert!(Loss::approx_ndcg(bad, 0.).is_err(), "{}", bad);
        }
        assert_eq!(Loss::approx_ndcg(0.5, 0.).unwrap(), Loss::ApproxNdcg { temperature: 0.5, weight_decay: 0. });
    }

    /// The exact NDCG of ranking by `scores`, for checking ApproxNDCG's low-temperature limit.
    fn exact_ndcg(scores: &[f32], gains: &[f32]) -> f32 {
        let g = |y: f32| 2f32.powf(y) - 1f32;
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]));
        let dcg: f32 = order.iter().enumerate().map(|(r, i)| g(gains[*i]) / (2f32 + r as f32).log2()).sum();
        let mut ideal: Vec<f32> = gains.iter().map(|y| g(*y)).collect();
        ideal.sort_by(|a, b| b.total_cmp(a));
        let idcg: f32 = ideal.iter().enumerate().map(|(r, x)| x / (2f32 + r as f32).log2()).sum();
        dcg / idcg
    }

    /// At a low temperature ApproxNDCG is one minus the exact NDCG, whatever the order.
    #[test]
    fn test_approx_ndcg_matches_exact_ndcg_at_low_temperature() {
        let gains = [0.5f32, 0.3, 0.2, 0., 0., 0.];
        for scores in [[6f32, 5., 4., 3., 2., 1.], [1., 2., 3., 4., 5., 6.], [4., 6., 1., 5., 2., 3.]] {
            let loss = approx_ndcg_loss(&Variable::new(scores.to_vec()), &gains, 0.01).value()[0];
            let want = 1f32 - exact_ndcg(&scores, &gains);
            assert!((loss - want).abs() < 1e-4, "scores {:?}: {} vs {}", scores, loss, want);
        }
        // The ideal order scores 0; no positives, 0 too.
        let perfect = approx_ndcg_loss(&Variable::new(vec![6., 5., 4., 3., 2., 1.]), &gains, 0.01).value()[0];
        assert!(perfect.abs() < 1e-4);
        assert_eq!(approx_ndcg_loss(&Variable::new(vec![1., 2.]), &[0., 0.], 0.1).value()[0], 0.);
    }

    /// Its gradient raises the positives' scores and lowers the negatives', and stays finite for
    /// large score gaps.
    #[test]
    fn test_approx_ndcg_gradient() {
        let scores = Variable::new(vec![1., 3., 2., -500.]);
        let loss = approx_ndcg_loss(&scores, &[0.6, 0.4, 0., 0.], 0.5);
        let mut graph = Graph::new();
        graph.backward(&loss);
        let grad = graph.get_grad(&scores).unwrap();
        assert!(grad.iter().all(|g| g.is_finite()), "{:?}", grad);
        assert!(grad[0] < 0. && grad[2] > 0., "descent raises positive 0, lowers negative 2: {:?}", grad);
    }

    /// Named PPR lists check their sizes, and can keep only the other node type.
    #[test]
    fn test_ppr_lists_named() {
        use crate::named_graph::NamedGraph;
        use crate::vocab::Vocab;
        use crate::graph::{CSR, CumCSR};
        use std::sync::Arc;
        let mut vocab = Vocab::new();
        for (t, n) in [("q", "a"), ("p", "x"), ("q", "b"), ("p", "y")] { vocab.get_or_insert(t, n); }
        let edges = vec![(0, 1, 1.), (1, 0, 1.), (2, 1, 1.), (1, 2, 1.), (2, 3, 1.), (3, 2, 1.)];
        let graph = NamedGraph::new(Arc::new(vocab), Arc::new(CumCSR::convert(CSR::construct_from_edges(edges, false)))).unwrap();
        let steps = Sample::steps(0.5).unwrap();
        assert!(ppr_lists_named(&graph, &PprListOptions::new(0, 10, steps)).is_err());
        assert!(ppr_lists_named(&graph, &PprListOptions::new(2, 0, steps)).is_err());
        let opts = PprListOptions { other_type_only: true, indicator: false, ..PprListOptions::new(3, 200, steps) };
        let (nodes, weights) = ppr_lists_named(&graph, &opts).unwrap().into_parts();
        let types = [0, 1, 0, 1];
        for v in 0..4 {
            for (u, w) in nodes[v * 3..v * 3 + 3].iter().zip(&weights[v * 3..v * 3 + 3]) {
                if *w > 0. { assert_ne!(types[*u], types[v], "node {} kept {}", v, u); }
            }
        }
    }
}
