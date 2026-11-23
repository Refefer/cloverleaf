//! The Embedding Propagation framework parameterizes over the feature aggregator - that is, given
//! a node with a set of features, how do we combine them to product a node embedding?
//! This module defines them
use std::sync::{Arc, Mutex, RwLock};

use crate::autograd::*;
use hashbrown::HashMap;
use rand::prelude::*;

use crate::FeatureStore;
use crate::EmbeddingStore;
use crate::error::Error;
use crate::graph::{Graph as CGraph,NodeID, CDFtoP};
use crate::algos::aggregator::FeatureAggregator;
use crate::algos::grad_utils::optimizer::DenseAdam;
use crate::algos::utils::{Sample,weighted_reservoir_sample,reservoir_sample};
use super::attention::{attention_pool,AttentionVars,AttentionWeights};

/// Main interface for model.  Needs to be threadsafe
pub trait Model: Send + Sync {

    /// What a forward pass reads besides the feature embeddings: nothing for the averaged model,
    /// the attention weights as autograd variables for attention.  Made once per forward pass
    /// (`params`), so all of one node's embeddings use the same weights.
    type Params;

    /// Gets the model ready to train `dims`-wide feature embeddings: attention initialises its
    /// weights, or checks the ones it was warm-started with fit.
    fn prepare(&self, dims: usize, rng: &mut impl Rng) -> Result<(), Error>;

    fn params(&self) -> Self::Params;

    /// Given a node, construct a node embedding from its features.
    fn construct_node_embedding<R: Rng>(
        &self,
        params: &Self::Params,
        node: NodeID,
        weight: f32,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut R
    ) -> (NodeCounts, ANode);

    /// Given a node, reconstruct a node from its neighborhood
    fn reconstruct_node_embedding<G: CGraph, R: Rng>(
        &self,
        params: &Self::Params,
        graph: &G,
        node: NodeID,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut R
    ) -> (NodeCounts, ANode);

    /// Construct multiple node embeddings - we use this for negatives, typically.
    fn construct_from_multiple_nodes<I: Iterator<Item=(NodeID, f32)>, R: Rng>(
        &self,
        params: &Self::Params,
        nodes: I,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut R
    ) -> (NodeCounts, ANode);

    /// The gradients of the model's own weights after a backward pass through a forward pass
    /// that used `params`, flattened; None for a model without weights or a loss that didn't
    /// reach them.
    fn dense_gradients(&self, params: &Self::Params, graph: &Graph) -> Option<Vec<f32>>;

    /// Applies summed `dense_gradients` with learning rate `alpha`.
    fn update_dense(&self, gradients: &[f32], alpha: f32);
}

/// Creates node embeddings by averaging features together
pub struct AveragedFeatureModel {
    /// Randomly sample max_features if provided
    max_features: Sample,

    /// In the case of the node reconstruction, we use max_neighbor_nodes instead of the entire
    /// neighborhood.  This is critical when we have nodes with large neighbors.
    max_neighbor_nodes: Sample,

    /// If true, samples neighborhoods proportionally to their edge weights
    weighted_neighbor_sampling: bool,
       
    /// If true, during reconstruction, nodes are blended according to their edge weights.
    weighted_neighbor_averaging: bool
}

impl AveragedFeatureModel {
    pub fn new(
        max_features: Sample,
        max_neighbor_nodes: Sample,
        weighted_neighbor_sampling: bool,
        weighted_neighbor_averaging: bool
    ) -> Self {
        AveragedFeatureModel { 
            max_features, 
            max_neighbor_nodes, 
            weighted_neighbor_averaging,
            weighted_neighbor_sampling
        }
    }
}

impl Model for AveragedFeatureModel {
    type Params = ();

    fn prepare(&self, _dims: usize, _rng: &mut impl Rng) -> Result<(), Error> {
        Ok(())
    }

    fn params(&self) {}

    fn construct_node_embedding<R: Rng>(
        &self,
        _params: &(),
        node: NodeID,
        weight: f32,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut R
    ) -> (NodeCounts, ANode) {
        construct_node_embedding(
            node,
            weight,
            feature_store,
            feature_embeddings,
            self.max_features,
            rng)
    }

    fn reconstruct_node_embedding<G: CGraph, R: Rng>(
        &self,
        _params: &(),
        graph: &G,
        node: NodeID,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut R
    ) -> (NodeCounts, ANode){
        reconstruct_node_embedding(
            graph,
            node,
            feature_store,
            feature_embeddings,
            self.max_neighbor_nodes,
            self.max_features,
            None,
            self.weighted_neighbor_sampling,
            self.weighted_neighbor_averaging,
            rng)
    }

    fn construct_from_multiple_nodes<I: Iterator<Item=(NodeID, f32)>, R: Rng>(
        &self,
        _params: &(),
        nodes: I,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut R
    ) -> (NodeCounts, ANode) { 
        construct_from_multiple_nodes(
            nodes, feature_store, 
            feature_embeddings, 
            self.max_features,
            None, rng)
    }

    fn dense_gradients(&self, _params: &(), _graph: &Graph) -> Option<Vec<f32>> {
        None
    }

    fn update_dense(&self, _gradients: &[f32], _alpha: f32) {}
}

/// Attention pooling (see [`AttentionWeights`]): each node's embedding is multi-head
/// self-attention over its feature embeddings, with query, key, value and output projections
/// shared by every feature and learned alongside them.  A reconstruction pools each sampled
/// neighbour from its own features, then takes the neighbours' weighted mean, as the GPU
/// trainer's nested reconstruction does.  Slower than averaging, but it learns which features
/// matter together.
pub struct AttentionFeatureModel {
    heads: usize,

    /// Randomly sample max_features of each node's features if provided
    max_features: Sample,

    /// Neighbors to consider for a reconstruction
    max_neighbor_nodes: Sample,

    /// If true, samples neighborhoods proportionally to their edge weights
    weighted_neighbor_sampling: bool,

    /// If true, a reconstruction weights its neighbours by their edge weights
    weighted_neighbor_averaging: bool,

    /// The current weights: None until `prepare` initialises them, unless warm-started.
    /// Readers clone the Arc; an update swaps in a new one.
    weights: RwLock<Option<Arc<AttentionWeights>>>,

    /// The weights' optimiser state, made by `prepare`.  Updates hold it while they run, so
    /// they apply one at a time.
    optimizer: Mutex<Option<DenseAdam>>
}

impl AttentionFeatureModel {
    pub fn new(
        heads: usize,
        max_features: Sample,
        max_neighbor_nodes: Sample,
        weighted_neighbor_sampling: bool,
        weighted_neighbor_averaging: bool
    ) -> Result<Self, Error> {
        if heads == 0 {
            return Err(Error::InvalidArgument("attention needs at least one head".into()))
        }
        Ok(AttentionFeatureModel {
            heads, max_features, max_neighbor_nodes, weighted_neighbor_sampling,
            weighted_neighbor_averaging, weights: RwLock::new(None), optimizer: Mutex::new(None)
        })
    }

    /// Starts training from `weights` (e.g. a GPU-trained model's) rather than fresh ones.
    pub fn warm_start(&self, weights: AttentionWeights) -> Result<(), Error> {
        if weights.heads() != self.heads {
            return Err(Error::InvalidArgument(format!(
                "the weights have {} heads, the model {}", weights.heads(), self.heads)))
        }
        *self.weights.write().expect("attention weights lock") = Some(Arc::new(weights));
        Ok(())
    }

    /// The current weights, once there are some.
    pub fn weights(&self) -> Option<Arc<AttentionWeights>> {
        self.weights.read().expect("attention weights lock").clone()
    }

    /// The aggregator that embeds nodes as this model does, once it has weights.
    pub fn aggregator(&self) -> Option<FeatureAggregator> {
        self.weights().map(|w| FeatureAggregator::attention((*w).clone()))
    }
}

/// Either feature model, chosen at run time: averaged, or attention with some number of heads.
pub enum FeatureModel {
    Averaged(AveragedFeatureModel),
    Attention(AttentionFeatureModel)
}

impl FeatureModel {
    /// Attention pooling with `attention_heads` heads, or averaging when None.
    pub fn new(
        attention_heads: Option<usize>,
        max_features: Sample,
        max_neighbor_nodes: Sample,
        weighted_neighbor_sampling: bool,
        weighted_neighbor_averaging: bool
    ) -> Result<Self, Error> {
        Ok(match attention_heads {
            Some(heads) => FeatureModel::Attention(AttentionFeatureModel::new(heads, max_features,
                max_neighbor_nodes, weighted_neighbor_sampling, weighted_neighbor_averaging)?),
            None => FeatureModel::Averaged(AveragedFeatureModel::new(max_features, max_neighbor_nodes,
                weighted_neighbor_sampling, weighted_neighbor_averaging))
        })
    }

    /// Continues an attention model from `aggregator`'s weights, which must be attention
    /// weights for `dims`-dimensional embeddings with the model's heads.
    pub fn warm_start_from(&self, aggregator: &FeatureAggregator, dims: usize) -> Result<(), Error> {
        let FeatureModel::Attention(model) = self else {
            return Err(Error::InvalidArgument("an aggregator warm-starts attention models only".into()))
        };
        let weights = aggregator.attention_weights().ok_or_else(||
            Error::InvalidArgument("the warm start must be an attention aggregator".into()))?;
        if weights.dims() != dims {
            return Err(Error::InvalidArgument(format!(
                "the aggregator is for {}-dimensional embeddings, the model trains {}", weights.dims(), dims)))
        }
        model.warm_start(weights.clone())
    }

    /// The aggregator that embeds nodes as the model does: averaging, or the attention weights
    /// once there are some.
    pub fn aggregator(&self) -> Option<FeatureAggregator> {
        match self {
            FeatureModel::Averaged(_) => Some(FeatureAggregator::averaged()),
            FeatureModel::Attention(model) => model.aggregator()
        }
    }

    /// [`EmbeddingPropagation::learn_features`](super::EmbeddingPropagation::learn_features)
    /// with this model.
    pub fn learn_features(
        &self,
        ep: &super::EmbeddingPropagation,
        graph: &crate::named_graph::NamedGraph,
        features: &mut crate::feature_store::FeatureSet,
        warm_start: Option<crate::embeddings::NodeEmbeddings>
    ) -> Result<crate::embeddings::NodeEmbeddings, Error> {
        match self {
            FeatureModel::Averaged(model) => ep.learn_features(graph, features, warm_start, model),
            FeatureModel::Attention(model) => ep.learn_features(graph, features, warm_start, model)
        }
    }
}

impl Model for AttentionFeatureModel {
    type Params = AttentionVars;

    fn prepare(&self, dims: usize, rng: &mut impl Rng) -> Result<(), Error> {
        let mut weights = self.weights.write().expect("attention weights lock");
        match weights.as_ref() {
            Some(w) if w.dims() != dims => return Err(Error::InvalidArgument(format!(
                "the attention weights are for {}-dimensional embeddings, training {}", w.dims(), dims))),
            Some(_) => {},
            None => *weights = Some(Arc::new(AttentionWeights::init(dims, self.heads, rng)?))
        }
        let count = weights.as_ref().map(|w| w.parameters().len()).unwrap_or(0);
        *self.optimizer.lock().expect("attention optimizer lock") = Some(DenseAdam::new(count));
        Ok(())
    }

    fn params(&self) -> AttentionVars {
        AttentionVars::new(&self.weights().expect("prepare runs before training"))
    }

    fn construct_node_embedding<R: Rng>(
        &self,
        params: &AttentionVars,
        node: NodeID,
        weight: f32,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut R
    ) -> (NodeCounts, ANode) {
        construct_from_multiple_nodes(std::iter::once((node, weight)), feature_store,
            feature_embeddings, self.max_features, Some(params), rng)
    }

    fn reconstruct_node_embedding<G: CGraph, R: Rng>(
        &self,
        params: &AttentionVars,
        graph: &G,
        node: NodeID,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut R
    ) -> (NodeCounts, ANode){
        reconstruct_node_embedding(
            graph,
            node,
            feature_store,
            feature_embeddings,
            self.max_neighbor_nodes,
            self.max_features,
            Some(params),
            self.weighted_neighbor_sampling,
            self.weighted_neighbor_averaging,
            rng)
    }

    fn construct_from_multiple_nodes<I: Iterator<Item=(NodeID, f32)>, R: Rng>(
        &self,
        params: &AttentionVars,
        nodes: I,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut R
    ) -> (NodeCounts, ANode) { 
        construct_from_multiple_nodes(nodes, feature_store, feature_embeddings,
            self.max_features, Some(params), rng)
    }

    fn dense_gradients(&self, params: &AttentionVars, graph: &Graph) -> Option<Vec<f32>> {
        params.gradients(graph)
    }

    fn update_dense(&self, gradients: &[f32], alpha: f32) {
        let mut optimizer = self.optimizer.lock().expect("attention optimizer lock");
        let optimizer = optimizer.as_mut().expect("prepare runs before training");
        let current = self.weights().expect("prepare runs before training");
        let mut params = current.parameters();
        optimizer.step(&mut params, gradients, alpha);
        // A step that produced non-finite weights is dropped rather than poisoning every
        // later embedding.
        if let Ok(updated) = current.with_parameters(params) {
            *self.weights.write().expect("attention weights lock") = Some(Arc::new(updated));
        }
    }
}

/// We track the number of times a features has been seen to help reduce the gradient graph we need
/// to compute.  It's a bit of a headache for the book keeping but the speed up is worth it.  Could
/// probably be abstracted better.
pub type NodeCounts = HashMap<usize, (ANode, f32)>;

/// Gets the feature embeddings for a node, adding or updating the counts, and returns the
/// features it sampled.
pub fn collect_embeddings_from_node<R: Rng>(
    node: NodeID,
    mut weight: f32,
    feature_store: &FeatureStore,
    feature_embeddings: &EmbeddingStore,
    feat_map: &mut NodeCounts,
    max_features: Sample,
    rng: &mut R
) -> Vec<usize> {
    let feats = feature_store.get_features(node);
    // Scale the node weight by the dropout correction so the feature sample stays unbiased.
    let (max_features, scalar) = max_features.sample(feats.len(), true, rng);
    weight *= scalar;
    let chosen: Vec<usize> = feats.choose_multiple(rng, max_features).cloned().collect();
    for feat in chosen.iter() {
        if let Some((_emb, count)) = feat_map.get_mut(feat) {
            *count += weight;
        } else {
            let emb = feature_embeddings.get_embedding(*feat);
            let v = Variable::pooled(emb);
            feat_map.insert(*feat, (v, weight));
        }
    }
    chosen
}

// H(n)
// Average the features associated with a node
// to create the node embedding
pub fn construct_node_embedding<R: Rng>(
    node: NodeID,
    weight: f32,
    feature_store: &FeatureStore,
    feature_embeddings: &EmbeddingStore,
    max_features: Sample,
    rng: &mut R
) -> (NodeCounts, ANode) {
    let mut feature_map = HashMap::new();
    collect_embeddings_from_node(node, weight, 
                                 feature_store, 
                                 feature_embeddings, 
                                 &mut feature_map,
                                 max_features,
                                 rng);

    let mean = mean_embeddings(feature_map.values());
    (feature_map, mean)
}

// ~H(n)
// The Expensive function.  We grab a node'ss neighbors
// and use the average of their features to construct
// an estimate of H(n)
fn reconstruct_node_embedding<G: CGraph, R: Rng>(
    graph: &G,
    node: NodeID,
    feature_store: &FeatureStore,
    feature_embeddings: &EmbeddingStore,
    max_nodes: Sample,
    max_features: Sample,
    attention: Option<&AttentionVars>,
    weighted_neighbor_sampling: bool,
    weighted_neighbor_averaging: bool,
    rng: &mut R
) -> (NodeCounts, ANode) {
    let (edges, cdf) = graph.get_edges(node);
    let degree = edges.len() as f32;
    let (nodes_to_sample, scale_weight) = max_nodes.sample(edges.len(), true, rng);

    // Edge probabilities, scaled so the average neighbor has weight 1.
    let neighbors = || edges.iter().cloned().zip(CDFtoP::new(cdf).map(move |p| p * degree));

    // Each branch picks the neighbors and their averaging weights.  When weighted averaging is
    // on, the reconstruction is an estimate of the edge-weighted neighborhood mean.
    let blend = move |(n, w): (NodeID, f32)| {
        (n, scale_weight * if weighted_neighbor_averaging { w } else { 1f32 })
    };
    let sampled: Box<dyn Iterator<Item=(NodeID, f32)>> = if nodes_to_sample == edges.len() {
        // Nothing to sample: blend all neighbors directly, streaming rather than collecting
        // since hub nodes can have millions of edges.
        Box::new(neighbors().map(blend))
    } else if weighted_neighbor_sampling {
        // Neighbors are drawn in proportion to edge weight, which already biases the sample
        // toward heavy edges.  Averaging it uniformly estimates the weighted mean; weighting it
        // again would count the edge weight twice.
        let sample = weighted_reservoir_sample(neighbors(), nodes_to_sample, rng);
        Box::new(sample.into_iter().map(move |(n, _)| (n, scale_weight)))
    } else {
        // A uniform sample, so weighted averaging has to apply the edge weights itself.
        Box::new(reservoir_sample(neighbors(), nodes_to_sample, rng).into_iter().map(blend))
    };

    construct_from_multiple_nodes(sampled,
        feature_store,
        feature_embeddings,
        max_features,
        attention,
        rng)
}

/// Embeds weighted `nodes` together.  Averaging pools all their features as one set (each
/// weighted by its nodes' weights); attention pools each node from its own features and takes
/// the nodes' weighted mean.  Either way every feature is one autograd leaf, in the returned
/// map, however many nodes share it.
fn construct_from_multiple_nodes<I: Iterator<Item=(NodeID, f32)>, R: Rng>(
    nodes: I,
    feature_store: &FeatureStore,
    feature_embeddings: &EmbeddingStore,
    max_features: Sample,
    attention: Option<&AttentionVars>,
    rng: &mut R,
) -> (NodeCounts, ANode) {
    let mut feature_map = HashMap::new();
    let mut pooled = Vec::new();
    for (node, weight) in nodes {
        let chosen = collect_embeddings_from_node(node, weight, feature_store,
                                     feature_embeddings, 
                                     &mut feature_map,
                                     max_features,
                                     rng);
        if let Some(vars) = attention {
            // A node's own features all count equally; its weight applies to its embedding.
            let tokens: Vec<(ANode, f32)> = chosen.iter()
                .map(|f| (feature_map[f].0.clone(), 1f32))
                .collect();
            pooled.push((attention_pool(&tokens, vars), weight));
        }
    }

    let mean = if attention.is_some() {
        mean_embeddings(pooled.iter())
    } else {
        mean_embeddings(feature_map.values())
    };
    (feature_map, mean)
}

pub fn mean_embeddings<'a>(
    items: impl Iterator<Item=&'a (ANode, f32)>
) -> ANode {
    let mut vs = Vec::new();
    let mut n = 0f32;
    items.for_each(|(emb, count)| {
        // Floating point math is pretty meh 
        if (*count - 1f32).abs() < 1e-6 {
            vs.push(emb.clone());
        } else {
            vs.push(emb * *count);
        }
        n += *count;
    });
    vs.sum_all() / n as f32
}

#[cfg(test)]
mod model_tests {
    use super::*;
    use rand_xorshift::XorShiftRng;
    use crate::distance::Distance;
    use crate::graph::{CSR, CumCSR};

    fn store_with_four_features() -> (FeatureStore, EmbeddingStore) {
        let mut fs = FeatureStore::new(1);
        fs.set_features(0, ["a", "b", "c", "d"].iter().map(|f| ("f", *f)));
        let es = EmbeddingStore::new(fs.num_features(), 2, Distance::Cosine);
        (fs, es)
    }

    /// The dropout correction is a multiplicative scale on the node weight.
    #[test]
    fn test_collect_embeddings_scales_weight() {
        let (fs, es) = store_with_four_features();
        let mut rng = XorShiftRng::seed_from_u64(1);

        // No dropout: each feature carries exactly the node weight.
        let mut fm = HashMap::new();
        collect_embeddings_from_node(0, 0.5, &fs, &es, &mut fm, Sample::All, &mut rng);
        assert_eq!(fm.len(), 4);
        for (_, (_, count)) in fm.iter() {
            assert_eq!(*count, 0.5);
        }

        // Keep 2 of 4 features: weight is scaled by n/k = 2.
        let mut fm = HashMap::new();
        collect_embeddings_from_node(0, 0.5, &fs, &es, &mut fm, Sample::Fixed(2), &mut rng);
        assert_eq!(fm.len(), 2);
        for (_, (_, count)) in fm.iter() {
            assert_eq!(*count, 1.0);
        }
    }

    /// Node 0 has a heavy edge to node 1 (weight 98) and light edges to nodes 2 and 3 (weight 1
    /// each).  Every node has a single feature of its own.
    fn weighted_star() -> (CumCSR, FeatureStore, EmbeddingStore) {
        let csr = CSR::construct_from_edges(vec![(0, 1, 98.), (0, 2, 1.), (0, 3, 1.)], false);
        let graph = CumCSR::convert(csr);
        let mut fs = FeatureStore::new(graph.len());
        for node in 0..graph.len() {
            fs.set_features(node, std::iter::once(("f", node.to_string())));
        }
        let es = EmbeddingStore::new(fs.num_features(), 2, Distance::Cosine);
        (graph, fs, es)
    }

    fn feature_of(fs: &FeatureStore, node: NodeID) -> usize {
        fs.get_features(node)[0]
    }

    /// Weighted sampling must pick neighbors in proportion to edge weight even when weighted
    /// averaging is off.
    #[test]
    fn test_weighted_neighbor_sampling_follows_edge_weights() {
        let (graph, fs, es) = weighted_star();
        let model = AveragedFeatureModel::new(Sample::All, Sample::Fixed(1), true, false);
        let heavy = feature_of(&fs, 1);

        let trials = 2000;
        let heavy_picks = (0..trials).filter(|seed| {
            let mut rng = XorShiftRng::seed_from_u64(*seed);
            let (counts, _) = model.reconstruct_node_embedding(&(), &graph, 0, &fs, &es, &mut rng);
            assert_eq!(counts.len(), 1);
            counts.contains_key(&heavy)
        }).count();

        // Expected rate is 98%; uniform sampling would give about 33%.
        let rate = heavy_picks as f32 / trials as f32;
        assert!(rate > 0.95, "heavy neighbor picked {:.3} of the time", rate);
    }

    /// Weighted sampling already favors heavy edges, so a weighted sample is averaged uniformly.
    /// Weighting it again would count the edge weight twice (an effective weight of w^2).
    #[test]
    fn test_weighted_sampling_and_averaging_do_not_double_count() {
        let (graph, fs, es) = weighted_star();
        let model = AveragedFeatureModel::new(Sample::All, Sample::Fixed(2), true, true);

        for seed in 0..50 {
            let mut rng = XorShiftRng::seed_from_u64(seed);
            let (counts, _) = model.reconstruct_node_embedding(&(), &graph, 0, &fs, &es, &mut rng);
            let weights: Vec<f32> = counts.values().map(|(_, w)| *w).collect();
            assert_eq!(weights.len(), 2);
            assert!((weights[0] - weights[1]).abs() < 1e-6,
                    "seed {}: sampled neighbors weighted {:?}", seed, weights);
        }
    }

    /// When the whole neighborhood is used there is nothing to sample, so weighted averaging
    /// blends neighbors by edge weight whether or not weighted sampling is on.
    #[test]
    fn test_weighted_averaging_over_full_neighborhood() {
        let (graph, fs, es) = weighted_star();
        for wns in [false, true] {
            let model = AveragedFeatureModel::new(Sample::All, Sample::All, wns, true);
            let mut rng = XorShiftRng::seed_from_u64(0);
            let (counts, _) = model.reconstruct_node_embedding(&(), &graph, 0, &fs, &es, &mut rng);
            let w = |node| counts[&feature_of(&fs, node)].1;
            assert!((w(1) / w(2) - 98.).abs() < 1e-3, "wns={}: {} vs {}", wns, w(1), w(2));
            assert!((w(2) - w(3)).abs() < 1e-6);
        }
    }
}

