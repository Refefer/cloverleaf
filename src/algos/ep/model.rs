//! The Embedding Propagation framework parameterizes over the feature aggregator - that is, given
//! a node with a set of features, how do we combine them to product a node embedding?
//! This module defines them
use crate::autograd::*;
use hashbrown::HashMap;
use rand::prelude::*;

use crate::FeatureStore;
use crate::EmbeddingStore;
use crate::graph::{Graph as CGraph,NodeID, CDFtoP};
use crate::algos::utils::{Sample,weighted_reservoir_sample,reservoir_sample};
use super::attention::{attention_mean,MultiHeadedAttention};

/// Main interface for model.  Needs to be threadsafe
pub trait Model: Send + Sync {

    /// Given a node, construct a node embedding from its features.
    fn construct_node_embedding<R: Rng>(
        &self,
        node: NodeID,
        weight: f32,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut R
    ) -> (NodeCounts, ANode);

    /// Given a node, reconstruct a node from its neighborhood
    fn reconstruct_node_embedding<G: CGraph, R: Rng>(
        &self,
        graph: &G,
        node: NodeID,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut R
    ) -> (NodeCounts, ANode);

    /// Construct multiple node embeddings - we use this for negatives, typically.
    fn construct_from_multiple_nodes<I: Iterator<Item=(NodeID, f32)>, R: Rng>(
        &self,
        nodes: I,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut R
    ) -> (NodeCounts, ANode); 

    /// Indicates whether it uses attention
    fn uses_attention(&self) -> bool;

    /// Size of the node embedding.  
    fn feature_dims(&self, d_model: usize) -> usize;

    /// Currently unused and should be axed (YAGNI).  If models have parmeters they can learn, we
    /// can expose them here.  Not wired up currently
    fn parameters(&self) -> Vec<ANode>;
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
    fn construct_node_embedding<R: Rng>(
        &self,
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

    fn feature_dims(&self, d_model: usize) -> usize {
        d_model
    }

    fn uses_attention(&self) -> bool {
        false
    }

    fn parameters(&self) -> Vec<ANode> {
        Vec::with_capacity(0)
    }
 
}

/// Attention model. Uses attention to combine features into node embeddings.  Slow, but pretty
/// powerful when used on the right feature set. 
pub struct AttentionFeatureModel {
    /// Defines the attention type and number of heads
    mha: MultiHeadedAttention,

    /// Max features to consider
    max_features: Sample,
    
    /// Max neighbors to consider for reconstruction
    max_neighbor_nodes: Sample,
    
    /// If true, samples neighborhoods proportionally to their edge weights
    weighted_neighbor_sampling: bool
       
}

impl AttentionFeatureModel {
    pub fn new(
        mha: MultiHeadedAttention,
        max_features: Sample,
        max_neighbor_nodes: Sample,
        weighted_neighbor_sampling: bool
    ) -> Self {
        AttentionFeatureModel { mha, max_features, max_neighbor_nodes, weighted_neighbor_sampling }
    }
}

impl Model for AttentionFeatureModel {
    fn construct_node_embedding<R: Rng>(
        &self,
        node: NodeID,
        _weight: f32,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut R
    ) -> (NodeCounts, ANode) {
        attention_construct_node_embedding(
            node,
            feature_store,
            feature_embeddings,
            self.max_features,
            self.mha.clone(),
            rng)
    }

    fn reconstruct_node_embedding<G: CGraph, R: Rng>(
        &self,
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
            Some(self.mha.clone()),
            self.weighted_neighbor_sampling,
            false,
            rng)
    }

    fn construct_from_multiple_nodes<I: Iterator<Item=(NodeID, f32)>, R: Rng>(
        &self,
        nodes: I,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut R
    ) -> (NodeCounts, ANode) { 
        construct_from_multiple_nodes(
            nodes, feature_store, 
            feature_embeddings, 
            self.max_features,
            Some(self.mha.clone()), rng)
    }

    fn uses_attention(&self) -> bool {
        true
    }

    fn feature_dims(&self, d_model: usize) -> usize {
        self.mha.num_heads * (self.mha.d_k * 2 + d_model)
    }

    fn parameters(&self) -> Vec<ANode> {
        Vec::with_capacity(0)
    }
 
}

/// We track the number of times a features has been seen to help reduce the gradient graph we need
/// to compute.  It's a bit of a headache for the book keeping but the speed up is worth it.  Could
/// probably be abstracted better.
pub type NodeCounts = HashMap<usize, (ANode, f32)>;

/// Gets the feature embeddings for a node, adding or updating the counts
pub fn collect_embeddings_from_node<R: Rng>(
    node: NodeID,
    mut weight: f32,
    feature_store: &FeatureStore,
    feature_embeddings: &EmbeddingStore,
    feat_map: &mut NodeCounts,
    max_features: Sample,
    rng: &mut R
) {
    let feats = feature_store.get_features(node);
    // Scale the node weight by the dropout correction so the feature sample stays unbiased.
    let (max_features, scalar) = max_features.sample(feats.len(), true, rng);
    weight *= scalar;
    for feat in feats.choose_multiple(rng, max_features) {
        if let Some((_emb, count)) = feat_map.get_mut(feat) {
            *count += weight;
        } else {
            let emb = feature_embeddings.get_embedding(*feat);
            let v = Variable::pooled(emb);
            feat_map.insert(*feat, (v, weight));
        }
    }
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

// Attention H(n)
// Use scaled attention between features associated with a node
// to create the node embedding
pub fn attention_construct_node_embedding<R: Rng>(
    node: NodeID,
    feature_store: &FeatureStore,
    feature_embeddings: &EmbeddingStore,
    max_features: Sample,
    mha: MultiHeadedAttention,
    rng: &mut R
) -> (NodeCounts, ANode) {
    let mut feature_map = HashMap::new();
    collect_embeddings_from_node(node, 1f32, 
                                 feature_store, 
                                 feature_embeddings, 
                                 &mut feature_map,
                                 max_features,
                                 rng);

    let mean = if mha.preserve_feature_order() {
        // Need to preserve order of features for context windows
        let feats = feature_store.get_features(node);
        let it = feats.iter()
            .filter(|f| feature_map.contains_key(*f))
            .map(|f| {
                feature_map.get(f).expect("Some type of error!")
            });
        attention_mean(it, &mha, rng)
    } else {
        attention_mean(feature_map.values(), &mha, rng)
    };
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
    mha: Option<MultiHeadedAttention>,
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
        mha,
        rng)
}

fn construct_from_multiple_nodes<I: Iterator<Item=(NodeID, f32)>, R: Rng>(
    nodes: I,
    feature_store: &FeatureStore,
    feature_embeddings: &EmbeddingStore,
    max_features: Sample,
    mha: Option<MultiHeadedAttention>,
    rng: &mut R,
) -> (NodeCounts, ANode) {
    let mut feature_map = HashMap::new();
    let mut new_nodes = Vec::with_capacity(0);
    for (node, weight) in nodes {
        if mha.is_some() {
            new_nodes.push(node.clone());
        }

        collect_embeddings_from_node(node, weight, feature_store, 
                                     feature_embeddings, 
                                     &mut feature_map,
                                     max_features,
                                     rng);
    }

    let mean = if let Some(attention) = mha {
        attention_multiple(new_nodes, feature_store, &feature_map, attention, rng)
    } else {
        mean_embeddings(feature_map.values())
    };
    (feature_map, mean)
}


fn attention_multiple(
    new_nodes: Vec<NodeID>,
    feature_store: &FeatureStore,
    feature_map: &NodeCounts,
    mha: MultiHeadedAttention,
    rng: &mut impl Rng
) -> ANode {
    let mut feats_per_node = HashMap::new();
    let mut output = Vec::new();
    for node in new_nodes {
        feats_per_node.clear();
        let feats = feature_store.get_features(node);
        for feat in feats.iter() {
            if let Some((node, _)) = feature_map.get(feat) {
                let e = feats_per_node.entry(feat).or_insert_with(|| (node.clone(), 0f32));
                e.1 += 1f32;
            }
        }
        let it = feats.iter()
            .filter(|f| feats_per_node.contains_key(*f))
            .map(|f| {
                feats_per_node.get(f).expect("Some type of error!")
            });

        output.push((attention_mean(it, &mha, rng), 1f32))
    }
    mean_embeddings(output.iter())
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
            let (counts, _) = model.reconstruct_node_embedding(&graph, 0, &fs, &es, &mut rng);
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
            let (counts, _) = model.reconstruct_node_embedding(&graph, 0, &fs, &es, &mut rng);
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
            let (counts, _) = model.reconstruct_node_embedding(&graph, 0, &fs, &es, &mut rng);
            let w = |node| counts[&feature_of(&fs, node)].1;
            assert!((w(1) / w(2) - 98.).abs() < 1e-3, "wns={}: {} vs {}", wns, w(1), w(2));
            assert!((w(2) - w(3)).abs() < 1e-6);
        }
    }
}

