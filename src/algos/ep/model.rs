//! The Embedding Propagation framework parameterizes over the feature aggregator - that is, given
//! a node with a set of features, how do we combine them to product a node embedding?
//! This module defines them
use simple_grad::*;
use hashbrown::HashMap;
use rand::prelude::*;
use rand_distr::{Distribution,Uniform};

use crate::FeatureStore;
use crate::EmbeddingStore;
use crate::graph::{Graph as CGraph,NodeID, CDFtoP};
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
    max_features: Option<usize>,

    /// In the case of the node reconstruction, we use max_neighbor_nodes instead of the entire
    /// neighborhood.  This is critical when we have nodes with large neighbors.
    max_neighbor_nodes: Option<usize>
}

impl AveragedFeatureModel {
    pub fn new(
        max_features: Option<usize>,
        max_neighbor_nodes: Option<usize>
    ) -> Self {
        AveragedFeatureModel { max_features, max_neighbor_nodes }
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
    max_features: Option<usize>,
    
    /// Max neighbors to consider for reconstruction
    max_neighbor_nodes: Option<usize>
}

impl AttentionFeatureModel {
    pub fn new(
        mha: MultiHeadedAttention,
        max_features: Option<usize>,
        max_neighbor_nodes: Option<usize>
    ) -> Self {
        AttentionFeatureModel { mha, max_features, max_neighbor_nodes }
    }
}

impl Model for AttentionFeatureModel {
    fn construct_node_embedding<R: Rng>(
        &self,
        node: NodeID,
        weight: f32,
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
    weight: f32,
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
    max_features: Option<usize>,
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
    max_features: Option<usize>,
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
    max_nodes: Option<usize>,
    max_features: Option<usize>,
    mha: Option<MultiHeadedAttention>,
    rng: &mut R
) -> (NodeCounts, ANode) {
    let (edges, weights) = &graph.get_edges(node);
    
    let mn = max_nodes.unwrap_or(edges.len());
    let weights = CDFtoP::new(weights).map(|p| p * edges.len() as f32);
    let it = edges.iter().cloned().zip(weights);
    if edges.len() <= mn {
        construct_from_multiple_nodes(it,
            feature_store,
            feature_embeddings,
            max_features,
            mha,
            rng)
    } else {
        let it = reservoir_sample(it, mn, rng).into_iter();
        construct_from_multiple_nodes(it,
            feature_store,
            feature_embeddings,
            max_features,
            mha,
            rng)
    }
}

fn reservoir_sample(
    it: impl Iterator<Item=(NodeID, f32)>,
    size: usize,
    rng: &mut impl Rng
) -> Vec<(NodeID, f32)> {
    let mut sample = Vec::with_capacity(size);
    for (i, n) in it.enumerate() {
        if i < size {
            sample.push(n);
        } else {
            let idx = Uniform::new(0, i).sample(rng);
            if idx < size {
                sample[idx] = n;
            }
        }
    }
    sample
}

fn construct_from_multiple_nodes<I: Iterator<Item=(NodeID, f32)>, R: Rng>(
    nodes: I,
    feature_store: &FeatureStore,
    feature_embeddings: &EmbeddingStore,
    max_features: Option<usize>,
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

