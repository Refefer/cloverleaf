use simple_grad::*;
use hashbrown::HashMap;
use rand::prelude::*;
use float_ord::FloatOrd;

use crate::FeatureStore;
use crate::EmbeddingStore;
use crate::graph::{Graph as CGraph,NodeID};

pub trait Model: Send + Sync {

    fn construct_node_embedding<R: Rng>(
        &self,
        node: NodeID,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut R
    ) -> (NodeCounts, ANode);

    fn reconstruct_node_embedding<G: CGraph, R: Rng>(
        &self,
        graph: &G,
        node: NodeID,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut R
    ) -> (NodeCounts, ANode);

    fn construct_from_multiple_nodes<I: Iterator<Item=NodeID>, R: Rng>(
        &self,
        nodes: I,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut R
    ) -> (NodeCounts, ANode); 

    fn parameters(&self) -> Vec<ANode>;
}

pub struct AveragedFeatureModel {
    max_features: Option<usize>,
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
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut R
    ) -> (NodeCounts, ANode) {
        construct_node_embedding(
            node,
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
            false,
            rng)
    }

    fn construct_from_multiple_nodes<I: Iterator<Item=NodeID>, R: Rng>(
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
            rng, false)
    }

    fn parameters(&self) -> Vec<ANode> {
        Vec::with_capacity(0)
    }
 
}

pub struct AttentionFeatureModel {
    max_features: Option<usize>,
    max_neighbor_nodes: Option<usize>
}

impl AttentionFeatureModel {
    pub fn new(
        max_features: Option<usize>,
        max_neighbor_nodes: Option<usize>
    ) -> Self {
        AttentionFeatureModel { max_features, max_neighbor_nodes }
    }
}

impl Model for AttentionFeatureModel {
    fn construct_node_embedding<R: Rng>(
        &self,
        node: NodeID,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut R
    ) -> (NodeCounts, ANode) {
        attention_construct_node_embedding(
            node,
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
            true,
            rng)
    }

    fn construct_from_multiple_nodes<I: Iterator<Item=NodeID>, R: Rng>(
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
            rng, true)
    }

    fn parameters(&self) -> Vec<ANode> {
        Vec::with_capacity(0)
    }
 
}

pub type NodeCounts = HashMap<usize, (ANode, usize)>;

pub fn collect_embeddings_from_node<R: Rng>(
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

// Attention H(n)
// Use scaled attention between features associated with a node
// to create the node embedding
pub fn attention_construct_node_embedding<R: Rng>(
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

    let mean = attention_mean(feature_map.values());
    (feature_map, mean)
}

pub fn attention_mean<'a>(
    it: impl Iterator<Item=&'a (ANode, usize)>
) -> ANode {
    let items: Vec<_> = it.collect();
    if items.len() == 1 {
        return get_value_vec(&items[0].0)
    }
    let attention_vec = items.iter().map(|it| get_attention_vec(&it.0)).collect::<Vec<_>>();
    
    // Get the attention for each feature
    let mut scaled = vec![Vec::with_capacity(items.len()); items.len()];
    for i in 0..items.len() {
        for j in (i+1)..items.len() {
            let (_iv, ic) = items[i];
            let iva = &attention_vec[i];
            let (_jv, jc) = items[j];
            let jva = &attention_vec[j];
            let dot = (&iva).dot(&jva);
            let num = ic * jc;
            let sdot = if num >= 1 {
                dot * (ic * jc) as f32
            } else {
                dot
            };
            scaled[i].push(sdot.clone());
            scaled[j].push(sdot);
        }
    }

    // Compute softmax
    let d_k = Constant::scalar((scaled[0][0].value().len() as f32).sqrt());

    let mut numers: Vec<_> = scaled.into_iter()
        .map(|dots| dots.sum_all() / &d_k)
        .collect();

    let max_value = numers.iter().map(|v| v.value()[0])
        .max_by_key(|v| FloatOrd(*v))
        .expect("Shouldn't be non-zero!");

    let mv = Constant::scalar(max_value);
    numers.iter_mut().for_each(|v| {
        *v = ((&*v) - &mv).exp()
    });

    let denom = numers.clone().sum_all();
    let softmax = numers.into_iter().map(|v| v / &denom);
    items.into_iter().zip(softmax)
        .map(|((feat, _c), attention)| get_value_vec(feat) * attention)
        .collect::<Vec<_>>().sum_all()
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
    attention: bool, 
    rng: &mut R
) -> (NodeCounts, ANode) {
    let edges = &graph.get_edges(node).0;
    
    if edges.len() <= max_nodes.unwrap_or(edges.len()) {
        construct_from_multiple_nodes(edges.iter().cloned(),
            feature_store,
            feature_embeddings,
            max_features,
            rng,
            attention)
    } else {
        let it = edges.choose_multiple(rng, max_nodes.unwrap()).cloned();
        construct_from_multiple_nodes(it,
            feature_store,
            feature_embeddings,
            max_features,
            rng,
            attention)
    }
}

fn construct_from_multiple_nodes<I: Iterator<Item=NodeID>, R: Rng>(
    nodes: I,
    feature_store: &FeatureStore,
    feature_embeddings: &EmbeddingStore,
    max_features: Option<usize>,
    rng: &mut R,
    attention: bool
) -> (NodeCounts, ANode) {
    let mut feature_map = HashMap::new();
    let mut new_nodes = Vec::with_capacity(0);
    for node in nodes {
        if attention {
            new_nodes.push(node.clone());
        }

        collect_embeddings_from_node(node, feature_store, 
                                     feature_embeddings, 
                                     &mut feature_map,
                                     max_features,
                                     rng);
    }

    let mean = if !attention {
        mean_embeddings(feature_map.values())
    } else {
        attention_multiple(new_nodes, feature_store, &feature_map)
    };
    (feature_map, mean)
}

fn get_value_vec(emb: &ANode) -> ANode {
    let v = emb.value().len();
    let mask_len = (v as f32 / 2f32) as usize;
    emb.slice(mask_len, v - mask_len)
}

fn get_attention_vec(emb: &ANode) -> ANode {
    let v = emb.value().len();
    let mask_len = (v as f32 / 2f32) as usize;
    emb.slice(0, mask_len)
}

fn attention_multiple(
    new_nodes: Vec<NodeID>,
    feature_store: &FeatureStore,
    feature_map: &NodeCounts
) -> ANode {
    let mut feats_per_node = HashMap::new();
    let mut output = Vec::new();
    for node in new_nodes {
        feats_per_node.clear();
        for feat in feature_store.get_features(node).iter() {
            if let Some((node, _)) = feature_map.get(feat) {
                let e = feats_per_node.entry(feat).or_insert_with(|| (node.clone(), 0usize));
                e.1 += 1;
            }
        }
        output.push((attention_mean(feats_per_node.values()), 1))
    }
    mean_embeddings(output.iter())
}

pub fn mean_embeddings<'a,I: Iterator<Item=&'a (ANode, usize)>>(items: I) -> ANode {
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


