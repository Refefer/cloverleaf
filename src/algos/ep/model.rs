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
            0,
            None,
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
            0, None, rng)
    }

    fn parameters(&self) -> Vec<ANode> {
        Vec::with_capacity(0)
    }
 
}

pub struct AttentionFeatureModel {
    dims: usize,
    window: Option<usize>,
    max_features: Option<usize>,
    max_neighbor_nodes: Option<usize>
}

impl AttentionFeatureModel {
    pub fn new(
        dims: usize,
        window: Option<usize>,
        max_features: Option<usize>,
        max_neighbor_nodes: Option<usize>
    ) -> Self {
        AttentionFeatureModel { dims, window, max_features, max_neighbor_nodes }
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
            self.dims,
            self.window,
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
            self.dims,
            self.window,
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
            self.dims, self.window, rng)
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
    attention_dims: usize,
    window: Option<usize>,
    rng: &mut R
) -> (NodeCounts, ANode) {
    let mut feature_map = HashMap::new();
    collect_embeddings_from_node(node, feature_store, 
                                 feature_embeddings, 
                                 &mut feature_map,
                                 max_features,
                                 rng);

    let mean = if window.is_some() {
        // Need to preserve order of features for context windows
        let feats = feature_store.get_features(node);
        let it = feats.iter()
            .filter(|f| feature_map.contains_key(*f))
            .map(|f| {
                feature_map.get(f).expect("Some type of error!")
            });
        attention_mean(it, attention_dims, window)
    } else {
        attention_mean(feature_map.values(), attention_dims, window)
    };
    (feature_map, mean)
}

#[derive(Clone)]
struct Attention {
    query: ANode,
    key: ANode,
    value: ANode
}

impl Attention {
    fn new(node: &ANode, attention_dims: usize) -> Self {
        let query = get_query_vec(&node, attention_dims);
        let key = get_key_vec(&node, attention_dims);
        let value = get_value_vec(&node, attention_dims);
        Attention {query, key, value}
    }
}

pub fn attention_mean<'a>(
    it: impl Iterator<Item=&'a (ANode, usize)>,
    attention_dims: usize,
    window: Option<usize>
) -> ANode {

    let items: Vec<_> = it.map(|(node, count)| {
        (Attention::new(node, attention_dims), *count)
    }).collect();

    if items.len() == 1 {
        return items[0].0.value.clone()
    }
    
    // Compute attention matrix
    let attention_matrix = compute_attention_matrix(&items, window);
    
    let att = compute_attention_softmax(attention_matrix, attention_dims);

    let summed_weights = att.sum_all();
    let n = items.len() as f32;
    items.into_iter().enumerate()
        .map(|(i, (at_i, _c))| at_i.value * summed_weights.slice(i, 1))
        .collect::<Vec<_>>().sum_all() / n
 }

fn compute_attention_matrix(
    items: &[(Attention, usize)],
    window: Option<usize>
) -> Vec<Vec<ANode>> {
    
     // Get the attention for each feature
    let zero = Constant::scalar(0.);
    let mut scaled = vec![vec![zero; items.len()]; items.len()];
    for i in 0..items.len() {
        let (j_start, j_end) = match window {
            Some(size) => {
                let start = if size > i { 0 } else {i - size };
                let stop = (i + size + 1).min(items.len());
                (start, stop)
            },
            None => (0, items.len())
        };

        let (at_i, ic) = &items[i];
        let row = &mut scaled[i];
        for j in j_start..j_end {
            let (at_j, jc) = &items[j];
            let mut dot_i_j = (&at_i.query).dot(&at_j.key);
            let num = ic * jc;
            if num >= 1 && window.is_none() {
                dot_i_j = dot_i_j * (num as f32);
            }
            row[j] = dot_i_j;
        }
    }
    scaled
}


fn compute_attention_softmax(
    attention_matrix: Vec<Vec<ANode>>,
    d_k: usize
) -> Vec<ANode> {
    // Compute softmax
    let d_k = Constant::scalar((d_k as f32).sqrt());

    // Compute softmax for each feature
    let mut att = Vec::with_capacity(attention_matrix.len());
    for row in attention_matrix.into_iter() {
        let row = row.concat() / &d_k;
        let sm = softmax(row);
        att.push(sm);
    }

    att
}

fn softmax(numers: ANode) -> ANode {

    let max_value = numers.value().iter()
        .max_by_key(|v| FloatOrd(**v))
        .expect("Shouldn't be non-zero!");
    let mv = Constant::scalar(*max_value);
    let n = (numers - &mv).exp();
    &n / n.sum()
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
    attention_dims: usize, 
    window: Option<usize>,
    rng: &mut R
) -> (NodeCounts, ANode) {
    let edges = &graph.get_edges(node).0;
    
    if edges.len() <= max_nodes.unwrap_or(edges.len()) {
        construct_from_multiple_nodes(edges.iter().cloned(),
            feature_store,
            feature_embeddings,
            max_features,
            attention_dims,
            window,
            rng)
    } else {
        let it = edges.choose_multiple(rng, max_nodes.unwrap()).cloned();
        construct_from_multiple_nodes(it,
            feature_store,
            feature_embeddings,
            max_features,
            attention_dims,
            window,
            rng)
    }
}

fn construct_from_multiple_nodes<I: Iterator<Item=NodeID>, R: Rng>(
    nodes: I,
    feature_store: &FeatureStore,
    feature_embeddings: &EmbeddingStore,
    max_features: Option<usize>,
    attention_dims: usize,
    window: Option<usize>,
    rng: &mut R,
) -> (NodeCounts, ANode) {
    let mut feature_map = HashMap::new();
    let mut new_nodes = Vec::with_capacity(0);
    for node in nodes {
        if attention_dims > 0 {
            new_nodes.push(node.clone());
        }

        collect_embeddings_from_node(node, feature_store, 
                                     feature_embeddings, 
                                     &mut feature_map,
                                     max_features,
                                     rng);
    }

    let mean = if attention_dims == 0 {
        mean_embeddings(feature_map.values())
    } else {
        attention_multiple(new_nodes, feature_store, &feature_map, attention_dims, window)
    };
    (feature_map, mean)
}

fn get_value_vec(emb: &ANode, dims: usize) -> ANode {
    let v = emb.value().len();
    emb.slice(2*dims, v - 2*dims)
}

fn get_query_vec(emb: &ANode, dims: usize) -> ANode {
    emb.slice(0, dims)
}

fn get_key_vec(emb: &ANode, dims: usize) -> ANode {
    emb.slice(dims, dims)
}

fn attention_multiple(
    new_nodes: Vec<NodeID>,
    feature_store: &FeatureStore,
    feature_map: &NodeCounts,
    attention_dims: usize,
    window: Option<usize>
) -> ANode {
    let mut feats_per_node = HashMap::new();
    let mut output = Vec::new();
    for node in new_nodes {
        feats_per_node.clear();
        let feats = feature_store.get_features(node);
        for feat in feats.iter() {
            if let Some((node, _)) = feature_map.get(feat) {
                let e = feats_per_node.entry(feat).or_insert_with(|| (node.clone(), 0usize));
                e.1 += 1;
            }
        }
        let it = feats.iter()
            .filter(|f| feats_per_node.contains_key(*f))
            .map(|f| {
                feats_per_node.get(f).expect("Some type of error!")
            });

        output.push((attention_mean(it, attention_dims, window), 1))
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


#[cfg(test)]
mod model_tests {
    use super::*;

    fn create_att_vecs() -> Vec<(Attention, usize)> {
        vec![
            (Attention::new(&Variable::new(vec![-1., -1., 1., 1.]), 1), 1),
            (Attention::new(&Variable::new(vec![0., 0., 2., 2.]), 1), 1),
            (Attention::new(&Variable::new(vec![1., 1., -1., -1.]), 1), 1)
        ]
    }

    #[test]
    fn test_attention_matrix_global() {
        let feats = create_att_vecs();

        let exp_att_matrix = vec![
            vec![vec![-1. * -1.], vec![-1. * 0.], vec![-1. * 1.]],
            vec![vec![0.  * -1.], vec![0.  * 0.], vec![0.  * 1.]],
            vec![vec![1.  * -1.], vec![1.  * 0.], vec![1.  * 1.]],
        ];

        let att_matrix = compute_attention_matrix(&feats, None);
        for (row, exp_row) in att_matrix.into_iter().zip(exp_att_matrix.into_iter()) {
            for (ri, eri) in row.into_iter().zip(exp_row.into_iter()) {
                assert_eq!(ri.value(), eri);
            }
        }

    }

    #[test]
    fn test_attention_matrix_cw() {
        let feats = create_att_vecs();

        let exp_att_matrix = vec![
            vec![vec![-1. * -1.], vec![-1. * 0.], vec![0.]],
            vec![vec![0.  * -1.], vec![0.  * 0.], vec![0.  * 1.]],
            vec![vec![0.], vec![1.  * 0.], vec![1.  * 1.]],
        ];

        let att_matrix = compute_attention_matrix(&feats, Some(1));
        for (row, exp_row) in att_matrix.into_iter().zip(exp_att_matrix.into_iter()) {
            assert_eq!(row.len(), exp_row.len());
            for (ri, eri) in row.into_iter().zip(exp_row.into_iter()) {
                assert_eq!(ri.value(), eri);
            }
        }

        let exp_att_matrix = vec![
            vec![vec![-1. * -1.], vec![-1. * 0.], vec![-1. * 1.]],
            vec![vec![0.  * -1.], vec![0.  * 0.], vec![0.  * 1.]],
            vec![vec![1.  * -1.], vec![1.  * 0.], vec![1.  * 1.]],
        ];

        // larger window than feat set
        let att_matrix = compute_attention_matrix(&feats, Some(10));
        for (row, exp_row) in att_matrix.into_iter().zip(exp_att_matrix.into_iter()) {
            assert_eq!(row.len(), exp_row.len());
            for (ri, eri) in row.into_iter().zip(exp_row.into_iter()) {
                assert_eq!(ri.value(), eri);
            }
        }
    }

    #[test]
    fn test_att_softmax() {
        let feats = create_att_vecs();

        let exp_softmax = vec![
            vec![0.66524096,0.24472847,0.09003057],
            vec![1./3.,1./3.,1./3.],
            vec![0.09003057, 0.24472847, 0.66524096],
        ];

        let att_matrix = compute_attention_matrix(&feats, None);
        let softmax_matrix = compute_attention_softmax(att_matrix, 1);

        assert_eq!(softmax_matrix.len(), exp_softmax.len());
        for (row, exp_row) in softmax_matrix.into_iter().zip(exp_softmax.into_iter()) {
            assert_eq!(row.value(), exp_row);
        }

    }

}
