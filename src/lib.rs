pub mod graph;
pub mod algos;
mod sampler;
mod vocab;
mod embeddings;
mod bitset;
mod hogwild;
mod progress;
mod feature_store;
mod io;

use std::sync::Arc;
use std::ops::Deref;
use std::fs::File;
use std::io::{Write,BufWriter,BufReader,BufRead};

use rayon::prelude::*;
use float_ord::FloatOrd;
use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError,PyIOError,PyKeyError};
use itertools::Itertools;
use fast_float::parse;

use crate::graph::{CSR,CumCSR,Graph as CGraph,NodeID};
use crate::vocab::Vocab;
use crate::sampler::Weighted;
use crate::embeddings::{EmbeddingStore,Distance as EDist,Entity};
use crate::feature_store::FeatureStore;
use crate::io::EmbeddingWriter;

use crate::algos::rwr::{Steps,RWR};
use crate::algos::grwr::{Steps as GSteps,GuidedRWR};
use crate::algos::reweighter::{Reweighter};
use crate::algos::ep::EmbeddingPropagation;
use crate::algos::ep::loss::Loss;
use crate::algos::ep::model::{AveragedFeatureModel,AttentionFeatureModel};
use crate::algos::ann::NodeDistance;
use crate::algos::aggregator::{WeightedAggregator,UnigramProbability,AvgAggregator,AttentionAggregator, EmbeddingBuilder};
use crate::algos::feat_propagation::propagate_features;
use crate::algos::alignment::{NeighborhoodAligner as NA};

const SEED: u64 = 20222022;

fn build_csr(edges: impl Iterator<Item=((String,String),(String,String),f32)>) -> (CSR, Vocab) {
    
    // Convert to NodeIDs
    let mut vocab = Vocab::new();
    eprintln!("Constructing vocab...");
    let edges: Vec<_> = edges.map(|((f_nt, f_n), (t_nt, t_n), w)| {
        let f_id = vocab.get_or_insert(f_nt, f_n);
        let t_id = vocab.get_or_insert(t_nt, t_n);
        (f_id, t_id, w)
    }).collect();

    eprintln!("Constructing CSR...");
    let csr = CSR::construct_from_edges(edges);
    (csr, vocab)
}

fn convert_scores(
    vocab: &Vocab, 
    scores: impl Iterator<Item=(NodeID, f32)>, 
    k: Option<usize>,
    filtered_node_type: Option<String>
) -> Vec<((String,String), f32)> {
    let mut scores: Vec<_> = scores.collect();
    scores.sort_by_key(|(_k, v)| FloatOrd(-*v));

    // Convert the list to named
    let k = k.unwrap_or(scores.len());
    scores.into_iter()
        .map(|(node_id, w)| {
            let (node_type, name) = vocab.get_name(node_id).unwrap();
            (((*node_type).clone(), (*name).clone()), w)
        })
        .filter(|((node_type, _node_name), _w)| {
            filtered_node_type.as_ref().map(|nt| nt == node_type).unwrap_or(true)
        })
        .take(k)
        .collect()
}

fn get_node_id(vocab: &Vocab, node_type: String, node: String) -> PyResult<NodeID> {
    if let Some(node_id) = vocab.get_node_id(node_type.clone(), node.clone()) {
        Ok(node_id)
    } else {
        Err(PyKeyError::new_err(format!(" Node '{}:{}' does not exist!", node_type, node)))
    }
}

#[pyclass]
#[derive(Clone)]
pub enum Distance {
    Cosine,
    Euclidean,
    ALT,
    Jaccard,
    Hamming
}

impl Distance {
    fn to_edist(&self) -> EDist {
        match self {
            Distance::Cosine => EDist::Cosine,
            Distance::Euclidean => EDist::Euclidean,
            Distance::ALT => EDist::ALT,
            Distance::Hamming => EDist::Hamming,
            Distance::Jaccard => EDist::Jaccard
        }
    }
}

#[pyclass]
pub struct Graph {
    graph: Arc<CumCSR>,
    vocab: Arc<Vocab>
}

#[pymethods]
impl Graph {

    #[new]
    fn new(edges: Vec<((String,String),(String,String),f32)>) -> Self {
        let (graph, vocab) = build_csr(edges.into_iter());
        eprintln!("Converting to CDF format...");
        Graph {
            graph: Arc::new(CumCSR::convert(graph)),
            vocab: Arc::new(vocab)
        }
    }

    pub fn contains_node(&self, name: (String, String)) -> bool {
        get_node_id(self.vocab.deref(), name.0, name.1).is_ok()
    }

    pub fn nodes(&self) -> usize {
        self.graph.len()
    }

    pub fn edges(&self) -> usize {
        self.graph.edges()
    }

    pub fn get_edges(&self, node: (String,String)) -> PyResult<(Vec<(String, String)>, Vec<f32>)> {
        let node_id = get_node_id(self.vocab.deref(), node.0, node.1)?;
        let (edges, weights) = self.graph.get_edges(node_id);
        let names = edges.into_iter()
            .map(|node_id| {
                let (nt, n) = self.vocab.get_name(*node_id).unwrap();
                ((*nt).clone(), (*n).clone())
            }).collect();
        Ok((names, weights.to_vec()))
    }

    pub fn vocab(&self) -> VocabIterator {
        VocabIterator::new(self.vocab.clone())
    }

    /// Saves a graph to disk
    pub fn save(&self, path: &str) -> PyResult<()> {
        let f = File::create(path)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        let mut bw = BufWriter::new(f);
        for node in 0..self.graph.len() {
            let (f_node_type, f_name) = self.vocab.get_name(node)
                .expect("Programming error!");

            let (edges, weights) = self.graph.get_edges(node);
            for (out_node, weight) in edges.iter().zip(weights.iter()) {

                let (t_node_type, t_name) = self.vocab.get_name(*out_node)
                    .expect("Programming error!");
                writeln!(&mut bw, "{}\t{}\t{}\t{}\t{}", f_node_type, f_name, t_node_type, t_name, weight)
                    .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;
            }
        }
        Ok(())
    }
    
    /// Loads a graph from disk
    #[staticmethod]
    pub fn load(path: &str, edge_type: EdgeType) -> PyResult<Self> {
        let f = File::open(path)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        let br = BufReader::new(f);
        let mut vocab = Vocab::new();
        let mut edges = Vec::new();
        for (i, line) in br.lines().enumerate() {
            let line = line.unwrap();
            let pieces: Vec<_> = line.split('\t').collect();
            if pieces.len() != 5 {
                return Err(PyValueError::new_err(format!("{}: Malformed graph file: Expected 5 fields!", i)))
            }
            let f_id = vocab.get_or_insert(pieces[0].to_string(), pieces[1].to_string());
            let t_id = vocab.get_or_insert(pieces[2].to_string(), pieces[3].to_string());
            let w: f32 = pieces[4].parse()
                .map_err(|e| PyValueError::new_err(format!("{}: Malformed graph file! {} - {:?}", i, e, pieces[4])))?;

            edges.push((f_id, t_id, w));
            if matches!(edge_type, EdgeType::Undirected) {
                edges.push((t_id, f_id, w));
            }
        }
        eprintln!("Read {} nodes, {} edges...", vocab.len(), edges.len());

        let csr = CSR::construct_from_edges(edges);

        let g = Graph {
            graph: Arc::new(CumCSR::convert(csr)),
            vocab: Arc::new(vocab)
        };

        Ok(g)

    }


}

#[pyclass]
#[derive(Clone)]
struct RandomWalker {
    restarts: f32,
    walks: usize,
    beta: Option<f32>
}

#[pymethods]
impl RandomWalker {

    #[new]
    fn new(restarts: f32, walks: usize, beta: Option<f32>) -> Self {
        RandomWalker { restarts, walks, beta }
    }

    pub fn walk(
        &self, 
        graph: &Graph,
        node: (String, String), 
        seed: Option<u64>, 
        k: Option<usize>, 
        filter_type: Option<String>
    ) -> PyResult<Vec<((String,String), f32)>> {

        let node_id = get_node_id(graph.vocab.deref(), node.0, node.1)?;
        
        let steps = if self.restarts >= 1. {
            Steps::Fixed(self.restarts as usize)
        } else if self.restarts > 0. {
            Steps::Probability(self.restarts)
        } else {
            return Err(PyValueError::new_err("Alpha must be between [0, inf)"))
        };

        let rwr = RWR {
            steps: steps,
            walks: self.walks,
            beta: self.beta.unwrap_or(0.5),
            seed: seed.unwrap_or(SEED)
        };

        let results = rwr.sample(graph.graph.as_ref(), &Weighted, node_id);

        Ok(convert_scores(&graph.vocab, results.into_iter(), k, filter_type))
    }

}

#[pyclass]
#[derive(Clone)]
struct BiasedRandomWalker {
    restarts: f32,
    walks: usize,
    beta: Option<f32>,
    blend: Option<f32>,
}

#[pymethods]
impl BiasedRandomWalker {

    #[new]
    fn new(restarts: f32, walks: usize, beta: Option<f32>, blend: Option<f32>) -> Self {
        BiasedRandomWalker { restarts, walks, beta, blend }
    }

    pub fn walk(
        &self, 
        graph: &Graph,
        embeddings: &NodeEmbeddings,
        node: (String,String), 
        context: &Query,
        k: Option<usize>, 
        seed: Option<u64>, 
        rerank_context: Option<&Query>,
        filter_type: Option<String>
    ) -> PyResult<Vec<((String,String), f32)>> {
        let node_id = get_node_id(graph.vocab.deref(), node.0, node.1)?;
        let g_emb = lookup_embedding(context, embeddings)?;
        
        let steps = if self.restarts >= 1. {
            GSteps::Fixed(self.restarts as usize)
        } else if self.restarts > 0. {
            GSteps::Probability(self.restarts, (1. / self.restarts).ceil() as usize)
        } else {
            return Err(PyValueError::new_err("Alpha must be between [0, inf)"))
        };

        let grwr = GuidedRWR {
            steps: steps,
            walks: self.walks,
            alpha: self.blend.unwrap_or(0.5),
            beta: self.beta.unwrap_or(0.5),
            seed: seed.unwrap_or(SEED)
        };

        let node_embeddings = &embeddings.embeddings;
        let mut results = grwr.sample(graph.graph.as_ref(), 
                                  &Weighted, node_embeddings, node_id, g_emb);
        
        // Reweight results if requested
        if let Some(cn) = rerank_context {
            println!("Reranking...");
            let c_emb = lookup_embedding(cn, embeddings)?;
            Reweighter::new(self.blend.unwrap_or(0.5))
                .reweight(&mut results, node_embeddings, c_emb);
        }

        Ok(convert_scores(&graph.vocab, results.into_iter(), k, filter_type))
    }


}

#[pyclass]
#[derive(Clone)]
pub enum EdgeType {
    Directed,
    Undirected
}

#[pyclass]
struct GraphBuilder {
    vocab: Vocab,
    edges: Vec<(NodeID, NodeID, f32)>
}

#[pymethods]
impl GraphBuilder {
    #[new]
    pub fn new() -> Self {
        GraphBuilder {
            vocab: Vocab::new(),
            edges: Vec::new()
        }
    }

    pub fn add_edge(
        &mut self, 
        from_node: (String, String), 
        to_node: (String,String),
        weight: f32, 
        node_type: EdgeType
    ) {
        let f_id = self.vocab.get_or_insert(from_node.0, from_node.1);
        let t_id = self.vocab.get_or_insert(to_node.0, to_node.1);
        self.edges.push((f_id, t_id, weight));
        if matches!(node_type, EdgeType::Undirected) {
            self.edges.push((t_id, f_id, weight));
        }
    }

    pub fn build_graph(&mut self) -> Graph {
        let mut vocab = Vocab::new(); 
        std::mem::swap(&mut vocab, &mut self.vocab);
        let mut edges = Vec::new();
        std::mem::swap(&mut edges, &mut self.edges);
        let graph = CSR::construct_from_edges(edges);

        Graph {
            graph: Arc::new(CumCSR::convert(graph)),
            vocab: Arc::new(vocab)
        }
    }

}

#[pyclass]
#[derive(Clone)]
struct EPLoss {
    loss: Loss
}

#[pymethods]
impl EPLoss {

    #[staticmethod]
    pub fn margin(gamma: f32, negatives: Option<usize>) -> Self {
        EPLoss { loss: Loss::MarginLoss(gamma, negatives.unwrap_or(1)) }
    }

    #[staticmethod]
    pub fn contrastive(temperature: f32, negatives: usize) -> Self {
        EPLoss { loss: Loss::Contrastive(temperature, negatives.max(1)) }
    }

    #[staticmethod]
    pub fn starspace(gamma: f32, negatives: usize) -> Self {
        EPLoss { loss: Loss::StarSpace(gamma, negatives.max(1)) }
    }

    #[staticmethod]
    pub fn ppr(gamma: f32, negatives: usize, restart_p: f32) -> Self {
        EPLoss { loss: Loss::PPR(gamma, negatives.max(1), restart_p) }
    }

}

enum ModelType {
    Averaged(AveragedFeatureModel),
    Attention(AttentionFeatureModel)
}

#[pyclass]
struct EmbeddingPropagator {
    ep: EmbeddingPropagation,
    model: ModelType 
}

#[pymethods]
impl EmbeddingPropagator {
    #[new]
    pub fn new(
        alpha: Option<f32>, 
        loss: Option<EPLoss>,
        batch_size: Option<usize>, 
        dims: Option<usize>,
        passes: Option<usize>,
        seed: Option<u64>,
        max_nodes: Option<usize>,
        max_features: Option<usize>,
        indicator: Option<bool>,
        attention: Option<usize>
    ) -> Self {
        let ep = EmbeddingPropagation {
            alpha: alpha.unwrap_or(0.9),
            batch_size: batch_size.unwrap_or(50),
            dims: dims.unwrap_or(100),
            passes: passes.unwrap_or(100),
            loss: loss.map(|l|l.loss).unwrap_or(Loss::MarginLoss(1f32,1)),
            seed: seed.unwrap_or(SEED),
            indicator: indicator.unwrap_or(true)
        };

        let model = if let Some(dims) = attention {
            ModelType::Attention(AttentionFeatureModel::new(dims, None, max_features, max_nodes))
        } else {
            ModelType::Averaged(AveragedFeatureModel::new(max_features, max_nodes))
        };

        EmbeddingPropagator{ ep, model }
    }

    pub fn learn_features(
        &mut self, 
        graph: &Graph, 
        features: &mut FeatureSet,
        feature_embeddings: Option<&mut NodeEmbeddings>
    ) -> NodeEmbeddings {

        features.features.fill_missing_nodes();

        // Pull out the EmbeddingStore
        let feature_embeddings = feature_embeddings.map(|fes| {
           let mut sfes = EmbeddingStore::new(fes.vocab.len(), 0, EDist::Cosine);
           std::mem::swap(&mut sfes, &mut fes.embeddings);
           sfes
        });

        let feat_embeds = match &self.model {
            ModelType::Averaged(model) => {
                self.ep.learn(
                    graph.graph.as_ref(), 
                    &mut features.features,
                    feature_embeddings,
                    model
                )
            },
            ModelType::Attention(model) => {
                self.ep.learn(
                    graph.graph.as_ref(), 
                    &mut features.features,
                    feature_embeddings,
                    model
                )
            }
        };

        let vocab = features.features.clone_vocab();

        let feature_embeddings = NodeEmbeddings {
            vocab: Arc::new(vocab),
            embeddings: feat_embeds};

        feature_embeddings

    }
}

#[pyclass]
pub struct FeatureSet {
    vocab: Arc<Vocab>,
    features: FeatureStore
}

#[pymethods]
impl FeatureSet {
    #[new]
    pub fn new(graph: &Graph, namespace: Option<String>) -> Self {
        let ns = namespace.unwrap_or_else(|| "feat".to_string());
        FeatureSet {
            features: FeatureStore::new(graph.graph.len(), ns),
            vocab: graph.vocab.clone()
        }
    }

    pub fn add_features(&mut self, node: (String,String), features: Vec<String>) -> PyResult<()> {
        let node_id = get_node_id(self.vocab.deref(), node.0, node.1)?;
        self.features.set_features(node_id, features);
        Ok(())
    }

    pub fn get_features(&self, node: (String,String)) -> PyResult<Vec<String>> {
        let node_id = get_node_id(self.vocab.deref(), node.0, node.1)?;
        Ok(self.features.get_pretty_features(node_id))
    }

    pub fn load_features(&mut self, path: String, error_on_missing: Option<bool>) -> PyResult<()> {
        let f = File::open(path)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        let br = BufReader::new(f);
        let strict = error_on_missing.unwrap_or(true);
        for line in br.lines() {
            let line = line.unwrap();
            let pieces: Vec<_> = line.split('\t').collect();
            if pieces.len() != 3 {
                return Err(PyValueError::new_err("Malformed feature line! Need node_type<TAB>name<TAB>f1 f2 ..."))
            }
            let bow = pieces[2].split_whitespace()
                .map(|s| s.to_string()).collect();
            let ret = self.add_features((pieces[0].to_string(), pieces[1].to_string()), bow);
            if strict {
                ret?
            } 
        }
        Ok(())
    }

    pub fn num_features(&self) -> usize {
        self.features.num_features()
    }

    pub fn vocab(&self) -> VocabIterator {
        VocabIterator::new(self.vocab.clone())
    }

    pub fn prune_min_count(&self, count: usize) -> Self {
        FeatureSet {
            features: self.features.prune_min_count(count),
            vocab: self.vocab.clone()
        }
    }

}

#[pyclass]
pub struct FeaturePropagator {
    k: usize,
    threshold: f32,
    max_iters: usize
}

#[pymethods]
impl FeaturePropagator {
    #[new]
    pub fn new(k: usize, threshold: Option<f32>, max_iters: Option<usize>) -> Self {
        FeaturePropagator { 
            k: k, 
            threshold: threshold.unwrap_or(0.),
            max_iters: max_iters.unwrap_or(20)
        }
    }

    pub fn propagate(&self,
        graph: &Graph,
        features: &mut FeatureSet
    ) {
        propagate_features(
            graph.graph.deref(), 
            &mut features.features, 
            self.max_iters,
            self.k,
            self.threshold);
    }

}

#[derive(Clone)]
enum AggregatorType {
    Averaged,
    Weighted {
        alpha: f32, 
        vocab: Arc<Vocab>,
        unigrams: Arc<UnigramProbability>
    },
    Attention {
        dims: usize,
        window: Option<usize>
    }
}

#[pyclass]
#[derive(Clone)]
pub struct FeatureAggregator {
    at: AggregatorType
}

#[pymethods]
impl FeatureAggregator {

    #[staticmethod]
    pub fn Averaged() -> Self {
        FeatureAggregator { at: AggregatorType::Averaged }
    }

    #[staticmethod]
    pub fn Attention(dims: usize, window: Option<usize>) -> Self {
        FeatureAggregator { at: AggregatorType::Attention {dims, window} }
    }

    #[staticmethod]
    pub fn Weighted(alpha: f32, fs: &FeatureSet) -> Self {
        let unigrams = Arc::new(UnigramProbability::new(&fs.features));
        let vocab = fs.vocab.clone();
        FeatureAggregator { at: AggregatorType::Weighted {alpha, vocab, unigrams} }
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        let f = File::create(path)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        let mut bw = BufWriter::new(f);
        match &self.at {
            AggregatorType::Averaged => {
                writeln!(&mut bw, "Averaged")
                    .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;
            },
            AggregatorType::Attention { dims, window } => {
                writeln!(&mut bw, "Attention")
                    .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;
                writeln!(&mut bw, "{}", dims)
                    .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;
                writeln!(&mut bw, "{}", window.unwrap_or(0))
                    .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;
            },
            AggregatorType::Weighted { alpha, vocab, unigrams } => {
                writeln!(&mut bw, "Weighted")
                    .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;
                writeln!(&mut bw, "{}", alpha)
                    .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;
                for (node, p_wi) in unigrams.iter().enumerate() {
                    if let Some((f_node_type, f_name)) = vocab.get_name(node) {
                        writeln!(&mut bw, "{}\t{}\t{}", f_node_type, f_name, *p_wi as f64)
                            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;
                    } else {
                        // We've moved to node individual embeddings
                        break
                    }
                }
            }
        }
        Ok(())
    }

    #[staticmethod]
    pub fn load(path: String) -> PyResult<Self> {
        let f = File::open(path)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        let mut br = BufReader::new(f);

        // Find the type
        let mut line = String::new();
        br.read_line(&mut line)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        match line.trim_end() {
            "Averaged" => Ok(FeatureAggregator::Averaged()),
            "Attention" => {
                line.clear();
                br.read_line(&mut line)
                    .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;
                let dims = line.trim_end().parse::<usize>()
                    .map_err(|_e| PyValueError::new_err(format!("invalid dim! {:?}", line)))?;
                let window = line.trim_end().parse::<usize>()
                    .map_err(|_e| PyValueError::new_err(format!("invalid dim! {:?}", line)))?;
                let window = if window == 0 {
                    None
                } else {
                    Some(window)
                };

                Ok(FeatureAggregator::Attention(dims, window))
            },
            "Weighted" => {
                // get alpha
                line.clear();
                br.read_line(&mut line)
                    .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;
                let alpha = line.trim_end().parse::<f32>()
                    .map_err(|_e| PyValueError::new_err(format!("invalid dim! {:?}", line)))?;

                let mut vocab = Vocab::new();
                let mut p_w = Vec::new();
                for line in br.lines() {
                    let line = line.unwrap();
                    let pieces: Vec<_> = line.split('\t').collect();
                    if pieces.len() != 3 {
                        return Err(PyValueError::new_err("Malformed feature line! Need node_type<TAB>name<TAB>weight ..."))
                    }
                    let p_wi = pieces[2].parse::<f64>()
                        .map_err(|_e| PyValueError::new_err(format!("Tried to parse weight and failed:{:?}", line)))?;
                    let node_id = vocab.get_or_insert(pieces[0].into(), pieces[1].into());
                    if node_id < p_w.len() {
                        return Err(PyValueError::new_err(format!("Duplicate feature found:{} {}", pieces[0], pieces[1])))
                    }
                    p_w.push(p_wi as f32);
                }
                let unigrams = Arc::new(UnigramProbability::from_vec(p_w));
                let at = AggregatorType::Weighted { alpha, vocab:Arc::new(vocab), unigrams };
                Ok(FeatureAggregator { at })

            },
            line => Err(PyValueError::new_err(format!("Unknown aggregator type: {}", line)))?
        }

    }


}

#[pyclass]
pub struct NodeEmbedder {
    feat_agg: FeatureAggregator
}

impl NodeEmbedder {

    fn get_aggregator<'a>(
        &'a self, 
        es: &'a EmbeddingStore, 
        aggregator_type: &'a AggregatorType
    ) -> Box<dyn EmbeddingBuilder + Send + Sync + 'a> {
        match aggregator_type {
            AggregatorType::Weighted {alpha, vocab:_, unigrams } => {
                Box::new(WeightedAggregator::new(es, unigrams, *alpha))
            },
            AggregatorType::Averaged => {
                Box::new(AvgAggregator::new(es))
            },
            AggregatorType::Attention {dims, window} => {
                Box::new(AttentionAggregator::new(es, *dims, *window))
            }

        }
    }

    fn get_attention_size(&self) -> usize {
        match self.feat_agg.at {
            AggregatorType::Attention { dims, window:_ } => 2 * dims,
            _ => 0
        }
    }

    fn get_dims(&self, feat_embs: &NodeEmbeddings) -> usize {
        feat_embs.dims() - self.get_attention_size()
    }
}

#[pymethods]
impl NodeEmbedder {
    #[new]
    pub fn new(feat_agg: FeatureAggregator) -> Self {
        NodeEmbedder { feat_agg }
    }

    pub fn embed_graph(
        &self, 
        graph: &Graph,
        feat_set: &FeatureSet, 
        feature_embeddings: &NodeEmbeddings,
    ) -> NodeEmbeddings {

        let num_nodes = graph.nodes();
        let dims = self.get_dims(feature_embeddings);

        let es = EmbeddingStore::new(num_nodes, dims, EDist::Cosine);
        let agg = self.get_aggregator(&feature_embeddings.embeddings, &self.feat_agg.at);
        (0..num_nodes).into_par_iter().for_each(|node| {
            // Safe to access in parallel
            let embedding = es.get_embedding_mut_hogwild(node);
            agg.construct(feat_set.features.get_features(node), embedding);
        });

        NodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings: es
        }
    }

    pub fn embed_adhoc(
        &self, 
        features: Vec<(String, String)>,
        feature_embeddings: &NodeEmbeddings,
        strict: Option<bool>
    ) -> PyResult<Vec<f32>> {
        let v = feature_embeddings.vocab.deref();
        let mut ids = Vec::new();
        for (node_type, node_name) in features.into_iter() {
            if let Ok(feat_id) = get_node_id(v, node_type.clone(), node_name.clone()) {
                ids.push(feat_id)
            } else if strict.unwrap_or(true) {
                return Err(PyKeyError::new_err(format!("Feature {} {} does not exist!", node_type, node_name)))
            }
        }

        let dims = self.get_dims(feature_embeddings);
        let mut embedding = vec![0.; dims];
        if ids.len() > 0 {
            let agg = self.get_aggregator(&feature_embeddings.embeddings, &self.feat_agg.at);
            agg.construct(&ids, &mut embedding);
        }
        Ok(embedding)
    }

}

fn count_lines(path: &str, node_type: &Option<String>) -> std::io::Result<usize> {
    let f = File::open(path)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

    let br = BufReader::new(f);
    let mut count = 0;
    let filter_node = node_type.as_ref();
    for line in br.lines() {
        let line = line?;
        if let Some(p) = filter_node {
            if line.starts_with(p) {
                count += 1;
            }
        } else {
            count += 1;
        }
    }
    Ok(count)
}

#[pyclass]
struct DistanceEmbedder {
    landmarks: algos::dist::LandmarkSelection,
    n_landmarks: usize
}

#[pymethods]
impl DistanceEmbedder {
    #[new]
    pub fn new(n_landmarks: usize, seed: Option<u64>) -> Self {
        let ls = if let Some(seed) = seed {
            algos::dist::LandmarkSelection::Random(seed)
        } else {
            algos::dist::LandmarkSelection::Degree
        };
        DistanceEmbedder {
            landmarks: ls,
            n_landmarks
        }

    }

    pub fn learn(&self, graph: &Graph) -> NodeEmbeddings {
        let es = crate::algos::dist::construct_walk_distances(graph.graph.as_ref(), self.n_landmarks, self.landmarks);
        NodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings: es
        }
    }
}

#[pyclass]
struct ClusterLPAEmbedder{
    k: usize, 
    passes: usize, 
    seed: Option<u64>
}

#[pymethods]
impl ClusterLPAEmbedder {
    #[new]
    pub fn new(k: usize, passes: usize, seed: Option<u64>) -> Self {
        ClusterLPAEmbedder {
            k, passes, seed
        }
    }

    pub fn learn(&self, graph: &Graph) -> NodeEmbeddings {
        let seed = self.seed.unwrap_or(SEED);
        let es = crate::algos::lpa::construct_lpa_embedding(graph.graph.as_ref(), self.k, self.passes, seed);
        NodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings: es
        }
    }
}

#[pyclass]
struct SLPAEmbedder {
    k: usize, 
    threshold: f32, 
    seed: Option<u64>
}

#[pymethods]
impl SLPAEmbedder {
    #[new]
    pub fn new(k: usize, threshold: f32, seed: Option<u64>) -> Self {
        SLPAEmbedder {
            k, threshold, seed
        }
    }

    pub fn learn(&self, graph: &Graph) -> NodeEmbeddings {
        let seed = self.seed.unwrap_or(SEED);
        let es = crate::algos::slpa::construct_slpa_embedding(graph.graph.as_ref(), self.k, self.threshold, seed);
        NodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings: es
        }
    }

}

#[pyclass]
pub struct NodeEmbeddings {
    vocab: Arc<Vocab>,
    embeddings: EmbeddingStore
}

#[pymethods]
impl NodeEmbeddings {
    #[new]
    pub fn new(graph: &Graph, dims: usize, distance: Distance) -> Self {
        let dist = distance.to_edist();

        let es = EmbeddingStore::new(graph.graph.len(), dims, dist);
        NodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings: es
        }
    }

    pub fn contains(&self, node: (String, String)) -> bool {
        get_node_id(self.vocab.deref(), node.0, node.1).is_ok()
    }

    pub fn get_embedding(&mut self, node: (String,String)) -> PyResult<Vec<f32>> {
        let node_id = get_node_id(self.vocab.deref(), node.0, node.1)?;
        Ok(self.embeddings.get_embedding(node_id).to_vec())
    }

    pub fn set_embedding(&mut self, node: (String,String), embedding: Vec<f32>) -> PyResult<()> {
        let node_id = get_node_id(self.vocab.deref(), node.0, node.1)?;
        let es = &mut self.embeddings;
        es.set_embedding(node_id, &embedding);
        Ok(())
    }

    pub fn vocab(&self) -> VocabIterator {
        VocabIterator::new(self.vocab.clone())
    }

    pub fn nearest_neighbor(
        &self, 
        emb: Vec<f32>, 
        k: usize,
        filter_type: Option<String>
    ) -> Vec<((String,String), f32)> {
        let emb = Entity::Embedding(&emb);
        let dists = if let Some(node_type) = filter_type {
            let ant = Arc::new(node_type);
            let filter_type = Some(&ant);
            self.embeddings.nearest_neighbor(&emb, k, |node_id| {
                let nt = self.vocab.get_node_type(node_id);
                nt == filter_type
            })
        } else {
            self.embeddings.nearest_neighbor(&emb, k, |_node_id| true)
        };
        convert_node_distance(&self.vocab, dists)
    }

    pub fn dims(&self) -> usize {
        self.embeddings.dims()
    }

    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        let mut writer = EmbeddingWriter::new(path, self.vocab.as_ref())
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        let it = (0..self.vocab.len())
            .map(|node_id| (node_id, self.embeddings.get_embedding(node_id)));

        writer.stream(it)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        Ok(())
    }

    #[staticmethod]
    pub fn load(
        path: &str, 
        distance: Distance, 
        node_type: Option<String>, 
        chunk_size: Option<usize>
    ) -> PyResult<Self> {
        let num_embeddings = count_lines(path, &node_type)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        let f = File::open(path)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        let br = BufReader::new(f);
        let mut vocab = Vocab::new();
        
        // Place holder
        let mut es = EmbeddingStore::new(0, 0, EDist::Cosine);
        let filter_node = node_type.as_ref();
        let mut i = 0;
        let mut buffer = Vec::with_capacity(chunk_size.unwrap_or(1_000));
        let mut p_buffer = Vec::with_capacity(buffer.capacity());
        for chunk in &br.lines().map(|l| l.unwrap()).chunks(buffer.capacity()) {
            buffer.clear();
            p_buffer.clear();
            
            // Read lines into a buffer for parallelizing
            chunk.filter(|line| {
                // If it doesn't match the pattern, move along
                if let Some(node_type) = filter_node {
                    line.starts_with(node_type)
                } else {
                    true
                }
            }).for_each(|l| buffer.push(l));

            // Parse lines
            buffer.par_drain(..).map(|line| {
                line_to_embedding(line)
                    .ok_or_else(|| PyValueError::new_err("Error parsing line"))
            }).collect_into_vec(&mut p_buffer);

            for record in p_buffer.drain(..) {
                let (node_type, node_name, emb) = record?;

                if i == 0 {
                    es = EmbeddingStore::new(num_embeddings, emb.len(), distance.to_edist());
                }

                let node_id = vocab.get_or_insert(node_type, node_name);
                let m = es.get_embedding_mut(node_id);
                if m.len() != emb.len() {
                    return Err(PyValueError::new_err("Embeddings have different sizes!"));
                }
                m.copy_from_slice(&emb);
                i += 1;
            }
        }

        let ne = NodeEmbeddings {
            vocab: Arc::new(vocab),
            embeddings: es
        };
        Ok(ne)
    }
}

fn line_to_embedding(line: String) -> Option<(String,String,Vec<f32>)> {
    let pieces:Vec<_> = line.split('\t').collect();
    if pieces.len() != 3 {
        return None
    }

    let node_type = pieces[0];
    let name = pieces[1];
    let e = pieces[2];
    let emb: Result<Vec<f32>,_> = e[1..e.len() - 1].split(',')
        .map(|wi| parse(wi.trim())).collect();

    emb.ok().map(|e| (node_type.to_string(), name.to_string(), e))
}

#[pyclass]
pub struct VocabIterator {
    vocab: Arc<Vocab>,
    cur_idx: usize
}

impl VocabIterator {
    fn new(vocab: Arc<Vocab>) -> Self {
        VocabIterator {
            vocab, cur_idx: 0
        }
    }
}

#[pymethods]
impl VocabIterator {

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyObject> {
        Python::with_gil(|py| -> Option<PyObject> {
            if slf.cur_idx < slf.vocab.len() {
                let (node_type, name) = slf.vocab.get_name(slf.cur_idx).unwrap();
                let node_type = (*node_type).clone().into_py(py);
                let name = (*name).clone().into_py(py);
                slf.cur_idx += 1;
                Some((node_type, name).into_py(py))
            } else {
                None
            }
        })
    }
}

#[pyclass]
struct NeighborhoodAligner {
    aligner: NA
}

#[pymethods]
impl NeighborhoodAligner {
    #[new]
    pub fn new(alpha: Option<f32>, max_neighbors: Option<usize>) -> Self {
        let aligner = NA::new(alpha, max_neighbors);
        NeighborhoodAligner {aligner}
    }

    pub fn align(&self, 
        embeddings: &NodeEmbeddings, 
        graph: &Graph
    ) -> NodeEmbeddings {
        
        let new_embeddings = EmbeddingStore::new(embeddings.embeddings.len(), 
                                                 embeddings.embeddings.dims(),
                                                 embeddings.embeddings.distance());
        let num_nodes = graph.nodes();
        (0..num_nodes).into_par_iter().for_each(|node| {
            let new_emb = new_embeddings.get_embedding_mut_hogwild(node);
            self.aligner.align(&(*graph.graph), &embeddings.embeddings, node, new_emb);
        });

        NodeEmbeddings {
            embeddings: new_embeddings,
            vocab: embeddings.vocab.clone()
        }
    }

    pub fn align_to_disk(
        &self, 
        path: &str,
        embeddings: &NodeEmbeddings, 
        graph: &Graph,
        chunk_size: Option<usize>
    ) -> PyResult<()> {
       
        let mut writer = EmbeddingWriter::new(path, embeddings.vocab.as_ref())
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        let cs = chunk_size.unwrap_or(10_000);
        let mut buffer = vec![vec![0.; embeddings.embeddings.dims()]; cs];
        let mut ids = Vec::with_capacity(cs);
        for chunk in &(0..graph.nodes()).chunks(cs) {
            ids.clear();
            chunk.for_each(|id| ids.push(id));

            ids.par_iter().zip(buffer.par_iter_mut()).for_each(|(node, new_emb)| {
                new_emb.iter_mut().for_each(|wi| *wi = 0f32);
                self.aligner.align(&(*graph.graph), &embeddings.embeddings, *node, new_emb);
            });

            writer.stream(ids.iter().copied().zip(buffer.iter()).take(ids.len()))?;
        }
        Ok(())
    }


}


#[pyclass]
struct Ann {
    graph: Arc<CumCSR>,
    vocab: Arc<Vocab>,
    max_steps: usize
}

#[pymethods]
impl Ann {
    #[new]
    pub fn new(graph: &Graph, max_steps: Option<usize>) -> Self {
        Ann {
            graph: graph.graph.clone(),
            vocab: graph.vocab.clone(),
            max_steps: max_steps.unwrap_or(1000),
        }

    }

    pub fn find(
        &self, 
        query: &Query,
        embeddings: &NodeEmbeddings, 
        k: usize, 
        seed: Option<u64>
    ) -> PyResult<Vec<((String, String), f32)>> {
        let query_embedding = lookup_embedding(query, embeddings)?;
        let seed = seed.unwrap_or(SEED + 10);
        let ann = algos::ann::Ann::new(k, self.max_steps + k, seed);
        let nodes = ann.find(query_embedding, &(*self.graph), &embeddings.embeddings);
        Ok(convert_node_distance(&self.vocab, nodes))
    }
}

fn lookup_embedding<'a>(
    query: &'a Query, 
    embeddings: &'a NodeEmbeddings
) -> PyResult<&'a [f32]> {
    match &query.qt {
        QueryType::Node(nt, nn) => {
            let node_id = get_node_id(embeddings.vocab.deref(), nt.clone(), nn.clone())?;
            Ok(embeddings.embeddings.get_embedding(node_id))
        },
        QueryType::Embedding(ref emb) => Ok(emb)
    }
}

enum QueryType {
    Node(String, String),
    Embedding(Vec<f32>)
}

#[pyclass]
pub struct Query {
    qt: QueryType
}

#[pymethods]
impl Query {

    #[staticmethod]
    pub fn node(
        node_type: String,
        node_name: String
    ) -> Self {
        Query { qt: QueryType::Node(node_type, node_name) }
    }

    #[staticmethod]
    pub fn embedding(
        emb: Vec<f32>
    ) -> Self {
        Query { qt: QueryType::Embedding(emb) }
    }

}

fn convert_node_distance(
    vocab: &Vocab, 
    dists: Vec<NodeDistance>
) -> Vec<((String, String), f32)> {
    dists.into_iter()
        .map(|n| {
            let (node_id, dist) = n.to_tup();
            let (node_type, name) = vocab.get_name(node_id)
                .expect("Can't find node id in graph!");
            (((*node_type).clone(), (*name).clone()), dist)
        }).collect()
}

#[pymodule]
fn cloverleaf(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Graph>()?;
    m.add_class::<Distance>()?;
    m.add_class::<GraphBuilder>()?;
    m.add_class::<EdgeType>()?;
    m.add_class::<EmbeddingPropagator>()?;
    m.add_class::<DistanceEmbedder>()?;
    m.add_class::<ClusterLPAEmbedder>()?;
    m.add_class::<SLPAEmbedder>()?;
    m.add_class::<NodeEmbeddings>()?;
    m.add_class::<VocabIterator>()?;
    m.add_class::<EPLoss>()?;
    m.add_class::<Ann>()?;
    m.add_class::<FeatureSet>()?;
    m.add_class::<FeaturePropagator>()?;
    m.add_class::<NodeEmbedder>()?;
    m.add_class::<FeatureAggregator>()?;
    m.add_class::<Query>()?;
    m.add_class::<RandomWalker>()?;
    m.add_class::<BiasedRandomWalker>()?;
    m.add_class::<NeighborhoodAligner>()?;
    Ok(())
}

