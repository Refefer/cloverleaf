//#[cfg(not(target_env = "msvc"))]
//use tikv_jemallocator::Jemalloc;
//
//#[cfg(not(target_env = "msvc"))]
//#[global_allocator]
//static GLOBAL: Jemalloc = Jemalloc;

pub mod graph;
pub mod algos;
mod sampler;
mod vocab;
mod embeddings;
mod bitset;
mod hogwild;
mod progress;

use std::sync::Arc;
use std::ops::Deref;
use std::fs::File;
use std::io::{Write,BufWriter,Read,BufReader,BufRead};
use std::fmt::Write as FmtWrite;

use rayon::prelude::*;
use float_ord::FloatOrd;
use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError,PyIOError,PyKeyError};

use crate::graph::{CSR,CumCSR,Graph,NodeID};
use crate::algos::rwr::{Steps,RWR};
use crate::algos::grwr::{Steps as GSteps,GuidedRWR};
use crate::algos::reweighter::{Reweighter};
use crate::algos::ep::{EmbeddingPropagation,Loss};
use crate::algos::utils::FeatureStore;
use crate::algos::ann::NodeDistance;
use crate::vocab::Vocab;
use crate::sampler::Weighted;
use crate::embeddings::{EmbeddingStore,Distance as EDist,Entity};
use crate::algos::aggregator::{WeightedAggregator,UnigramProbability,AvgAggregator,EmbeddingBuilder};

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

fn convert_scores(vocab: &Vocab, scores: impl Iterator<Item=(NodeID, f32)>, k: Option<usize>) -> Vec<((String,String), f32)> {
    let mut scores: Vec<_> = scores.collect();
    scores.sort_by_key(|(_k, v)| FloatOrd(-*v));

    // Convert the list to named
    let k = k.unwrap_or(scores.len());
    scores.into_iter().take(k)
        .map(|(node_id, w)| {
            let (node_type, name) = vocab.get_name(node_id).unwrap();
            (((*node_type).clone(), (*name).clone()), w)
        })
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
pub struct RwrGraph {
    graph: Arc<CumCSR>,
    vocab: Arc<Vocab>
}

#[pymethods]
impl RwrGraph {

    #[new]
    fn new(edges: Vec<((String,String),(String,String),f32)>) -> Self {
        let (graph, vocab) = build_csr(edges.into_iter());
        eprintln!("Converting to CDF format...");
        RwrGraph {
            graph: Arc::new(CumCSR::convert(graph)),
            vocab: Arc::new(vocab)
        }
    }

    pub fn walk(
        &self, 
        name: (String, String), 
        restarts: f32, 
        walks: usize, 
        seed: Option<u64>, 
        beta: Option<f32>,
        k: Option<usize>, 
    ) -> PyResult<Vec<((String,String), f32)>> {

        let node_id = get_node_id(self.vocab.deref(), name.0, name.1)?;
        
        let steps = if restarts >= 1. {
            Steps::Fixed(restarts as usize)
        } else if restarts > 0. {
            Steps::Probability(restarts)
        } else {
            return Err(PyValueError::new_err("Alpha must be between [0, inf)"))
        };

        let rwr = RWR {
            steps: steps,
            walks: walks,
            beta: beta.unwrap_or(0.5),
            seed: seed.unwrap_or(SEED)
        };

        let results = rwr.sample(self.graph.as_ref(), &Weighted, node_id);

        Ok(convert_scores(&self.vocab, results.into_iter(), k))
    }

    pub fn biased_walk(
        &self, 
        embeddings: &NodeEmbeddings,
        name: (String,String), 
        biased_context: (String,String),
        restarts: f32, 
        walks: usize, 
        blend: Option<f32>,
        beta: Option<f32>,
        k: Option<usize>, 
        seed: Option<u64>, 
        rerank_context: Option<(String,String)>,
    ) -> PyResult<Vec<((String,String), f32)>> {
        let node_id = get_node_id(self.vocab.deref(), name.0, name.1)?;
        let g_node_id = get_node_id(self.vocab.deref(), biased_context.0, biased_context.1)?;
        
        let steps = if restarts >= 1. {
            GSteps::Fixed(restarts as usize)
        } else if restarts > 0. {
            GSteps::Probability(restarts, (1. / restarts).ceil() as usize)
        } else {
            return Err(PyValueError::new_err("Alpha must be between [0, inf)"))
        };

        let grwr = GuidedRWR {
            steps: steps,
            walks: walks,
            alpha: blend.unwrap_or(0.5),
            beta: beta.unwrap_or(0.5),
            seed: seed.unwrap_or(SEED)
        };

        let embeddings = &embeddings.embeddings;
        let mut results = grwr.sample(self.graph.as_ref(), 
                                  &Weighted, embeddings, node_id, g_node_id);
        
        // Reweight results if requested
        if let Some(cn) = rerank_context {
            println!("Reranking...");
            let c_node_id = get_node_id(self.vocab.deref(), cn.0, cn.1)?;
            Reweighter::new(blend.unwrap_or(0.5))
                .reweight(&mut results, embeddings, c_node_id);
        }

        Ok(convert_scores(&self.vocab, results.into_iter(), k))
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

        let g = RwrGraph {
            graph: Arc::new(CumCSR::convert(csr)),
            vocab: Arc::new(vocab)
        };

        Ok(g)

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

    pub fn build_graph(&mut self) -> RwrGraph {
        let mut vocab = Vocab::new(); 
        std::mem::swap(&mut vocab, &mut self.vocab);
        let mut edges = Vec::new();
        std::mem::swap(&mut edges, &mut self.edges);
        let graph = CSR::construct_from_edges(edges);

        RwrGraph {
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


}

#[pyclass]
struct EmbeddingPropagator {
    ep: EmbeddingPropagation
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
        wd: Option<f32>,
        gamma: Option<f32>,
        seed: Option<u64>,
        max_nodes: Option<usize>,
        max_features: Option<usize>,
        indicator: Option<bool>

    ) -> Self {
        let ep = EmbeddingPropagation {
            alpha: alpha.unwrap_or(0.9),
            batch_size: batch_size.unwrap_or(50),
            dims: dims.unwrap_or(100),
            passes: passes.unwrap_or(100),
            gamma: gamma.unwrap_or(0f32),
            wd: wd.unwrap_or(0f32),
            loss: loss.map(|l|l.loss).unwrap_or(Loss::MarginLoss(1f32,1)),
            seed: seed.unwrap_or(SEED),
            max_nodes: max_nodes,
            max_features: max_features,
            indicator: indicator.unwrap_or(true)
        };

        EmbeddingPropagator{ ep }
    }

    pub fn learn_features(
        &mut self, 
        graph: &RwrGraph, 
        features: &mut FeatureSet
    ) -> NodeEmbeddings {
        features.features.fill_missing_nodes();
        let feat_embeds = self.ep.learn(graph.graph.as_ref(), &mut features.features);

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
    pub fn new(graph: &RwrGraph, namespace: Option<String>) -> Self {
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

}

#[pyclass]
pub struct FeatureEmbeddingAggregator {
    vocab: Arc<Vocab>,
    unigrams: UnigramProbability
}

impl FeatureEmbeddingAggregator {

    fn get_aggregator<'a>(
        &'a self, 
        es: &'a EmbeddingStore, 
        alpha: Option<f32>
    ) -> Box<dyn EmbeddingBuilder + Send + Sync + 'a> {
        if let Some(alpha) = alpha {
            Box::new(WeightedAggregator::new(es, &self.unigrams, alpha))
        } else {
            Box::new(AvgAggregator::new(es))
        }
    }

}

#[pymethods]
impl FeatureEmbeddingAggregator {
    #[new]
    pub fn new(fs: &FeatureSet) -> Self {
        let up = UnigramProbability::new(&fs.features);
        FeatureEmbeddingAggregator {
            unigrams: up,
            vocab: Arc::new(fs.features.clone_vocab())
        }
    }

    pub fn embed_graph(
        &self, 
        graph: &RwrGraph,
        feat_set: &FeatureSet, 
        feature_embeddings: &NodeEmbeddings,
        alpha: Option<f32>
    ) -> NodeEmbeddings {

        let num_nodes = graph.nodes();
        let es = EmbeddingStore::new(num_nodes, feature_embeddings.dims(), EDist::Cosine);
        let agg = self.get_aggregator(&feature_embeddings.embeddings, alpha);
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
        alpha: Option<f32>,
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

        let mut embedding = vec![0.; feature_embeddings.embeddings.dims()];
        let agg = self.get_aggregator(&feature_embeddings.embeddings, alpha);
        agg.construct(&ids, &mut embedding);
        Ok(embedding)
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        let f = File::create(path)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        let mut bw = BufWriter::new(f);
        for (node, p_wi) in self.unigrams.iter().enumerate() {
            if let Some((f_node_type, f_name)) = self.vocab.get_name(node) {
                writeln!(&mut bw, "{}\t{}\t{}", f_node_type, f_name, *p_wi as f64)
                    .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;
            } else {
                // We've moved to node individual embeddings
                break
            }
        }
        Ok(())
    }

    #[staticmethod]
    pub fn load(path: String) -> PyResult<Self> {
        let f = File::open(path)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        let br = BufReader::new(f);
        let mut vocab = Vocab::new();
        let mut p_w = Vec::new();
        for line in br.lines() {
            let line = line.unwrap();
            let pieces: Vec<_> = line.split('\t').collect();
            if pieces.len() != 3 {
                return Err(PyValueError::new_err("Malformed feature line! Need node_type<TAB>name<TAB>weight ..."))
            }
            let p_wi = pieces[2].parse::<f64>()
                .map_err(|e| PyValueError::new_err(format!("Tried to parse weight and failed:{:?}", line)))?;
            let node_id = vocab.get_or_insert(pieces[0].into(), pieces[1].into());
            if node_id < p_w.len() {
                return Err(PyValueError::new_err(format!("Duplicate feature found:{} {}", pieces[0], pieces[1])))
            }
            p_w.push(p_wi as f32);
        }

        Ok(FeatureEmbeddingAggregator { 
            vocab: Arc::new(vocab),
            unigrams: UnigramProbability::from_vec(p_w) 
        })
    }


}

fn count_lines(path: &str) -> std::io::Result<usize> {
    let f = File::open(path)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

    let br = BufReader::new(f);
    let mut count = 0;
    for line in br.lines() {
        line?;
        count += 1;
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

    pub fn learn(&self, graph: &RwrGraph) -> NodeEmbeddings {
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

    pub fn learn(&self, graph: &RwrGraph) -> NodeEmbeddings {
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

    pub fn learn(&self, graph: &RwrGraph) -> NodeEmbeddings {
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
    pub fn new(graph: &RwrGraph, dims: usize, distance: Distance) -> Self {
        let dist = distance.to_edist();

        let es = EmbeddingStore::new(graph.graph.len(), dims, dist);
        NodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings: es
        }
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

    pub fn nearest_neighbor(&self, emb: Vec<f32>, k: usize) -> Vec<((String,String), f32)> {
        let dists = self.embeddings.nearest_neighbor(&Entity::Embedding(&emb), k);
        convert_node_distance(&self.vocab, dists)
    }

    pub fn dims(&self) -> usize {
        self.embeddings.dims()
    }
    
    pub fn save(&self, path: &str) -> PyResult<()> {
        let f = File::create(path)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        let mut bw = BufWriter::new(f);
        let mut s = String::new();
        for node_id in 0..self.vocab.len() {
            let (node_type, name) = self.vocab.get_name(node_id)
                .expect("Programming error!");

            // Build the embedding to string
            s.clear();
            for (idx, wi) in self.embeddings.get_embedding(node_id).iter().enumerate() {
                if idx > 0 {
                    s.push_str(",");
                }
                write!(&mut s, "{}", wi).expect("Shouldn't error writing to a string");
            }

            writeln!(&mut bw, "{}\t{}\t[{}]", node_type, name, s)
                .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;
        }
        Ok(())
    }

    #[staticmethod]
    pub fn load(path: &str, distance: Distance) -> PyResult<Self> {
        let num_embeddings = count_lines(path)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        let f = File::open(path)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        let br = BufReader::new(f);
        let mut vocab = Vocab::new();
        
        // Place holder
        let mut es = EmbeddingStore::new(0, 0, EDist::Cosine);
        for (i, line) in br.lines().enumerate() {
            let line = line.unwrap();
            let (node_type, node_name, emb) = line_to_embedding(line)
                .ok_or_else(|| PyValueError::new_err("Error parsing line"))?;

            if i == 0 {
                es = EmbeddingStore::new(num_embeddings, emb.len(), distance.to_edist());
            }

            let node_id = vocab.get_or_insert(node_type, node_name);
            let m = es.get_embedding_mut(node_id);
            if m.len() != emb.len() {
                return Err(PyValueError::new_err("Embeddings have different sizes!"));
            }
            m.copy_from_slice(&emb);
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
        .map(|wi| wi.trim().parse()).collect();

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
struct Ann {
    graph: Arc<CumCSR>,
    vocab: Arc<Vocab>,
    max_steps: usize
}

#[pymethods]
impl Ann {
    #[new]
    pub fn new(graph: &RwrGraph, max_steps: Option<usize>) -> Self {
        Ann {
            graph: graph.graph.clone(),
            vocab: graph.vocab.clone(),
            max_steps: max_steps.unwrap_or(1000),
        }

    }

    pub fn find(
        &self, 
        query: Vec<f32>,
        embeddings: &NodeEmbeddings, 
        k: usize, 
        seed: Option<u64>
    ) -> Vec<((String, String), f32)> {
        let seed = seed.unwrap_or(SEED + 10);
        let ann = algos::ann::Ann::new(k, self.max_steps + k, seed);
        let nodes = ann.find(query.as_slice(), &(*self.graph), &embeddings.embeddings);
        convert_node_distance(&self.vocab, nodes)
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
    m.add_class::<RwrGraph>()?;
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
    m.add_class::<FeatureEmbeddingAggregator>()?;
    Ok(())
}

