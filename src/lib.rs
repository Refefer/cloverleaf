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

use std::sync::Arc;
use std::ops::Deref;

use float_ord::FloatOrd;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::graph::{CSR,CumCSR,Graph,NodeID};
use crate::algos::rwr::{Steps,RWR};
use crate::algos::grwr::{Steps as GSteps,GuidedRWR};
use crate::algos::reweighter::{Reweighter};
use crate::algos::ep::{FeatureStore,EmbeddingPropagation};
use crate::vocab::Vocab;
use crate::sampler::Weighted;
use crate::embeddings::{EmbeddingStore,Distance as EDist};

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
        Err(PyValueError::new_err(format!(" Node '{}:{}' does not exist!", node_type, node)))
    }
}


#[pyclass]
#[derive(Clone)]
enum Distance {
    Cosine,
    Euclidean,
    ALT,
    Jaccard,
    Hamming
}

#[pyclass]
struct RwrGraph {
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

}

#[pyclass]
#[derive(Clone)]
enum EdgeType {
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
struct EmbeddingPropagator {
    vocab: Arc<Vocab>,
    features: FeatureStore
}

#[pymethods]
impl EmbeddingPropagator {
    #[new]
    pub fn new(graph: &RwrGraph) -> Self {
        EmbeddingPropagator {
            features: FeatureStore::new(graph.graph.len()),
            vocab: graph.vocab.clone()
        }
    }


    pub fn add_features(&mut self, node: (String,String), features: Vec<String>) -> PyResult<()> {
        let node_id = get_node_id(self.vocab.deref(), node.0, node.1)?;
        self.features.set_features(node_id, features);
        Ok(())
    }

    pub fn learn(
        &mut self, 
        graph: &mut RwrGraph, 
        alpha: f32, 
        gamma: f32, 
        batch_size: usize, 
        dims: usize,
        passes: usize,
        seed: Option<u64>
    ) -> NodeEmbeddings{
        let ep = EmbeddingPropagation {
            alpha,
            gamma,
            batch_size,
            dims,
            passes,
            seed: seed.unwrap_or(SEED)
        };

        self.features.fill_missing_nodes();
        let (embeddings, _feat_embeds) = ep.learn(graph.graph.as_ref(), &mut self.features);
        NodeEmbeddings {
            vocab: self.vocab.clone(),
            embeddings
        }
    }

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
struct NodeEmbeddings {
    vocab: Arc<Vocab>,
    embeddings: EmbeddingStore
}

#[pymethods]
impl NodeEmbeddings {
    #[new]
    pub fn new(graph: &RwrGraph, dims: usize, distance: Distance) -> Self {
        let dist = match distance {
            Distance::Cosine => EDist::Cosine,
            Distance::Euclidean => EDist::Euclidean,
            Distance::ALT => EDist::ALT,
            Distance::Hamming => EDist::Hamming,
            Distance::Jaccard => EDist::Jaccard
        };

        let es = EmbeddingStore::new(graph.graph.len(), dims, dist);
        NodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings: es
        }
    }

    fn get_node_id(&self, node_type: String, node: String) -> PyResult<NodeID> {
        if let Some(node_id) = self.vocab.get_node_id(node_type.clone(), node.clone()) {
            Ok(node_id)
        } else {
            Err(PyValueError::new_err(format!(" Node '{}:{}' does not exist!", node_type, node)))
        }
    }

    pub fn get_embedding(&mut self, node: (String,String)) -> PyResult<Vec<f32>> {
        let node_id = get_node_id(self.vocab.deref(), node.0, node.1)?;
        Ok(self.embeddings.get_embedding(node_id).to_vec())
    }

    pub fn set_embedding(&mut self, node: (String,String), embedding: Vec<f32>) -> PyResult<()> {
        let node_id = get_node_id(self.vocab.deref(), node.0, node.1)?;
        let mut es = &mut self.embeddings;
        es.set_embedding(node_id, &embedding);
        Ok(())
    }

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
    Ok(())
}

