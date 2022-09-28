pub mod graph;
pub mod algos;
mod sampler;
mod vocab;
mod embeddings;
mod bitset;

use float_ord::FloatOrd;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::graph::{CSR,CDFGraph,CumCSR,Graph,NodeID};
use crate::algos::rwr::{Steps,RWR};
use crate::algos::grwr::{Steps as GSteps,GuidedRWR};
use crate::algos::reweighter::{Reweighter};
use crate::vocab::Vocab;
use crate::sampler::Weighted;
use crate::embeddings::{EmbeddingStore,Distance as EDist};

const SEED: u64 = 20222022;

fn build_csr(edges: impl Iterator<Item=(String,String,f32)>) -> (CSR, Vocab) {
    
    // Convert to NodeIDs
    let mut vocab = Vocab::new();
    eprintln!("Constructing vocab...");
    let edges: Vec<_> = edges.map(|(f_n, t_n, w)| {
        let f_id = vocab.get_or_insert(f_n);
        let t_id = vocab.get_or_insert(t_n);
        (f_id, t_id, w)
    }).collect();

    eprintln!("Constructing CSR...");
    let csr = CSR::construct_from_edges(edges);
    (csr, vocab)
}

fn convert_scores(vocab: &Vocab, scores: impl Iterator<Item=(NodeID, f32)>, k: Option<usize>) -> Vec<(String, f32)> {
    let mut scores: Vec<_> = scores.collect();
    scores.sort_by_key(|(_k, v)| FloatOrd(-*v));

    // Convert the list to named
    let k = k.unwrap_or(scores.len());
    scores.into_iter().take(k)
        .map(|(node_id, w)| {
            let name = vocab.get_name(node_id).unwrap();
            ((*name).clone(), w)
        })
        .collect()
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
    graph: CumCSR,
    vocab: Vocab,
    embeddings: Option<EmbeddingStore>
}

impl RwrGraph {
    fn get_node_id(&self, node: String) -> PyResult<NodeID> {
        if let Some(node_id) = self.vocab.get_node_id(node.clone()) {
            Ok(node_id)
        } else {
            Err(PyValueError::new_err(format!(" Node '{}' does not exist!", node)))
        }
    }

    fn get_embeddings(&self) -> PyResult<&EmbeddingStore> {
        if let Some(es) = &self.embeddings {
            Ok(es)
        } else {
            Err(PyValueError::new_err("Embedding store wasn't initialized!"))
        }
    }

    fn get_embeddings_mut(&mut self) -> PyResult<&mut EmbeddingStore> {
        if let Some(es) = &mut self.embeddings {
            Ok(es)
        } else {
            Err(PyValueError::new_err("Embedding store wasn't initialized!"))
        }
    }

}


#[pymethods]
impl RwrGraph {

    #[new]
    fn new(edges: Vec<(String,String,f32)>) -> Self {
        let (graph, vocab) = build_csr(edges.into_iter());
        eprintln!("Converting to CDF format...");
        RwrGraph {
            graph: CumCSR::convert(graph),
            vocab: vocab,
            embeddings: None
        }
    }

    pub fn compute(
        &self, 
        name: String, 
        restarts: f32, 
        walks: usize, 
        seed: Option<u64>, 
        k: Option<usize>, 
        guided_context: Option<String>,
        rerank_context: Option<String>,
        blend: Option<f32>,
        beta: Option<f32>
    ) -> PyResult<Vec<(String, f32)>> {
        let node_id = self.get_node_id(name)?;

        
        let mut results = if let Some(guided_node) = guided_context {
            let g_node_id = self.get_node_id(guided_node)?;
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
            let embeddings = self.get_embeddings()?;
            grwr.sample(&self.graph, &Weighted, embeddings, node_id, g_node_id)
        } else {
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

            rwr.sample(&self.graph, &Weighted, node_id)
        };

        // Reweight results if requested
        if let Some(cn) = rerank_context {
            println!("Reranking...");
            let c_node_id = self.get_node_id(cn)?;
            let embeddings = self.get_embeddings()?;
            Reweighter::new(blend.unwrap_or(0.5))
                .reweight(&mut results, embeddings, c_node_id);
        }

        Ok(convert_scores(&self.vocab, results.into_iter(), k))
    }

    pub fn initialize_embeddings(&mut self, dims: usize, distance: Distance) -> PyResult<()>{
        let dist = match distance {
            Distance::Cosine => EDist::Cosine,
            Distance::Euclidean => EDist::Euclidean,
            Distance::ALT => EDist::ALT,
            Distance::Hamming => EDist::Hamming,
            Distance::Jaccard => EDist::Jaccard
        };
        self.embeddings = Some(EmbeddingStore::new(self.graph.len(), dims, dist));
        Ok(())
    }

    pub fn create_distance_embeddings(&mut self, landmarks: usize, seed: Option<u64>) -> PyResult<()> {
        let ls = if let Some(seed) = seed {
            algos::dist::LandmarkSelection::Random(seed)
        } else {
            algos::dist::LandmarkSelection::Degree
        };
        let es = crate::algos::dist::construct_walk_distances(&self.graph, landmarks, ls);
        self.embeddings = Some(es);
        Ok(())
    }

    pub fn create_cluster_embeddings(&mut self, k: usize, passes: usize, seed: Option<u64>) -> PyResult<()> {
        let seed = seed.unwrap_or(SEED);
        let es = crate::algos::lpa::construct_lpa_embedding(&self.graph, k, passes, seed);
        self.embeddings = Some(es);
        Ok(())
    }

    pub fn create_slpa_embeddings(&mut self, k: usize, threshold: f32, seed: Option<u64>) -> PyResult<()> {
        let seed = seed.unwrap_or(SEED);
        let es = crate::algos::slpa::construct_slpa_embedding(&self.graph, k, threshold, seed);
        self.embeddings = Some(es);
        Ok(())
    }

    pub fn contains_node(&self, name: String) -> bool {
        self.vocab.get_node_id(name).is_some()
    }

    pub fn get_embedding(&mut self, name: String) -> PyResult<Vec<f32>> {
        let node_id = self.get_node_id(name)?;
        let es = self.get_embeddings()?;
        Ok(es.get_embedding(node_id).to_vec())
    }

    pub fn set_embedding(&mut self, name: String, embedding: Vec<f32>) -> PyResult<()> {
        let node_id = self.get_node_id(name)?;
        let mut es = self.get_embeddings_mut()?;
        es.set_embedding(node_id, &embedding);
        Ok(())
    }

    pub fn nodes(&self) -> usize {
        self.graph.len()
    }

    pub fn edges(&self) -> usize {
        self.graph.edges()
    }

    pub fn get_edges(&self, name: String) -> PyResult<(Vec<String>, Vec<f32>)> {
        let node_id = self.get_node_id(name)?;
        let (edges, weights) = self.graph.get_edges(node_id);
        let names = edges.into_iter()
            .map(|node_id| (*self.vocab.get_name(*node_id).unwrap()).clone())
            .collect();
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

    pub fn add_edge(&mut self, from_node: String, to_node: String, weight: f32, node_type: EdgeType) {
        let f_id = self.vocab.get_or_insert(from_node);
        let t_id = self.vocab.get_or_insert(to_node);
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
            graph: CumCSR::convert(graph),
            vocab: vocab,
            embeddings: None
        }
    }

}


#[pymodule]
fn cloverleaf(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RwrGraph>()?;
    m.add_class::<Distance>()?;
    m.add_class::<GraphBuilder>()?;
    m.add_class::<EdgeType>()?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
}
