mod graph;
mod algos;
mod sampler;
mod vocab;
mod embeddings;

use float_ord::FloatOrd;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::graph::{CSR,CDFGraph,CumCSR,Graph,NodeID};
use crate::algos::rwr::{Steps,RWR};
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
    ALT
}

#[pyclass]
struct RwrGraph {
    graph: CumCSR,
    vocab: Vocab,
    embeddings: Option<EmbeddingStore>
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
        alpha: f32, 
        walks: usize, 
        seed: Option<u64>, 
        k: Option<usize>, 
        context: Option<String>,
        blend: Option<f32>
    ) -> PyResult<Vec<(String, f32)>> {
        if let Some(node_id) = self.vocab.get_node_id(name) {
            let steps = if alpha >= 1. {
                Steps::Fixed(alpha as usize)
            } else if alpha > 0. {
                Steps::Probability(alpha)
            } else {
                return Err(PyValueError::new_err("Alpha must be between [0, inf)"))
            };

            let rwr = RWR {
                steps: steps,
                walks: walks,
                seed: seed.unwrap_or(SEED)
            };

            let mut results = rwr.sample(&self.graph, &Weighted, node_id);
            // Reweight results if requested
            if let Some(cn) = context {
                if let Some(c_node_id) = self.vocab.get_node_id(cn) {
                    if let Some(es) = &self.embeddings {
                        Reweighter::new(blend.unwrap_or(0.5))
                            .reweight(&mut results, es, c_node_id);
                    } else {
                        return Err(PyValueError::new_err("No embeddings added for graph!"))
                    }
                } else {
                    return Err(PyValueError::new_err("Context Node does not exist!"))
                }
            }
            Ok(convert_scores(&self.vocab, results.into_iter(), k))
        } else {
            Err(PyValueError::new_err("Node does not exist!"))
        }
    }

    pub fn initialize_embeddings(&mut self, dims: usize, distance: Distance) -> PyResult<()>{
        let dist = match distance {
            Distance::Cosine => EDist::Cosine,
            Distance::Euclidean => EDist::Euclidean,
            Distance::ALT => EDist::ALT
        };
        self.embeddings = Some(EmbeddingStore::new(self.graph.len(), dims, dist));
        Ok(())
    }

    pub fn set_embedding(&mut self, name: String, embedding: Vec<f32>) -> PyResult<()> {
        if let Some(node_id) = self.vocab.get_node_id(name) {
            if let Some(es) = &mut self.embeddings {
                es.set_embedding(node_id, &embedding);
                Ok(())
            } else {
                Err(PyValueError::new_err("Embedding store wasn't initialized!"))
            }
        } else {
            Err(PyValueError::new_err("Node does not exist!"))
        }
    }

    pub fn nodes(&self) -> usize {
        self.graph.len()
    }

    pub fn edges(&self) -> usize {
        self.graph.len()
    }

}

#[pymodule]
fn cloverleaf(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    //m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<RwrGraph>()?;
    m.add_class::<Distance>()?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
}
