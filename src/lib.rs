mod graph;
mod algos;
mod sampler;
mod vocab;

use float_ord::FloatOrd;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use graph::{CSR,CDFGraph,CumCSR,Graph};
use vocab::Vocab;

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

#[pyclass]
struct RwrGraph {
    graph: CumCSR,
    vocab: Vocab
}

#[pymethods]
impl RwrGraph {
    #[new]
    fn new(edges: Vec<(String,String,f32)>) -> Self {
        let (graph, vocab) = build_csr(edges.into_iter());
        eprintln!("Converting to CDF format...");
        RwrGraph {
            graph: CumCSR::convert(graph),
            vocab: vocab
        }
    }

    pub fn compute(&self, name: String, alpha: f32, walks: usize, seed: Option<u64>, k: Option<usize>) -> PyResult<Vec<(String, f32)>> {
        if let Some(node_id) = self.vocab.get_node_id(name) {
            let rwr = algos::rwr::RWR {
                alpha: alpha,
                walks: walks,
                seed: seed.unwrap_or(SEED)
            };

            // Run RWR, sort the values descending by score
            let mut scores: Vec<_> = rwr.weighted(&self.graph, node_id)
                .into_iter()
                .collect();

            scores.sort_by_key(|(_k, v)| FloatOrd(-*v));

            // Convert the list to named
            let k = k.unwrap_or(scores.len());
            let ret = scores.into_iter().take(k)
                .map(|(node_id, w)| {
                    let name = self.vocab.get_name(node_id).unwrap();
                    ((*name).clone(), w)
                })
                .collect();
            Ok(ret)
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
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
}
