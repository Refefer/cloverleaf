//! This is the main interface for Cloverleaf
//! It has tight coupling to python, specifically as the lingua franca of the machine learning
//! world.  Consequently, this coupling has a couple of nuances that limit cloverleaf's ability
//! as a standalone module but that's ok :)

/// Main interface for defining graphs
pub mod graph;

/// We define all the algorithms within this module
pub mod algos;

/// How can we efficiently sample from the graph?
mod sampler;

/// Maps node types, node names to internal IDs and back
mod vocab;

/// Where we store embeddings.  These are both node and feature embeddings
mod embeddings;

/// Simple bitset
mod bitset;

/// This interface allows us to update embeddings (and other structures) in multiple threads
/// without having to gain exclusive write access.  Do _not_ clone hogwild structures as they
/// will still point to the underlying data
mod hogwild;

/// Who doesn't like progress bars?
mod progress;

/// Mapping from nodes -> features
mod feature_store;

/// Beginnings of refactoring out IO operations for efficient loading/writing of different data
/// structures
mod io;

use std::sync::Arc;
use std::ops::Deref;
use std::fs::File;
use std::io::{Write,BufWriter,BufReader,BufRead};

use rayon::prelude::*;
use float_ord::FloatOrd;
use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError,PyIOError,PyKeyError,PyIndexError};
use itertools::Itertools;
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rand_distr::Uniform;

use crate::graph::{CSR,CumCSR,Graph as CGraph,NodeID,CDFtoP};
use crate::vocab::Vocab;
use crate::sampler::{Weighted,Unweighted};
use crate::embeddings::{EmbeddingStore,Distance as EDist,Entity};
use crate::feature_store::FeatureStore;
use crate::io::{EmbeddingWriter,EmbeddingReader,GraphReader,open_file_for_reading,open_file_for_writing};

use crate::algos::rwr::{Steps,RWR,ppr_estimate,rollout};
use crate::algos::grwr::{Steps as GSteps,GuidedRWR};
use crate::algos::reweighter::{Reweighter};
use crate::algos::ep::{EmbeddingPropagation,LossWeighting as EPLW};
use crate::algos::ep::loss::Loss;
use crate::algos::ep::model::{AveragedFeatureModel,AttentionFeatureModel};
use crate::algos::ep::attention::{AttentionType,MultiHeadedAttention};
use crate::algos::graph_ann::NodeDistance;
use crate::algos::aggregator::{WeightedAggregator,UnigramProbability,AvgAggregator,AttentionAggregator, EmbeddingBuilder};
use crate::algos::feat_propagation::propagate_features;
use crate::algos::alignment::{NeighborhoodAligner as NA};
use crate::algos::smci::SupervisedMCIteration;
use crate::algos::pprrank::{PprRank, Loss as PprLoss};
use crate::algos::ann::Ann;
use crate::algos::pprembed::PPREmbed;
use crate::algos::instantembedding::{InstantEmbeddings as IE,Estimator};
use crate::algos::lsr::{LSR as ALSR};
use crate::algos::connected::find_connected_components;

/// Defines a constant seed for use when a seed is not provided.  This is specifically hardcoded to
/// allow for deterministic performance across all algorithms using any stochasticity.
const SEED: u64 = 20222022;

/// Simplifies a lot of the type signatures
type FQNode = (String, String);

/// Maps an iterator of node ids and scores back to their pretty names with optional top K and
/// filtering by node types.
fn convert_scores(
    vocab: &Vocab, 
    scores: impl Iterator<Item=(NodeID, f32)>, 
    k: Option<usize>,
    filtered_node_type: Option<String>
) -> Vec<(FQNode, f32)> {
    let mut scores: Vec<_> = scores.collect();
    scores.sort_by_key(|(_k, v)| FloatOrd(-*v));

    // Convert the list to named
    let k = k.unwrap_or(scores.len());
    scores.into_iter()
        .map(|(node_id, w)| {
            let (node_type, name) = vocab.get_name(node_id).unwrap();
            (((*node_type).clone(), name.to_string()), w)
        })
        .filter(|((node_type, _node_name), _w)| {
            filtered_node_type.as_ref().map(|nt| nt == node_type).unwrap_or(true)
        })
        .take(k)
        .collect()
}

fn convert_node_id_to_fqn(
    vocab: &Vocab,
    node_id: NodeID
) -> FQNode {
    let (node_type, name) = vocab.get_name(node_id).unwrap();
    ((*node_type).clone(), name.to_string())
}


/// Convenience method for getting an internal node id from pretty name
fn get_node_id(vocab: &Vocab, node_type: String, node: String) -> PyResult<NodeID> {
    if let Some(node_id) = vocab.get_node_id(node_type.clone(), node.clone()) {
        Ok(node_id)
    } else {
        Err(PyKeyError::new_err(format!(" Node '{}:{}' does not exist!", node_type, node)))
    }
}


#[derive(Clone)]
enum QueryType {
    Node(String,String),
    Embedding(Vec<f32>)
}

/// Type of Query to issue: a direct node lookup or an embedding
#[pyclass]
#[derive(Clone)]
pub struct Query {
    qt: QueryType
}

#[pymethods]
impl Query {

    ///    Creates a query using a node type and name as lookup.
    ///
    ///    Parameters
    ///    ----------
    ///    node_type : str 
    ///        The type of node.
    ///    node_name : str
    ///        The name of the node
    ///
    ///    Returns
    ///    -------
    ///    Query
    #[staticmethod]
    pub fn node(
        node_type: String,
        node_name: String
    ) -> Self {
        Query { qt: QueryType::Node(node_type, node_name) }
    }

    ///    Creates a query using a provided embedding.
    ///
    ///    Parameters
    ///    ----------
    ///    emb :  List[float]
    ///        A list of floating point numbers to lookup.
    ///
    ///    Returns
    ///    -------
    ///    Query

    #[staticmethod]
    pub fn embedding(
        emb: Vec<f32>
    ) -> Self {
        Query { qt: QueryType::Embedding(emb) }
    }

}


///    Contains the set of distance metrics used by NodeEmbeddings, typically application
///    dependent.
///
#[pyclass]
#[derive(Clone)]
pub enum Distance {
    /// Uses Cosine distance - useful for general embedding problems.
    Cosine,
    
    /// Euclidean distance
    Euclidean,

    /// Simple un-normalized dot products.
    Dot,

    /// Landmark triangulation distance, useful for Distance embeddingsji
    ALT,

    /// Computes the jaccard between embeddings, treating each value as a discrete class
    Jaccard,

    /// Computes the hamming distance between embeddings, treating each value as a discrete class
    Hamming
}

impl Distance {
    fn to_edist(&self) -> EDist {
        match self {
            Distance::Cosine => EDist::Cosine,
            Distance::Dot => EDist::Dot,
            Distance::Euclidean => EDist::Euclidean,
            Distance::ALT => EDist::ALT,
            Distance::Hamming => EDist::Hamming,
            Distance::Jaccard => EDist::Jaccard
        }
    }

    fn from_edist(dist: EDist) -> Distance {
        match dist {
            EDist::Cosine => Distance::Cosine,
            EDist::Dot => Distance::Dot,
            EDist::Euclidean => Distance::Euclidean,
            EDist::ALT => Distance::ALT,
            EDist::Hamming => Distance::Hamming,
            EDist::Jaccard => Distance::Jaccard  
        }
    }

}

#[pymethods]
impl Distance {
    pub fn compute(
        &self,
        e1: Vec<f32>,
        e2: Vec<f32>
    ) -> f32 {
        self.to_edist().compute(e1.as_slice(), e2.as_slice())
    }
}

/// 
/// Core Graph library in Cloverleaf.
///
/// Graphs contain a list of nodes, defined by their type and their name, and a list of directional edges
/// and corresponding weights that describe node connections.  
///
/// Graphs are encoded using Compressed Sparse Row Format to minimize
/// memory costs and allow for large graphs to be constructed on commodity systems.  Further, edge
/// weights are encoded using CDF format to optimizes certain access patterns, such as weighted
/// random walks.The downside is this makes graphs immutable: there are no update or delete methods available 
/// for defined graphs.
#[pyclass]
pub struct Graph {
    graph: Arc<CumCSR>,
    vocab: Arc<Vocab>
}

#[pymethods]
impl Graph {

    ///    Checks if a node is defined within the graph.
    ///
    ///    Parameters
    ///    ----------
    ///    name :  FQNode
    ///        A tuple containing the (node_type, node_name) to lookup.
    ///
    ///    Returns
    ///    -------
    ///    bool
    ///     Returns True if the node is defined in the graph, False otherwise
    pub fn contains_node(&self, name: FQNode) -> bool {
        get_node_id(self.vocab.deref(), name.0, name.1).is_ok()
    }

    ///    Returns the number of nodes that are defined in the graph
    ///
    ///    Parameters
    ///    ----------
    ///
    ///    Returns
    ///    -------
    ///    Int
    pub fn nodes(&self) -> usize {
        self.graph.len()
    }

    ///    Returns the number of edges defined in the graph.
    ///
    ///    Parameters
    ///    ----------
    ///
    ///    Returns
    ///    -------
    ///    Int
    ///     
    pub fn edges(&self) -> usize {
        self.graph.edges()
    }

    ///    Returns the set of outbound nodes and corresponding edge weights for a given node in the Graph.
    ///
    ///    Parameters
    ///    ----------
    ///    node: FQNode
    ///        A tuple containing the (node_type, node_name) to lookup.
    ///
    ///    normalized: Bool - optional:  
    ///        If provided, returns transition probability.  If omitted, returns in CDF format
    ///        which allows for fast weighted random sampling.
    ///
    ///    Returns
    ///    -------
    ///    (List[(str, str)], List[float])
    ///     Set of edges and the corresponding set of weights.
    ///
    ///     Throws a KeyError if the node doesn't exist in the graph
    ///     
    pub fn get_edges(&self, node: FQNode, normalized: Option<bool>) -> PyResult<(Vec<FQNode>, Vec<f32>)> {
        let node_id = get_node_id(self.vocab.deref(), node.0, node.1)?;
        let (edges, weights) = self.graph.get_edges(node_id);
        let names = edges.into_iter()
            .map(|node_id| {
                let (nt, n) = self.vocab.get_name(*node_id).unwrap();
                ((*nt).clone(), n.to_string())
            }).collect();

        if let Some(true) = normalized {
            Ok((names, CDFtoP::new(weights).collect()))
        } else {
            Ok((names, weights.to_vec()))
        }
    }

    ///    Returns an interator to the nodes defined in the graph
    ///
    ///    Parameters
    ///    ----------
    ///
    ///    Returns
    ///    -------
    ///    Iterator[(str, str)]
    ///         An iterator emitting node types and node names defined in the graph.
    ///     
    pub fn vocab(&self) -> VocabIterator {
        VocabIterator::new(self.vocab.clone())
    }

    ///    Saves a graph to disk at the provided path.
    ///
    ///    Parameters
    ///    ----------
    ///    path : String
    ///     Where to save the the graph.
    ///
    ///    Returns
    ///    -------
    ///     
    pub fn save(&self, path: &str, comp_level: Option<u32>) -> PyResult<()> {
        let mut bw = open_file_for_writing(path, comp_level)?;
        for node in 0..self.graph.len() {
            let (f_node_type, f_name) = self.vocab.get_name(node)
                .expect("Programming error!");

            let (edges, weights) = self.graph.get_edges(node);
            for (out_node, weight) in edges.iter().zip(CDFtoP::new(weights)) {

                let (t_node_type, t_name) = self.vocab.get_name(*out_node)
                    .expect("Programming error!");
                writeln!(&mut bw, "{}\t{}\t{}\t{}\t{}", f_node_type, f_name, t_node_type, t_name, weight)
                    .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;
            }
        }
        Ok(())
    }

    /// Returns the number of nodes in the graph
    pub fn __len__(&self) -> PyResult<usize> {
        Ok(self.nodes())
    }
    
    /// Simple represetnation of the Graph
    pub fn __repr__(&self) -> String {
        format!("Graph<Nodes={}, Edges={}>", self.graph.len(), self.graph.edges())
    }

    #[staticmethod]
    ///    Loads a graph from disk
    ///
    ///    Parameters
    ///    ----------
    ///    path : str
    ///        Path to load graph from, in graph format.
    ///    
    ///    edge_type : EdgeType
    ///        EdgeType to use, either Directed or Undirected
    ///    
    ///    Returns
    ///    -------
    ///    Self - Can throw exception
    ///        
    pub fn load(
        py: Python<'_>,
        path: &str, 
        edge_type: EdgeType, 
        chunk_size: Option<usize>,
        skip_rows: Option<usize>,
        weighted: Option<bool>
        ) -> PyResult<Self> {

        py.allow_threads(move || {
            let (vocab, csr) = GraphReader::load(
                path, 
                edge_type, 
                chunk_size.unwrap_or(1),
                skip_rows.unwrap_or(0),
                weighted.unwrap_or(true)
            )?;

            let g = Graph {
                graph: Arc::new(csr),
                vocab: Arc::new(vocab)
            };

            Ok(g)
        })

    }


}

/// Basic RP3b walker
#[pyclass]
#[derive(Clone)]
struct RandomWalker {
    restarts: f32,
    walks: usize,
    beta: Option<f32>
}

#[pymethods]
impl RandomWalker {

    ///    Instantiates a random walker instance which can perform random walks on a graph.
    ///    
    ///    Parameters
    ///    ----------
    ///    restarts : Float
    ///        If restarts is ~ (0,1), performs probabalistic termination (ie. pagewalk).  If
    ///        restarts is [1,inf], performs fixed length walks (ie. rp3b, pixie).
    ///    
    ///    walks : Int
    ///        Number of random walks to perform.  The higher the number, the higher the fidelity
    ///        of local neighborhood at the expense of more compute.  100_000 is usually a good
    ///        number to start with.
    ///    
    ///    beta : Float - Optional
    ///        If provided, beta ~ [0,1] specifes how much to discount node impact as a function of its degree.
    ///        Higher betas discount popular nodes more, biasing toward rarer nodes.  Lower betas
    ///        reinforce popular nodes more.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    #[new]
    fn new(restarts: f32, walks: usize, beta: Option<f32>) -> Self {
        RandomWalker { restarts, walks, beta }
    }

    /// Simple representation of the RandomWalker
    pub fn __repr__(&self) -> String {
        format!("RandomWalker<restarts={}, walks={}, beta={:?}>", self.restarts, self.walks, self.beta)
    }

    ///    Performs a random walk on a graph, returning a list of nodes and their approxmiate
    ///    scores where higher scores indicate a higher likelihood to terminate on those nodes.
    ///    
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph to perform random walks on
    ///    
    ///    node : FQNode
    ///        Fully qualified node: (NodeType, NodeName)
    ///    
    ///    seed : Int - Optional
    ///        If provided, sets the random seed.  Otherwise, uses a global fixed seed.
    ///    
    ///    k : Int - Optional
    ///        If provided, truncates the list to the top K.
    ///    
    ///    filter_type : String - Optional
    ///        If provided, only returns nodes that match the provided node type.
    ///    
    ///    weighted : Bool - Optional
    ///        Whether to perform a weighted random walk.  Default is True.
    ///    
    ///    Returns
    ///    -------
    ///    List[(FQNode, Float)] - Can throw exception
    ///        List of fully qualified nodes and their fractional scores.
    ///
    pub fn walk(
        &self, 
        graph: &Graph,
        node: FQNode, 
        seed: Option<u64>, 
        k: Option<usize>, 
        filter_type: Option<String>,
        single_threaded: Option<bool>,
        weighted: Option<bool>
    ) -> PyResult<Vec<(FQNode, f32)>> {

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
            single_threaded: single_threaded.unwrap_or(false),
            seed: seed.unwrap_or(SEED)
        };

        let results = if weighted.unwrap_or(true) {
            rwr.sample_bfs(graph.graph.as_ref(), node_id)
        } else {
            rwr.sample(graph.graph.as_ref(), &Unweighted, node_id)
        };

        Ok(convert_scores(&graph.vocab, results.into_iter(), k, filter_type))
    }

}

/// Rp3b walker with the ability to bias walks according to a provided embedding set.
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
    ///    Creates a BiasedRandomWalker instance.
    ///
    ///    BiasedRandomWalkers perform random walks while allowing external embeddings to influence
    ///    the direction a random walker takes.  When two nodes have a closer distance, the
    ///    randomwalker will reweight scores to explore in that direction more often.  This is
    ///    helpful when wanting to perform a random walk but also have it focus on areas compatible
    ///    with embeddings - for example, random walks between queries and products, where the
    ///    embeddings represent user preferences.
    ///    
    ///    Parameters
    ///    ----------
    ///    restarts : Float
    ///        If restarts is ~ (0,1), performs probabalistic termination (ie. pagewalk).  If
    ///        restarts is [1,inf], performs fixed length walks (ie. rp3b, pixie).
    ///    
    ///    walks : Int
    ///        Number of random walks to perform.  The higher the number, the higher the fidelity
    ///        of local neighborhood at the expense of more compute.  100_000 is usually a good
    ///        number to start with.
    ///    
    ///    beta : Float - Optional
    ///        If provided, beta ~ [0,1] specifes how much to discount node impact as a function of its degree.
    ///        Higher betas discount popular nodes more, biasing toward rarer nodes.  Lower betas
    ///        reinforce popular nodes more.
    ///    
    ///    blend : Float - Optional
    ///        If provided, determines how much the embedding influences the direction of the
    ///        random walk
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    fn new(restarts: f32, walks: usize, beta: Option<f32>, blend: Option<f32>) -> Self {
        BiasedRandomWalker { restarts, walks, beta, blend }
    }

    /// Simple representation of the BiasedRandomWalker
    pub fn __repr__(&self) -> String {
        format!("BiasedRandomWalker<restarts={}, walks={}, beta={:?}, blend={:?}>", self.restarts, self.walks, self.beta, self.blend)
    }
 
    ///    Performs the random walk with both starting node and bias context.  Further, a rerank
    ///    context can be provided to rerank the final results by yet an additional context.
    ///    
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph to perform random walks on
    ///    
    ///    embeddings : NodeEmbeddings
    ///        Set of embeddings which reference nodes within the graph.
    ///    
    ///    node : FQNode
    ///        Fully qualified node: (NodeType, NodeName)
    ///    
    ///    context : Query
    ///        Context, which can be either an embedding or a node lookup, for which to perform
    ///        distances against.  
    ///
    ///    seed : Int - Optional
    ///        If provided, sets the random seed.  Otherwise, uses a global fixed seed.
    ///    
    ///    k : Int - Optional
    ///        If provided, truncates the list to the top K.
    ///    
    ///    rerank_context : Query - Optional
    ///        If provided, reranks the final result set by the rerank context.
    ///    
    ///    filter_type : String - Optional
    ///        If provided, only returns nodes that match the provided node type.
    ///    
    ///    
    ///    Returns
    ///    -------
    ///    List[(FQNode, Float)] - Can throw exception
    ///        List of fully qualified nodes and their fractional scores.
    ///        
    pub fn walk(
        &self, 
        graph: &Graph,
        embeddings: &NodeEmbeddings,
        node: FQNode, 
        context: &Query,
        k: Option<usize>, 
        seed: Option<u64>, 
        rerank_context: Option<&Query>,
        filter_type: Option<String>
    ) -> PyResult<Vec<(FQNode, f32)>> {
        let node_id = get_node_id(graph.vocab.deref(), node.0, node.1)?;
        let g_emb = lookup_embedding(context, embeddings)?;
        
        let steps = if self.restarts >= 1. {
            GSteps::Fixed(self.restarts as usize)
        } else if self.restarts > 0. {
            let one_percent = 0.01f32.ln() / (1. - self.restarts).ln();
            GSteps::Probability(self.restarts, (one_percent).ceil() as usize)
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

/// Computes a SparsePPR
#[pyclass]
#[derive(Clone)]
struct SparsePPR {
    restarts: f32,
    eps: f32
}

#[pymethods]
impl SparsePPR {

    ///    Creates a Sparse Personalized Page Rank.  Unlike sampling approaches used by
    ///    RandomWalker, this uses a different personalized page rank estimator controllable by a
    ///    provided error.  In general, it's less flexible than sampling based approaches but can
    ///    be substantially faster for graphs with low degree counts.
    ///
    ///    In cases where degree count can be high, can be substantially slower than estimate based
    ///    approaches.
    ///    
    ///    Parameters
    ///    ----------
    ///    restarts : Float
    ///        restarts ~ (0,1), determines the probability a random walk will terminate with
    ///        lower restarts resulting in longer walks.
    ///    
    ///    eps : Float - Optional
    ///        If provided, specifies the tolerable error rate allowed for the estimation.  Default
    ///        is 1e-5.
    ///    
    ///    Returns
    ///    -------
    ///    Self - Can throw exception
    ///        
    ///    
    #[new]
    fn new(restarts: f32, eps: Option<f32>) -> PyResult<Self> {
        if restarts <= 0f32 || restarts >= 1f32 {
            return Err(PyValueError::new_err("restarts must be between (0, 1)"))
        }
        Ok(SparsePPR { restarts, eps: eps.unwrap_or(1e-5) })
    }

    /// Simple representation of the SparsePPR
    pub fn __repr__(&self) -> String {
        format!("SparsePPR<restarts={}, eps={}>", self.restarts, self.eps)
    }

    ///    Computes the personalized page rank estimate for a given node
    ///    
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph to perform the PPR on
    ///    
    ///    node : FQNode
    ///        Starting node for PPR.
    ///    
    ///    k : Int - Optional
    ///        If provided, returns only the top K nodes and scores; otherwise provides all.
    ///    
    ///    filter_type : String - Optional
    ///        If provided, filters out nodes that do not match the provided filter_type.
    ///    
    ///    Returns
    ///    -------
    ///    List[(FQNode, f32)] - Can throw exception
    ///        List of fully qualified nodes and their fractional scores
    ///    
    pub fn compute(
        &self, 
        graph: &Graph,
        node: FQNode, 
        k: Option<usize>, 
        filter_type: Option<String>
    ) -> PyResult<Vec<(FQNode, f32)>> {

        let node_id = get_node_id(graph.vocab.deref(), node.0, node.1)?;
        let results = ppr_estimate(graph.graph.as_ref(), node_id, self.restarts, self.eps);

        Ok(convert_scores(&graph.vocab, results.into_iter(), k, filter_type))
    }

}


/// Type of edge.  Undirected edges internally get converted to two directed edges.
#[pyclass]
#[derive(Clone)]
pub enum EdgeType {
    Directed,
    Undirected
}

/// Allows the user to build a graph incrementally before converting it into a proper CSR graph
#[pyclass]
struct GraphBuilder {
    vocab: Vocab,
    edges: Vec<(NodeID, NodeID, f32)>
}

impl GraphBuilder {
    fn compact_edges(edges: &mut Vec<(NodeID, NodeID, f32)>) {
        edges.par_sort_by_key(|(f_n, t_n, _)| (*f_n, *t_n));
        let mut cur_record = 0;
        let mut idx = 1;
        while idx < edges.len() {
            let (f_n, t_n, w) = edges[idx];
            let c_r = edges[cur_record];
            // Same edge, add the weights.
            if f_n == c_r.0 && t_n == c_r.1 {
                (&mut edges[cur_record]).2 += w;
            } else {
                // Different record, move it
                cur_record += 1;
                edges[cur_record] = edges[idx];
            }
            idx += 1;
        }
        edges.truncate(cur_record + 1);
    }
}

#[pymethods]
impl GraphBuilder {
    ///    Creates a new graph builder instance.  This allows for the programatic construction of
    ///    graphs, creating a fully fledged and optimized graph at the end.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[new]
    pub fn new() -> Self {
        GraphBuilder {
            vocab: Vocab::new(),
            edges: Vec::new()
        }
    }
    
    /// Simple representation of the GraphBuilder
    pub fn __repr__(&self) -> String {
        format!("GraphBuilder<Nodes={}, Edges={}>", self.vocab.len(), self.edges.len())
    }
 
    ///    Adds an edge to the graph.
    ///    
    ///    Parameters
    ///    ----------
    ///    from_node : FQNode
    ///        Originating node.
    ///    
    ///    to_node : FQNode
    ///        Destination Node
    ///    
    ///    weight : Float
    ///        Associated Edge weight, if application
    ///    
    ///    node_type : EdgeType
    ///        If Directed, only creates the edge in one direction.  If undirected, creates two
    ///        edges from from_node -> to_node and to_node -> from_node, each with the same weight.
    ///    
    pub fn add_edge(
        &mut self, 
        from_node: FQNode, 
        to_node: FQNode,
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

    ///    Constructs the graph
    ///    
    ///    Returns
    ///    -------
    ///    Graph - Optional
    ///        Creates a Graph for usage.  If no edges have been specified, returns None.
    ///    
    pub fn build_graph(&mut self) -> Option<Graph> {
        if self.edges.len() == 0 {
            return None
        }
        // We swap the internal buffers with new buffers; we do this to preserve memory whenever
        // possible.
        let mut vocab = Vocab::new(); 
        let mut edges = Vec::new();
        std::mem::swap(&mut vocab, &mut self.vocab);
        std::mem::swap(&mut edges, &mut self.edges);

        GraphBuilder::compact_edges(&mut edges);
        let graph = CSR::construct_from_edges(edges);

        Some(Graph {
            graph: Arc::new(CumCSR::convert(graph)),
            vocab: Arc::new(vocab)
        })
    }

}

#[pyclass]
#[derive(Clone)]
struct LossWeighting {
    loss: EPLW
}

#[pymethods]
impl LossWeighting {

    #[allow(non_snake_case)]
    #[staticmethod]
    pub fn Log() -> Self {
        LossWeighting { loss: EPLW::DegreeLog }
    }

    #[allow(non_snake_case)]
    #[staticmethod]
    pub fn Exponential(weight: f32) -> Self {
        LossWeighting { loss: EPLW::DegreeExponential(weight) }
    }

}

/// A python wrapper for the internal ADT used for defining losses
#[pyclass]
#[derive(Clone)]
struct EPLoss {
    loss: Loss
}

#[pymethods]
impl EPLoss {

    ///    Uses thresholded Margin loss for Embedding Propagation.  This is equivalent to the loss
    ///    used in the Embedding Propagation paper.
    ///    
    ///    Parameters
    ///    ----------
    ///    gamma : Float
    ///        Threshold for which to ignore distance computation.
    ///    
    ///    negatives : Int - Optional
    ///        If provided, the number of negatives samples to use.  Default is 1
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[staticmethod]
    pub fn margin(gamma: f32, negatives: Option<usize>) -> Self {
        EPLoss { loss: Loss::MarginLoss(gamma, negatives.unwrap_or(1)) }
    }

    ///    Uses temperature controlled contrastive loss with cosine similarity for optimization.
    /// 
    ///    Parameters
    ///    ----------
    ///    positive_margin : Float
    ///        Margin threshold for positives.  
    ///    
    ///    negative_margin : Float
    ///        Margin threshold for negatives.  
    ///    
    ///    negatives : Int
    ///        If provided, the number of negatives samples to use.  Default is 1.
    ///     
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[staticmethod]
    pub fn contrastive(positive_margin: f32, negative_margin: f32, negatives: usize) -> Self {
        EPLoss { loss: Loss::Contrastive(positive_margin, negative_margin, negatives.max(1)) }
    }

    ///    Uses the Starspace loss for optimization, as seen in the Starspace paper.
    ///    
    ///    Parameters
    ///    ----------
    ///    gamma : Float
    ///        Margin threshold.  Higher values spend less time optimizing scores which are
    ///        relatively close to the margin
    ///    
    ///    negatives : Int
    ///        If provided, the number of negatives samples to use.  Default is 1.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[staticmethod]
    pub fn starspace(gamma: f32, negatives: usize) -> Self {
        EPLoss { loss: Loss::StarSpace(gamma, negatives.max(1)) }
    }

    ///    Optimizes for the NLL of a 1-N classification task.  This is also known as ListNet,
    ///    except with a margin.  Uses dot products for similarity.
    ///    
    ///    Parameters
    ///    ----------
    ///    tau : Float
    ///        Tau serves as the margin threshold parameter
    ///    
    ///    negatives : Int
    ///        If provided, the number of negatives samples to use.  Default is 1.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[staticmethod]
    pub fn rank(tau: f32, negatives: usize) -> Self {
        EPLoss { loss: Loss::RankLoss(tau, negatives.max(1)) }
    }

    ///    Combines Starspace and RankLoss losses as an aggregate.  Starspace better regulates
    ///    magnitude of embeddings while RankLoss better organizes positive to negative sets.
    ///    
    ///    Parameters
    ///    ----------
    ///    tau : Float
    ///        Serves as the threshold parameter for both StarSpace and RankLoss.
    ///    
    ///    negatives : Int
    ///        If provided, the number of negatives samples to use.  Default is 1.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[staticmethod]
    pub fn rankspace(tau: f32, negatives: usize) -> Self {
        EPLoss { loss: Loss::RankSpace(tau, negatives.max(1)) }
    }

    ///    PPR is an interesting loss.  Unlike the other losses, it constructs the positive node
    ///    embedding via a personalized random walks instead of just the immediate neighbors.  This
    ///    has the effect of learning a smoothed node embedding.
    ///    
    ///    Parameters
    ///    ----------
    ///    gamma : Float
    ///        Margin parameter
    ///    
    ///    negatives : Int
    ///        Number of walks and number of negatives to use for constructing positive and
    ///        negative examples.
    ///    
    ///    restart_p : Float
    ///        Restart probability ~ [0,1]
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[staticmethod]
    pub fn ppr(gamma: f32, negatives: usize, restart_p: f32) -> Self {
        EPLoss { loss: Loss::PPR(gamma, negatives.max(1), restart_p) }
    }
    
    /// Simple representation of the GraphBuilder
    pub fn __repr__(&self) -> String {
        format!("EPLoss<{:?}>", self.loss)
    }

}

/// A wrapper for model types
enum ModelType {
    Averaged(AveragedFeatureModel),
    Attention(AttentionFeatureModel)
}

/// The main embedding class.  Flexible with loads of options.
#[pyclass]
struct EmbeddingPropagator {
    ep: EmbeddingPropagation,
    model: ModelType 
}

#[pymethods]
impl EmbeddingPropagator {
    ///    Instantiates a new EmbeddingPropagator.  This is a fairly complex method and more
    ///    details on many of the parameters are found in the original paper.
    ///    
    ///    Parameters
    ///    ----------
    ///    alpha : Float - Optional
    ///        If provided, the starting learning rate used.  Default is 9e-1.
    ///    
    ///    loss : EPLoss - Optional
    ///        Loss function to use.  This defines both the distance function as well as the loss
    ///        function.  Default is Margin(1, 1).
    ///    
    ///    batch_size : Int - Optional
    ///        Batch size to use.  Higher batch sizes are slower and have fewer update steps but
    ///        offer lower variance gradients.  Default is 50.
    ///    
    ///    dims : Int - Optional
    ///        Embedding dimentions for both features as well node embeddings.  Default is 100.
    ///    
    ///    passes : Int - Optional
    ///        Number of passes to run over the graph.  Default is 100.
    ///    
    ///    seed : Int - Optional
    ///        Random seed to use for optimization.  Default is a global constant.
    ///    
    ///    max_nodes : Int - Optional
    ///        Number of neighbor nodes to use for reconstructing the the node embedding estimate.
    ///        The larger the number, the better the estimate, but is more computationally
    ///        expensive.  These nodes are randomly selected each pass. Default is all.
    ///    
    ///    max_features : Int - Optional
    ///        Maximum number of features to use to construct a node embedding.  These features
    ///        will be randomly selected every node construction.  Default is all.
    ///    
    ///    valid_pct : Float - Optional
    ///        Takes a percentage of the nodes in the graph and uses them to measure validation.
    ///        Not very useful right now, default is 0.1
    ///    
    ///    hard_negatives : Int - Optional
    ///        Finds hard negatives by performing a random walk in the neighborhood and using it to
    ///        select a negative.  Default is 0
    ///    
    ///    indicator : Bool - Optional
    ///        Shows a progress bar.  Default is True
    ///    
    ///    attention : Int - Optional
    ///        If provided, uses softmax attention to construct node embeddings.  If provided, this
    ///        adds to the `dim` space into query, key, and value functions.  This specifies the
    ///        size used for Query and Value, resulting in an embedding that is "2*`attention` +
    ///        dim".
    ///
    ///        Note: while this functionality is here, it is _incredibly_ computationally expensive
    ///        to do on CPU.  Useful only for small graphs or extreme patience.
    ///
    ///        Default is 0.
    ///    
    ///    attention_heads : Int - Optional
    ///        Number of heads to use.  This expands the model size even more.
    ///
    ///        Default is 1.
    ///    
    ///    context_window : Int - Optional
    ///        If provided, uses sliding attention which can be helpful for NLP node features
    ///
    ///        Default is None
    ///    
    ///    noise : Float - Optional
    ///        If added, injects gaussian noise into the gradients to help with generalization.  
    ///
    ///        Default is None.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[new]
    pub fn new(
        // Learning rate
        alpha: Option<f32>, 

        // Optimization loss
        loss: Option<EPLoss>,

        // Batch size 
        batch_size: Option<usize>, 

        // Node embedding size
        dims: Option<usize>,

        // Number of passes to run
        passes: Option<usize>,

        // Random seed to use
        seed: Option<u64>,

        // Max neighbors to use for reconstructions
        max_nodes: Option<usize>,

        // How we sample neighbors
        weighted_neighbor_sampling: Option<bool>,

        // How we blend neighbors during reconstruction
        weighted_neighbor_averaging: Option<bool>,

        // Max features to use for optimization
        max_features: Option<usize>,

        // How to weight the loss in the graph
        loss_weighting: Option<LossWeighting>,

        // Percentage of nodes to use for validation
        valid_pct: Option<f32>,

        // Number of hard negatives, produced from random walks.  The quality of these deeply
        // depend on the quality of the graph
        hard_negatives: Option<usize>,

        // Whether to have a pretty indicator.
        indicator: Option<bool>,

        // Number of dims to use for attention.  If missing, uses the Averaged model
        attention: Option<usize>,

        // Number of heads to use.  Defaults to one, but only used if attention dims are set
        attention_heads: Option<usize>,

        // Use sliding window context.
        context_window: Option<usize>,

        // Use gradient noise where we sample from the normal distribution and blend with `noise`
        noise: Option<f32>
    ) -> Self {
        let loss_weighting = loss_weighting.map(|lw| lw.loss).unwrap_or(EPLW::None);
        let ep = EmbeddingPropagation {
            alpha: alpha.unwrap_or(0.9),
            batch_size: batch_size.unwrap_or(50),
            d_model: dims.unwrap_or(100),
            passes: passes.unwrap_or(100),
            loss: loss.map(|l|l.loss).unwrap_or(Loss::MarginLoss(1f32,1)),
            hard_negs: hard_negatives.unwrap_or(0),
            loss_weighting: loss_weighting,
            valid_pct: valid_pct.unwrap_or(0.1),
            seed: seed.unwrap_or(SEED),
            indicator: indicator.unwrap_or(true),
            noise: noise.unwrap_or(0.0)
        };

        let wns = weighted_neighbor_sampling.unwrap_or(false);
        let wna = weighted_neighbor_averaging.unwrap_or(false);

        let model = if let Some(d_k) = attention {
            let num_heads = attention_heads.unwrap_or(1);
            let at = if let Some(size) = context_window {
                AttentionType::Sliding{window_size: size}
            } else if let Some(k) = max_features {
                AttentionType::Random { num_features: k }
            } else {
                AttentionType::Full
            };
            let mha = MultiHeadedAttention::new(num_heads, d_k, at);
            ModelType::Attention(AttentionFeatureModel::new(mha, None, max_nodes, wns))
        } else {
            ModelType::Averaged(AveragedFeatureModel::new(
                    max_features, max_nodes, wns, wna
            ))
        };

        EmbeddingPropagator{ ep, model }
    }

    ///    Learns the features from a given graph
    ///    
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph to learn against.
    ///    
    ///    features : FeatureSet
    ///        FeatureSet for nodes in the graph
    ///    
    ///    feature_embeddings : mut NodeEmbeddings - Optional
    ///        If not provided, creates a new randomized feature_embedding set.
    ///    
    ///    Returns
    ///    -------
    ///    NodeEmbeddings
    ///        A mapping from features -> embedding
    ///    
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
    
    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.ep)
    }

}

/// Defines the FeatureSet class, which allows setting discrete features for a node
#[pyclass]
pub struct FeatureSet {
    vocab: Arc<Vocab>,
    features: FeatureStore
}

impl FeatureSet {

    fn read_vocab_from_file(path: &str) -> PyResult<Vocab> {
        let mut vocab = Vocab::new();
        let f = File::open(path)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        let br = BufReader::new(f);
        for line in br.lines() {
            let line = line.unwrap();
            let pieces: Vec<_> = line.split('\t').collect();
            if pieces.len() != 3 {
                return Err(PyValueError::new_err("Malformed feature line! Need node_type<TAB>name<TAB>f1 f2 ..."))
            }
            vocab.get_or_insert(pieces[0].into(), pieces[1].into());
        }
        Ok(vocab)
    }
}

#[pymethods]
impl FeatureSet {

    ///    Creates a FeatureSet tied to a graph.  We do this to minimize duplicate vocabularies and
    ///    guaranteed 1:1 mapping of nodes to features
    ///    
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph to construct FeatureSet for
    ///    
    ///    path : String - Optional
    ///        If provided, reads features from a file.  If not, creates a FeatureSet which is
    ///        empty.
    ///    
    ///    namespace : String - Optional
    ///        If provided, overrides the default 'node type' for a feature.  Default is 'feat'.
    ///    
    ///    Returns
    ///    -------
    ///    Self - Can throw exception
    ///        
    ///    
    #[staticmethod]
    pub fn new_from_graph(graph: &Graph, path: Option<String>, namespace: Option<String>) -> PyResult<Self> {

        let ns = namespace.unwrap_or_else(|| "feat".to_string());
        let mut fs = FeatureSet {
            vocab: graph.vocab.clone(),
            features: FeatureStore::new(graph.graph.len(), ns)
        };

        if let Some(path) = path {
            fs.load_into(path)?;
        }
        Ok(fs)
    }

    // Loads features tied to a graph
    ///    Loads a feature set from file.  This is less memory efficient than using
    ///    `new_from_graph`.
    ///    
    ///    Parameters
    ///    ----------
    ///    path : String
    ///        Path to load features from.
    ///    
    ///    namespace : String - Optional
    ///        If provided, overrides the default 'node type' for a feature.  Default is 'feat'.
    ///    
    ///    Returns
    ///    -------
    ///    Self - Can throw exception
    ///        
    ///    
    #[staticmethod]
    pub fn new_from_file(path: String, namespace: Option<String>) -> PyResult<Self> {

        // Build the vocab from the file first, get the length of the feature set
        let vocab = Arc::new(FeatureSet::read_vocab_from_file(&path)?);
        let ns = namespace.unwrap_or_else(|| "feat".to_string());
        let feats = FeatureStore::new(vocab.len(), ns);
        let mut fs = FeatureSet {
            vocab: vocab,
            features: feats
        };

        fs.load_into(path)?;
        Ok(fs)
    }

    ///    Sets the features for a Node.
    ///    
    ///    Parameters
    ///    ----------
    ///    node : FQNode
    ///        Fully qualified Node.
    ///    
    ///    features : List[String]
    ///        Features to set for this node.
    ///    
    ///    Returns
    ///    -------
    ///    () - Can throw exception
    ///        
    ///    
    pub fn set_features(&mut self, node: FQNode, features: Vec<String>) -> PyResult<()> {
        let node_id = get_node_id(self.vocab.deref(), node.0, node.1)?;
        self.features.set_features(node_id, features);
        Ok(())
    }

    ///    Retrieves the set of features defined for a node.
    ///    
    ///    Parameters
    ///    ----------
    ///    node : FQNode
    ///        Fully qualified Node.
    ///    
    ///    Returns
    ///    -------
    ///    List[String] - Can throw exception
    ///        Set of features specified for node
    ///    
    pub fn get_features(&self, node: FQNode) -> PyResult<Vec<String>> {
        let node_id = get_node_id(self.vocab.deref(), node.0, node.1)?;
        Ok(self.features.get_pretty_features(node_id))
    }

    ///    Loads a file defining fully qualified nodes to features into a feature set.
    ///    
    ///    Parameters
    ///    ----------
    ///    path : String
    ///        Path point to features definitions.
    ///    
    ///    Returns
    ///    -------
    ///    () - Can throw exception
    ///        
    ///    
    pub fn load_into(
        &mut self, 
        path: String
    ) -> PyResult<()> {
        let reader = open_file_for_reading(&path)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        for line in reader.lines() {
            let line = line.unwrap();
            let pieces: Vec<_> = line.split('\t').collect();
            if pieces.len() != 3 {
                return Err(PyValueError::new_err("Malformed feature line! Need node_type<TAB>name<TAB>f1 f2 ..."))
            }
            let bow = pieces[2].split_whitespace()
                .map(|s| s.to_string()).collect();
            let _ = self.set_features((pieces[0].to_string(), pieces[1].to_string()), bow);
                
        }
        Ok(())
    }

    ///    Returns the number of nodes in the feature set.
    ///    
    ///    Returns
    ///    -------
    ///    Int
    ///    
    pub fn nodes(&self) -> usize {
        self.vocab.len()
    }

    ///    Returns the number of unique features defined in the featureset
    ///    
    ///    Returns
    ///    -------
    ///    Int
    ///        
    ///    
    pub fn num_features(&self) -> usize {
        self.features.num_features()
    }

    ///    Iterator over fully qualified nodes.
    ///    
    ///    Returns
    ///    -------
    ///    VocabIterator
    ///        
    ///    
    pub fn vocab(&self) -> VocabIterator {
        VocabIterator::new(self.vocab.clone())
    }

    ///    Returns a new featureset with only features greater than `count`.
    ///    
    ///    Parameters
    ///    ----------
    ///    count : Int
    ///        Filtering threshhold.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    pub fn prune_min_count(&self, count: usize) -> Self {
        FeatureSet {
            features: self.features.prune_min_count(count),
            vocab: self.vocab.clone()
        }
    }
    
    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("FeatureSet<Nodes={},UniqueFeatures={}>", self.vocab.len(), self.features.num_features())
    }

}

/// Propagates features within a feature set, using a graph to find neighbors.
#[pyclass]
pub struct FeaturePropagator {
    /// Max number of features for each node
    k: usize,

    /// Filters out feature which don't meet the minimum feature count
    threshold: f32,

    /// Number of passes to run.  In practice, this should be pretty small.
    max_iters: usize
}

#[pymethods]
impl FeaturePropagator {
    ///    Uses a limited form of propagation to fill in FeatureSets where node features are
    ///    missing.  Of limited value in many cases but can be helpful for extremely large graphs
    ///    in some cases.
    ///    
    ///    Parameters
    ///    ----------
    ///    k : Int
    ///        Max number of nodes to use.
    ///    
    ///    threshold : Float - Optional
    ///        if the count / norm is less than threshold, filters out the feature.  Default is 0.
    ///    
    ///    max_iters : Int - Optional
    ///        Number of passes to run.  Default is 20.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[new]
    pub fn new(k: usize, threshold: Option<f32>, max_iters: Option<usize>) -> Self {
        FeaturePropagator { 
            k: k, 
            threshold: threshold.unwrap_or(0.),
            max_iters: max_iters.unwrap_or(20)
        }
    }
    
    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("FeaturePropagator<k={},threshold={},max_iters={}>", self.k, self.threshold, self.max_iters)
    }

    ///    Propagates features throughout the graph.
    ///    
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph to propagate.
    ///    
    ///    features : mut FeatureSet
    ///        Features to modify.
    ///    
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

/// After we train a set of features using EP, we need to use an aggregator to take a set of
/// features and glue them into a new NodeEmbedding
#[derive(Clone)]
enum AggregatorType {
    Averaged,
    Weighted {
        alpha: f32, 
        vocab: Arc<Vocab>,
        unigrams: Arc<UnigramProbability>
    },
    Attention {
        num_heads: usize,
        d_k: usize,
        window: Option<usize>
    }
}

/// This constructs node embeddings based on the AggregatorType and a set of FeatureEmbeddings.
#[pyclass]
#[derive(Clone)]
pub struct FeatureAggregator {
    at: AggregatorType
}

#[pymethods]
impl FeatureAggregator {

    ///    Averages features together to create node embeddings.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[allow(non_snake_case)]
    #[staticmethod]
    pub fn Averaged() -> Self {
        FeatureAggregator { at: AggregatorType::Averaged }
    }

    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        let t = match &self.at {
            AggregatorType::Averaged => "Averaged".into(),
            AggregatorType::Weighted {alpha, vocab: _, unigrams: _} => format!("Weighted<alpha={}>", alpha),
            AggregatorType::Attention {num_heads, d_k, window} => format!("Attention<num_heads={},d_k={},window={:?}", num_heads, d_k, window)
        };
        format!("FeatureAggregator<{}>", t)
    }

    ///    Uses attention across features to construct the node embeddings.
    ///    
    ///    Parameters
    ///    ----------
    ///    num_heads : Int
    ///        Number of attention heads.
    ///    
    ///    d_k : Int
    ///        Dimension of the Query and Key fields.
    ///    
    ///    window : Int - Optional
    ///        If provided, uses sliding window attention.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[allow(non_snake_case)]
    #[staticmethod]
    pub fn Attention(num_heads: usize, d_k: usize, window: Option<usize>) -> Self {
        FeatureAggregator { at: AggregatorType::Attention {num_heads, d_k, window} }
    }

    ///    Uses weights derived from feature frequency to bias node embeddings to rarer features.
    ///    
    ///    Parameters
    ///    ----------
    ///    alpha : Float
    ///        Amount to bias weights.
    ///    
    ///    fs : FeatureSet
    ///        FeatureSet to learn weights from, using unigram probability.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[allow(non_snake_case)]
    #[staticmethod]
    pub fn Weighted(alpha: f32, fs: &FeatureSet) -> Self {
        let unigrams = Arc::new(UnigramProbability::new(&fs.features));
        let vocab = fs.vocab.clone();
        FeatureAggregator { at: AggregatorType::Weighted {alpha, vocab, unigrams} }
    }

    ///    Write the Aggregator to disk.  Stores any learned parameter, such as from Weighted, in
    ///    the data format
    ///    
    ///    Parameters
    ///    ----------
    ///    path : str
    ///        Path to write aggregator to.
    ///    
    ///    Returns
    ///    -------
    ///    () - Can throw exception
    ///        
    ///    
    pub fn save(&self, path: &str) -> PyResult<()> {
        let f = File::create(path)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        let mut bw = BufWriter::new(f);
        match &self.at {
            AggregatorType::Averaged => {
                writeln!(&mut bw, "Averaged")
                    .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;
            },
            AggregatorType::Attention { num_heads, d_k, window } => {
                writeln!(&mut bw, "Attention")
                    .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;
                writeln!(&mut bw, "{}", num_heads)
                    .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;
                writeln!(&mut bw, "{}", d_k)
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

    ///    Loads an Aggregator from file.
    ///    
    ///    Parameters
    ///    ----------
    ///    path : String
    ///        Path to the serialized aggregator.
    ///    
    ///    Returns
    ///    -------
    ///    Self - Can throw exception
    ///        
    ///    
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
                let num_heads = line.trim_end().parse::<usize>()
                    .map_err(|_e| PyValueError::new_err(format!("invalid dim! {:?}", line)))?;

                line.clear();
                br.read_line(&mut line)
                    .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;
                let d_k = line.trim_end().parse::<usize>()
                    .map_err(|_e| PyValueError::new_err(format!("invalid dim! {:?}", line)))?;

                line.clear();
                br.read_line(&mut line)
                    .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;
                let window = line.trim_end().parse::<usize>()
                    .map_err(|_e| PyValueError::new_err(format!("invalid dim! {:?}", line)))?;

                let window = if window == 0 {
                    None
                } else {
                    Some(window)
                };

                Ok(FeatureAggregator::Attention(num_heads, d_k, window))
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

/// Finally, a node embedder.  This wraps the FeatureAggregator because who doesn't like more
/// indirection?
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
            AggregatorType::Attention {num_heads, d_k, window} => {
                let at = if let Some(window_size) = window {
                    AttentionType::Sliding { window_size: *window_size }
                } else {
                    AttentionType::Full
                };
                let mha = MultiHeadedAttention::new(*num_heads, *d_k, at);
                Box::new(AttentionAggregator::new(es, mha))
            }

        }
    }

    fn get_dims(&self, feat_embs: &NodeEmbeddings) -> usize {
        match self.feat_agg.at {
            AggregatorType::Attention { num_heads, d_k, window:_ } =>  {
                let attention_dims = 2 * num_heads * d_k;
                (feat_embs.dims() - attention_dims) / num_heads
            },
            _ => feat_embs.dims()
        }
    }
}

#[pymethods]
impl NodeEmbedder {
    ///    Creates a new Embedder which takes a FeatureSet and Feature embeddings to construct a
    ///    node embeddings.
    ///    
    ///    Parameters
    ///    ----------
    ///    feat_agg : FeatureAggregator
    ///        Feature aggregator method to use.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[new]
    pub fn new(feat_agg: FeatureAggregator) -> Self {
        NodeEmbedder { feat_agg }
    }

    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("NodeEmbedder<aggregator={}>", self.feat_agg.__repr__())
    }

    ///    Embeds a full feature set into an set of Embeddings.
    ///    
    ///    Parameters
    ///    ----------
    ///    feat_set : FeatureSet
    ///        Feature set to embed.
    ///    
    ///    feature_embeddings : NodeEmbeddings
    ///        Feature embeddings.
    ///    
    ///    Returns
    ///    -------
    ///    NodeEmbeddings
    ///        
    ///    
    pub fn embed_feature_set(
        &self, 
        feat_set: &FeatureSet, 
        feature_embeddings: &NodeEmbeddings,
    ) -> NodeEmbeddings {

        let num_nodes = feat_set.vocab.len();
        let dims = self.get_dims(feature_embeddings);

        let es = EmbeddingStore::new(num_nodes, dims, EDist::Cosine);
        let agg = self.get_aggregator(&feature_embeddings.embeddings, &self.feat_agg.at);
        let fs_vocab = feat_set.features.get_vocab();
        (0..num_nodes).into_par_iter().for_each(|node| {
            // Safe to access in parallel
            let embedding = es.get_embedding_mut_hogwild(node);
            // Translate nodes
            let new_feats: Vec<_> = feat_set.features.get_features(node)
                .iter()
                .map(|feat_id| feature_embeddings.vocab.translate_node(&fs_vocab, *feat_id))
                .filter(|n| n.is_some())
                .map(|n| n.unwrap())
                .collect();
            
            if new_feats.len() > 0 {
                agg.construct(&new_feats, embedding);
            }
        });

        NodeEmbeddings {
            vocab: feat_set.vocab.clone(),
            embeddings: es
        }
    }

    ///    Embeds an adhoc feature set into an embedding.
    ///    
    ///    Parameters
    ///    ----------
    ///    features : List[FQNode]
    ///        List of fully qualified features to embed.  Usually 'feat' is the type.
    ///    
    ///    feature_embeddings : NodeEmbeddings
    ///        Feature embeddings.
    ///    
    ///    strict : Bool - Optional
    ///        If true, requires all features to exist in the embedding or throws an error.  If
    ///        false, will create an embedding only from such features that are defined.  
    ///
    ///        Default is True.
    ///    
    ///    Returns
    ///    -------
    ///    List[Float] - Can throw exception
    ///        Embedding
    ///    
    pub fn embed_adhoc(
        &self, 
        features: Vec<FQNode>,
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

    ///    Same as embed_adhoc but with multiple feature sets.
    ///    
    ///    Parameters
    ///    ----------
    ///    features_set : List[List[FQNode]]
    ///        Adhoc feature sets to use.
    ///    
    ///    feature_embeddings : NodeEmbeddings
    ///        Feature embeddings.
    ///        
    ///    
    ///    strict : Bool - Optional
    ///        If true, requires all features to exist in the embedding or throws an error.  If
    ///        false, will create an embedding only from such features that are defined.  
    ///
    ///        Default is True.
    ///        
    ///    
    ///    Returns
    ///    -------
    ///    List[List[Float]] - Can throw exception
    ///        
    ///    
    pub fn bulk_embed_adhoc(&self, 
        features_set: Vec<Vec<FQNode>>,
        feature_embeddings: &NodeEmbeddings,
        strict: Option<bool>
    ) -> PyResult<Vec<Vec<f32>>> {
        let results: PyResult<Vec<_>> = features_set.into_par_iter().map(|features| {
            self.embed_adhoc(features, feature_embeddings, strict)
        }).collect();

        results
    }

}

/// Struct for defining ALT embeddings
#[pyclass]
struct DistanceEmbedder {
    landmarks: algos::dist::LandmarkSelection,
    n_landmarks: usize
}

#[pymethods]
impl DistanceEmbedder {
    ///    Creates a Distance embedding.  Distance embedding uses a method called landmark
    ///    triangulation to compute a "Distance" embedding.  The procedure works as follows:
    ///
    ///    1. Select K landmarks.
    ///    2. Compute the distance from each of those landmarks to every node in the graph.
    ///    3. Create an embedding which is the distance for each node.
    ///    4. Compute distances between modes using an algorithm called ALT (A* landmark
    ///       triangulation).
    ///
    ///    It's very fast and can encode useful properties of a graph, but does require a fully
    ///    connected graph.
    ///    
    ///    Parameters
    ///    ----------
    ///    n_landmarks : Int
    ///        Number of landmarks to use.  The larger the landmark size, the greater the fidelity
    ///        of the embedding.
    ///    
    ///    seed : Int - Optional
    ///        If provided, randomly selects landmarks randomly rather than using the top K
    ///        landmarks by degree.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
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
    
    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("DistanceEmbedder<landmarks={:?}, n_landmarks={}>", self.landmarks, self.n_landmarks)
    }


    ///    Learns distance embeddings for a graph.
    ///    
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph to learn distance embeddings.
    ///    
    ///    Returns
    ///    -------
    ///    NodeEmbeddings
    ///        Embedding set, using the Distance.ALT method for similarity.
    ///    
    pub fn learn(&self, graph: &Graph) -> NodeEmbeddings {
        let es = crate::algos::dist::construct_walk_distances(graph.graph.as_ref(), self.n_landmarks, self.landmarks);
        NodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings: es
        }
    }
}

/// Struct for learning Clustered LPA embeddings.  Uses Hamming Distance for equivalence.
#[pyclass]
struct ClusterLPAEmbedder{
    k: usize, 
    passes: usize, 
    seed: Option<u64>
}

#[pymethods]
impl ClusterLPAEmbedder {
    ///    Creates a ClusterLPAEmbedder.
    ///
    ///    ClusterLPAEmbedder runs the LPA algoritm K times with different random seeds and creates
    ///    a HammingDistance NodeEmbedding set.  Very fast.
    ///    
    ///    Parameters
    ///    ----------
    ///    k : Int
    ///        Number of dimensions in NodeEmbeddings.
    ///    
    ///    passes : Int
    ///        Number of passes to run LPA.
    ///    
    ///    seed : Int - Optional
    ///        If provide, use the provided seed.  Default is global seed.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[new]
    pub fn new(k: usize, passes: usize, seed: Option<u64>) -> Self {
        ClusterLPAEmbedder {
            k, passes, seed
        }
    }

    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("ClusterLPAEmbedder<k={}, passes={}, seed={:?}>", self.k, self.passes, self.seed)
    }


    ///    Learns NodeEmebddings on the provided graph.
    ///    
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph to use.
    ///    
    ///    Returns
    ///    -------
    ///    NodeEmbeddings
    ///        Set of NodeEmbeddings with Distance.Hamming.
    ///    
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
#[derive(Clone,Copy)]
pub enum ListenerRule {
    Best,
    Probabilistic
}

/// Struct for learning Speaker-Listener multi-cluster embeddings.  Uses Hamming Distance for
/// distance.
#[pyclass]
struct SLPAEmbedder {
    t: usize, 
    threshold: usize, 
    rule: crate::algos::slpa::ListenerRule,
    memory_size: Option<usize>,
    seed: Option<u64>
}

#[pymethods]
impl SLPAEmbedder {
    ///    Creates a Speaker-Listener multi-cluster embedder.  Somewhat better than clustering LPA
    ///    by allowing overlapping communities in a more principaled way.
    ///    
    ///    Parameters
    ///    ----------
    ///    t : Int
    ///        Number of passes to run.
    ///    
    ///    threshold : Int
    ///        Filtering threshold: clusters which have a weight less than threshold are truncated.
    ///
    ///    memory_size : Int - Optional
    ///        Each node has a memory of memory_size.  If memory_size is smaller than t, overwrites
    ///        history.
    ///
    ///    rule : 
    ///    
    ///    seed : Int - Optional
    ///        If provided, use this seed.  Default is global seed.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[new]
    pub fn new(
        t: usize, 
        threshold: usize, 
        memory_size: Option<usize>, 
        rule: Option<ListenerRule>,
        seed: Option<u64>
    ) -> Self {
        use crate::algos::slpa::{ListenerRule as SLR};
        let lr_rule = if let Some(rule) = rule {
            match rule {
                ListenerRule::Best => SLR::Best,
                ListenerRule::Probabilistic => SLR::Probabilistic
            }
        } else {
            SLR::Best
        };

        SLPAEmbedder {
            t: t.max(1), 
            rule: lr_rule,
            threshold, 
            memory_size,
            seed
        }
    }

    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("SLPAEmbedder<t={}, threshold={}, seed={:?}>", self.t, self.threshold, self.seed)
    }

    ///    Learn SLPA Embeddings
    ///    
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph to learn embeddings.
    ///    
    ///    Returns
    ///    -------
    ///    NodeEmbeddings
    ///        
    ///    
    pub fn learn(&self, graph: &Graph) -> NodeEmbeddings {
        let seed = self.seed.unwrap_or(SEED);
        let es = crate::algos::slpa::construct_slpa_embedding(
            graph.graph.as_ref(), 
            self.rule,
            self.t, 
            self.threshold, 
            self.memory_size.unwrap_or(self.t),
            seed
        );
        NodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings: es
        }
    }

}

/// Computes the PageRank for all nodes in the graph.
#[pyclass]
struct PageRank {
    iterations: usize,
    damping: f32, 
    eps: f32
}

#[pymethods]
impl PageRank {
    ///    Initializes a PageRank struct.  It uses the power iteration approach for learning
    ///    results.
    ///    
    ///    Parameters
    ///    ----------
    ///    iterations : Int
    ///        Number of iterations to run the PageRank algorithm.
    ///    
    ///    damping : Float - Optional
    ///        If provided, sets the restart probability to (1 - damping).  Higher Damping leads to
    ///        more global stationary distributions.  Lower damping preserves more local structure.
    ///
    ///        Default is 0.85
    ///    
    ///    eps : Float - Optional
    ///        Early termination if the page rank scores are less than EPS.  Default is 1e-5.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[new]
    pub fn new(iterations: usize, damping: Option<f32>, eps: Option<f32>) -> Self {
        PageRank {iterations, damping: damping.unwrap_or(0.85), eps: eps.unwrap_or(1e-5) }
    }

    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("PageRank<iterations={}, damping={}, eps={}>", self.iterations, self.damping, self.eps)
    }

    ///    Computes PageRank on a graph
    ///    
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph to use.
    ///    
    ///    indicator : Bool - Optional
    ///        If provided, uses an indicator.  Default is True
    ///    
    ///    Returns
    ///    -------
    ///    NodeEmbeddings
    ///        NodeEmbeddings of dimension=1, where the value is the page rank score.
    ///    
    pub fn learn(&self, graph: &Graph, indicator: Option<bool>) -> NodeEmbeddings {
        let page_rank = crate::algos::pagerank::PageRank::new(self.iterations, self.damping, self.eps);
        let scores = page_rank.compute(graph.graph.as_ref(), indicator.unwrap_or(true));
        let es = EmbeddingStore::new(graph.graph.len(), 1, EDist::Euclidean);
        scores.par_iter().enumerate().for_each(|(node_id, score)| {
            let e1 = es.get_embedding_mut_hogwild(node_id);
            e1[0] = *score;
        });
        NodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings: es
        }
    }

}



/// Wrapper for EmbeddingStore.
#[pyclass]
pub struct NodeEmbeddings {
    vocab: Arc<Vocab>,
    embeddings: EmbeddingStore
}

#[pymethods]
impl NodeEmbeddings {
    ///    Creates a NodeEmbedding set from a given graph.
    ///    
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph to use.
    ///    
    ///    dims : Int
    ///        Number of dimensions for each embedding.
    ///    
    ///    distance : Distance
    ///        Distance metric for computing distances.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[new]
    pub fn new(graph: &Graph, dims: usize, distance: Distance) -> Self {
        let dist = distance.to_edist();

        let es = EmbeddingStore::new(graph.graph.len(), dims, dist);
        NodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings: es
        }
    }

    #[staticmethod]
    pub fn new_from_list(
        list: Vec<((String, String), Vec<f32>)>, 
        distance: Distance
    ) -> Self {
        let mut vocab = Vocab::new();
        let mut embs = Vec::with_capacity(list.len());
        let mut max_emb_size = 0;

        // Add to vocab
        list.into_iter().for_each(|((node_type, node_name), emb)| {
            let node_id = vocab.get_or_insert(node_type, node_name);
            max_emb_size = max_emb_size.max(emb.len());
            embs.push((node_id, emb))
        });

        // Update embeddings
        let mut es = EmbeddingStore::new(vocab.len(), max_emb_size, distance.to_edist());
        embs.into_iter().for_each(|(node_id, emb)| {
            es.set_embedding(node_id, &emb);
        });
        NodeEmbeddings {
            vocab: Arc::new(vocab),
            embeddings: es
        }
    }

    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("NodeEmbeddings<Nodes={}, Dims={}, Distance={:?}>", self.embeddings.len(), 
                self.embeddings.dims(), self.embeddings.distance())
    }

    ///    Checks if a Node exists within the NodeEmbeddings.
    ///    
    ///    Parameters
    ///    ----------
    ///    node : FQNode
    ///        Fully qualified node.
    ///    
    ///    Returns
    ///    -------
    ///    Bool
    ///        True if it exists, False otherwise.
    ///    
    pub fn contains(&self, node: FQNode) -> bool {
        get_node_id(self.vocab.deref(), node.0, node.1).is_ok()
    }

    ///    Returns the Embedding defined for a fully qualified Node.
    ///     
    ///    Parameters
    ///    ----------
    ///    node : FQNode
    ///        Fully qualified Node
    ///    
    ///    Returns
    ///    -------
    ///    List[Float] - Can throw exception
    ///        Embedding associated with the Node
    ///    
    pub fn get_embedding(&mut self, node: FQNode) -> PyResult<Vec<f32>> {
        let node_id = get_node_id(self.vocab.deref(), node.0, node.1)?;
        Ok(self.embeddings.get_embedding(node_id).to_vec())
    }

    ///    Sets the given embedding.
    ///    
    ///    Parameters
    ///    ----------
    ///    node : FQNode
    ///        Fully qualified Node
    ///    
    ///    embedding : List[Float]
    ///        Embedding to set it to.
    ///    
    ///    Returns
    ///    -------
    ///    () - Can throw exception
    ///        
    ///    
    pub fn set_embedding(&mut self, node: FQNode, embedding: Vec<f32>) -> PyResult<()> {
        let node_id = get_node_id(self.vocab.deref(), node.0, node.1)?;
        let es = &mut self.embeddings;
        es.set_embedding(node_id, &embedding);
        Ok(())
    }

    ///    Iterates over the Nodes defined in the NodeEmbeddings.
    ///    
    ///    Returns
    ///    -------
    ///    VocabIterator
    ///        
    ///    
    pub fn vocab(&self) -> VocabIterator {
        VocabIterator::new(self.vocab.clone())
    }

    pub fn get_distance(&self) -> Distance {
        Distance::from_edist(self.embeddings.distance())
    }

    ///    Using a provided embedding, finds the nearest K neighbors to that embedding.
    ///    
    ///    Parameters
    ///    ----------
    ///    emb : List[Float]
    ///        Embedding to nearest neighbor
    ///    
    ///    k : Int
    ///        Top K items to return
    ///    
    ///    filter_type : String - Optional
    ///        If provided, filters out nodes that don't match the filter_type
    ///    
    ///    Returns
    ///    -------
    ///    List[(FQNode, f32)]
    ///        Set of fully qualified nodes and distances.
    ///    
    pub fn nearest_neighbor(
        &self, 
        emb: Vec<f32>, 
        k: usize,
        filter_type: Option<String>
    ) -> Vec<(FQNode, f32)> {
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

    ///    Returns the number of dimensions for an embedding.
    ///    
    ///    Returns
    ///    -------
    ///    Int
    ///        
    ///    
    pub fn dims(&self) -> usize {
        self.embeddings.dims()
    }

    ///    Returns the number of nodes in the embedding set.
    ///    
    ///    Returns
    ///    -------
    ///    Int
    ///        
    ///    
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    ///    Returns the number of nodes in the embedding set.
    ///    
    ///    Returns
    ///    -------
    ///    Int - Can throw exception
    ///        
    ///    
    pub fn __len__(&self) -> PyResult<usize> {
        Ok(self.embeddings.len())
    }

    ///    Computes the distances between either a qualified node or embedding, using the current
    ///    distance method.  This is faster than extracting an embedding from the NodeEmbeddings
    ///    then issuing a distance call due to avoiding serialization (especially problematic for
    ///    big nodes).
    ///    
    ///    Parameters
    ///    ----------
    ///    e1 : Query
    ///        Fully Qualified Node or Embedding
    ///    
    ///    e2 : Query
    ///        Fully Qualified Node or Embedding
    ///    
    ///    Returns
    ///    -------
    ///    Float - Can throw exception
    ///        Throws an exception if a fully qualified node isn't present in the NodeEmbeddings
    ///    
    pub fn compute_distance(
        &self,
        e1: &Query,
        e2: &Query
    ) -> PyResult<f32> {
        let e1_emb = lookup_embedding(e1, self)?;
        let e2_emb = lookup_embedding(e2, self)?;
        let result = self.embeddings.distance().compute(e1_emb, e2_emb);
        Ok(result)
    }

    ///    Returns the node and embedding at internal index `idx`.
    ///    
    ///    Parameters
    ///    ----------
    ///    idx : isize
    ///        Internal index.
    ///    
    ///    Returns
    ///    -------
    ///    (FQNode, List[Float]) - Can throw exception
    ///        
    ///    
    pub fn __getitem__(&self, mut idx: isize) -> PyResult<(FQNode, Vec<f32>)> {
        let len = self.embeddings.len() as isize;
        if idx < 0 {
            idx += len;
        }
        if idx >= len {
            return Err(PyIndexError::new_err(format!("Index larger than embedding store!")));
        }

        let emb = self.embeddings.get_embedding(idx as usize).to_vec();
        let (nt, nn) = self.vocab.get_name(idx as usize).expect("Already validated bounds, something borked");
        Ok((((*nt).clone(), nn.to_string()), emb))
    }

    ///    L2Norms the embeddings, useful for cosine similarity.
    ///    
    pub fn l2norm(&self) {
        (0..self.embeddings.len()).into_par_iter().for_each(|idx| {
            let e = self.embeddings.get_embedding_mut_hogwild(idx);
            let norm = e.iter().map(|ei| ei.powf(2.)).sum::<f32>().sqrt();
            e.iter_mut().for_each(|ei| {
                *ei /= norm;
            });
        });
    }

    ///    Saves the NodeEmbeddings to disk
    ///    
    ///    Parameters
    ///    ----------
    ///    path : str
    ///        Path to store NodeEmbeddings.
    ///    
    ///    Returns
    ///    -------
    ///    () - Can throw exception
    ///        
    ///    
    pub fn save(&self, path: &str, comp_level: Option<u32>) -> PyResult<()> {
        let mut writer = EmbeddingWriter::new(path, self.vocab.as_ref(), comp_level)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        let it = (0..self.vocab.len())
            .map(|node_id| (node_id, self.embeddings.get_embedding(node_id)));

        writer.stream(it)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        Ok(())
    }

    ///    Loads NodeEmbeddings from disk.
    ///    
    ///    Parameters
    ///    ----------
    ///    path : str
    ///        Path where NodeEmbeddings are stored.
    ///    
    ///    distance : Distance
    ///        Distance method to use for computing embedding similarity.
    ///    
    ///    filter_type : String - Optional
    ///        If provided, only loads embeddings which match the provided node type.
    ///    
    ///        Default is None.
    ///
    ///    chunk_size : Int - Optional
    ///        If provided, changes the chunk size.  Larger chunk sizes lets Cloverleaf load
    ///        embeddings with greater parallelism at the expense of more memory.
    ///
    ///        Default is 1_000.
    ///    
    ///    Returns
    ///    -------
    ///    Self - Can throw exception
    ///        
    ///    
    #[staticmethod]
    pub fn load(
        py: Python<'_>,
        path: &str, 
        distance: Distance, 
        filter_type: Option<String>, 
        chunk_size: Option<usize>,
        skip_rows: Option<usize>
    ) -> PyResult<Self> {
        py.allow_threads(move || {
            let (vocab, es) = EmbeddingReader::load(
                path, 
                distance.to_edist(), 
                filter_type, 
                chunk_size, 
                skip_rows
            )?;

            let ne = NodeEmbeddings {
                vocab: Arc::new(vocab),
                embeddings: es
            };
            Ok(ne)
        })
    }
}

/// Allows the user to build NodeEmbeddings incrementally.
#[pyclass]
struct NodeEmbeddingsBuilder {
    vocab: Vocab,
    distance: Distance,
    embeddings: Vec<Vec<f32>>
}

#[pymethods]
impl NodeEmbeddingsBuilder {
    ///    Creates a NodeEmbeddingBuilder.  Allows progressive addition of nodes and embeddings.
    ///    
    ///    Parameters
    ///    ----------
    ///    dist : Distance
    ///        Distance to construct the NodeEmbeddings.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[new]
    pub fn new(dist: Distance) -> Self {
        NodeEmbeddingsBuilder {
            distance: dist,
            vocab: Vocab::new(),
            embeddings: Vec::new()
        }
    }
    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("NodeEmbeddingsBuilder<Nodes={}, Distance={:?}>", self.vocab.len(), self.distance.to_edist())
    }

    ///    Adds a fully qualified node and associated embedding to the buffer.
    ///    
    ///    Parameters
    ///    ----------
    ///    node : FQNode
    ///        Fully qualified Node
    ///    
    ///    embedding : List[Float]
    ///        Associated embedding
    ///    
    ///    Returns
    ///    -------
    ///    () - Can throw exception
    ///        
    ///    
    pub fn add_embedding(
        &mut self, 
        node: FQNode, 
        embedding: Vec<f32>
    ) -> PyResult<()> {

        let n = self.embeddings.len();
        if n > 1 && self.embeddings[0].len() != embedding.len() {
            return Err(PyValueError::new_err("Embedding dimensions mismatch!"));
        }

        let node_id = self.vocab.get_or_insert(node.0, node.1);
        if node_id < self.embeddings.len() {
            self.embeddings[node_id] = embedding;
        } else {
            self.embeddings.push(embedding);
        }
        Ok(())
    }

    ///    Creates the NodeEmbeddings.
    ///    
    ///    Returns
    ///    -------
    ///    NodeEmbeddings - Optional
    ///        If Embeddings have been added, creates the NodeEmbeddings.  Otherwise returns None.
    ///    
    pub fn build(&mut self) -> Option<NodeEmbeddings> {
        if self.embeddings.len() == 0 {
            return None
        }
        // We swap the internal buffers with new buffers; we do this to preserve memory whenever
        // possible.
        let mut vocab = Vocab::new(); 
        let mut embeddings = Vec::new();
        std::mem::swap(&mut vocab, &mut self.vocab);
        std::mem::swap(&mut embeddings, &mut self.embeddings);

        let es = EmbeddingStore::new(embeddings.len(), 
                                     embeddings[0].len(),
                                     self.distance.to_edist());

        embeddings.par_iter_mut().enumerate().for_each(|(i, emb)| {
            let e = es.get_embedding_mut_hogwild(i);
            e.iter_mut().zip(emb.iter()).for_each(|(ei, emb_i)| {
                *ei = *emb_i;
            });
        });

        Some(NodeEmbeddings {
            vocab: Arc::new(vocab),
            embeddings: es
        })
    }

}

/// Simple iterator over the vocab.
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
                let name = name.to_string().into_py(py);
                slf.cur_idx += 1;
                Some((node_type, name).into_py(py))
            } else {
                None
            }
        })
    }
}

/// Wraps the NeighborhoodAligner algorithm.
#[pyclass]
struct NeighborhoodAligner {
    aligner: NA
}

#[pymethods]
impl NeighborhoodAligner {
    ///    Creates a new NeighborhoodAligner.  NeighborhoodAligner adjusts provided NodeEmbeddings
    ///    to capture graph structure, which can be useful for capturing popularity and reinforces
    ///    community.
    ///    
    ///    Parameters
    ///    ----------
    ///    alpha : Float - Optional
    ///        If provided, alpha ~ [0,1] adjust how much influence to place on graph structure
    ///        versus the original embedding.  Higher values of alpha result in more preference
    ///        toward the original embedding, whereas lower values bias more toward graph structure
    ///    
    ///    max_neighbors : Int - Optional
    ///        Number of neighbors to consider for alignment.
    ///
    ///        Default is All.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[new]
    pub fn new(alpha: Option<f32>, max_neighbors: Option<usize>) -> Self {
        let aligner = NA::new(alpha, max_neighbors);
        NeighborhoodAligner {aligner}
    }

    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("NeighborhoodAligner<Alpha={:?}, MaxNeighbors={:?}>", self.aligner.alpha, self.aligner.max_neighbors)
    }

    ///    Creates a new NodeEmbeddings which is the result of neighborhood alignment.
    ///    
    ///    Parameters
    ///    ----------
    ///    embeddings : NodeEmbeddings
    ///        Original NodeEmbeddings
    ///    
    ///    graph : Graph
    ///        Graph structure to use for influencing a Node Embedding.
    ///    
    ///    Returns
    ///    -------
    ///    NodeEmbeddings
    ///        
    ///    
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

    ///    Instead of aligning embeddings in memory, instead progressively writes them out.  This
    ///    is key for managing extremely large graphs with large numbers of nodes, where memory is
    ///    dominated by the afforementioned items.
    ///    
    ///    Parameters
    ///    ----------
    ///    path : str
    ///        Path to align nodes to
    ///    
    ///    embeddings : NodeEmbeddings
    ///        Original NodeEmbeddings
    ///    
    ///    graph : Graph
    ///        Graph to use for neighborhood structure
    ///    
    ///    chunk_size : Int - Optional
    ///        Chunk size to do in parallel.  Higher numbers spend more time in multithreaded
    ///        computation at the expense of memory.
    ///
    ///        Default is 10_000
    ///    
    ///    Returns
    ///    -------
    ///    () - Can throw exception
    ///        
    ///    
    pub fn align_to_disk(
        &self, 
        path: &str,
        embeddings: &NodeEmbeddings, 
        graph: &Graph,
        chunk_size: Option<usize>,
        comp_level: Option<u32>
    ) -> PyResult<()> {
       
        let mut writer = EmbeddingWriter::new(path, embeddings.vocab.as_ref(), comp_level)
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

/// Aligns an adhoc embedding to a transform embedding state, typically from Neighborhood
/// transforms
#[pyclass]
struct EmbeddingAligner {
    num_nodes: usize,
    random_nodes: usize
}

#[pymethods]
impl EmbeddingAligner {
    ///    Creates a new EmbeddingAligner.
    ///
    ///    This allows for adjusting an adhoc embedding which doesn't exist in the original graph.
    ///    It does this by first constructing a set of nearest nodes via the original node
    ///    embeddings, computing their relative distances, and then attempting to preserve them in
    ///    the transformed space.
    ///
    ///    The idea is that nodes that were close in the original space should still roughly be
    ///    close in the transformed space.
    ///
    ///    It uses approximate nearest neighbors to make it tractable.
    ///    
    ///    Parameters
    ///    ----------
    ///    num_nodes : Int
    ///        Closest K nodes to consider to learn mapping.
    ///    
    ///    random_nodes : Int - Optional
    ///        If provided, also samples random nodes for distances.
    ///
    ///        Default is 0.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[new]
    pub fn new(num_nodes: usize, random_nodes: Option<usize>) -> Self {
        EmbeddingAligner { num_nodes, random_nodes: random_nodes.unwrap_or(0) }
    }

    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("EmbeddingAligner<NumNodes={}, RandomNodes={}>", self.num_nodes, self.random_nodes)
    }

    ///    Transforms an embedding to the post neighborhood aligned space.
    ///    
    ///    Parameters
    ///    ----------
    ///    orig_embeddings : NodeEmbeddings
    ///        Original embeddings.
    ///    
    ///    orig_ann : EmbAnn
    ///        Approximate Nearest Neighbor instance to find nearest neighbors.
    ///    
    ///    translated_embeddings : NodeEmbeddings
    ///        Translated NodeEmbeddings.
    ///    
    ///    emb : Query
    ///        Embedding to use
    ///    
    ///    seed : Int - Optional
    ///        If provided, uses the random seed.  Default is global seed
    ///    
    ///    Returns
    ///    -------
    ///    List[Float] - Can throw exception
    ///        Returns the new embedding
    ///
    pub fn align(&self, 
        orig_embeddings: &NodeEmbeddings, 
        orig_ann: &EmbAnn,
        translated_embeddings: &NodeEmbeddings,
        emb: &Query,
        seed: Option<u64>
    ) -> PyResult<Vec<f32>> {
        let query_embedding = lookup_embedding(emb, translated_embeddings)?;
        
        // Get the original neighbors and distances
        let neighbors = orig_ann.ann.predict(&orig_embeddings.embeddings, query_embedding);
        let rand_neighbors = if self.random_nodes > 0 {
            let mut rng = XorShiftRng::seed_from_u64(seed.unwrap_or(SEED + 123123));

            let dist = Uniform::new(0, orig_embeddings.embeddings.len());
            (0..self.random_nodes).map(|_| (dist.sample(&mut rng), 0.)).collect()
        } else {
            Vec::with_capacity(0)
        };

        // Find the closest embeddings
        let n_embs: Vec<_> = neighbors.iter()
            .filter(|nd| {
                let (node_id, _dist) = nd.to_tup();
                if let Some((node_type, node_name)) = orig_embeddings.vocab.get_name(node_id) {
                    get_node_id(translated_embeddings.vocab.deref(), 
                           (*node_type).clone(), node_name.to_string()).is_ok()
                } else {
                    false
                }
            })
            .take(self.num_nodes)
            .map(|nd| nd.to_tup())
            .chain(rand_neighbors.into_iter())
            .map(|(node_id, _dist)| {

                let old_embedding = orig_embeddings.embeddings.get_embedding(node_id);
                let euc_dist = EDist::Euclidean.compute(&query_embedding, old_embedding);
                
                // Translate the node id from one vocab to the other
                let (node_type, node_name) = orig_embeddings.vocab.get_name(node_id)?;

                let node_id = get_node_id(translated_embeddings.vocab.deref(), 
                                          (*node_type).clone(), node_name.to_string()).ok()?;
                let emb = translated_embeddings.embeddings.get_embedding(node_id);
                Some((emb, euc_dist))
            }).filter(|x| x.is_some())
            .map(|x| x.unwrap())
            .collect();

        let new_emb = crate::algos::emb_aligner::align_embedding(&query_embedding, n_embs.as_slice(), 1e-1, 1e-2);

        Ok(new_emb)
    }

    ///    Transforms a set of embeddings to the post neighborhood aligned space.
    ///    
    ///    Parameters
    ///    ----------
    ///    orig_embeddings : NodeEmbeddings
    ///        Original embeddings.
    ///    
    ///    orig_ann : EmbAnn
    ///        Approximate Nearest Neighbor instance to find nearest neighbors.
    ///    
    ///    translated_embeddings : NodeEmbeddings
    ///        Translated NodeEmbeddings.
    ///    
    ///    emb : List[Query]
    ///        Set of embeddings to use
    ///    
    ///    seed : Int - Optional
    ///        If provided, uses the random seed.  Default is global seed
    ///    
    ///    Returns
    ///    -------
    ///    List[List[Float]] - Can throw exception
    ///        
    ///    
    pub fn bulk_align(&self, 
        orig_embeddings: &NodeEmbeddings, 
        orig_ann: &EmbAnn,
        translated_embeddings: &NodeEmbeddings,
        queries: Vec<Query>,
        seed: Option<u64>
    ) -> PyResult<Vec<Vec<f32>>> {
        let results: PyResult<Vec<_>> = queries.into_par_iter().map(|query| {
            self.align(orig_embeddings, orig_ann, translated_embeddings, &query, seed)
        }).collect();

        results
    }


}

/// Wrapper for the relatively crappy ANN solution.
#[pyclass]
struct GraphAnn {
    graph: Arc<CumCSR>,
    vocab: Arc<Vocab>,
    max_steps: usize
}

#[pymethods]
impl GraphAnn {
    ///    Creates a Graph Approximate Nearest Neighbor instance.  GraphANN uses random walks and
    ///    A* to climb areas in the graph that are likely to lead to neighbors relatively close to
    ///    the desired provided embedding.
    ///
    ///    It's ok.  It highly depends on the quality of the Graph and how likely nodes of
    ///    interest are homopholous.
    ///    
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph to explore
    ///    
    ///    max_steps : Int - Optional
    ///        Maximum number of steps in graph space to explore.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[new]
    pub fn new(graph: &Graph, max_steps: Option<usize>) -> Self {
        GraphAnn {
            graph: graph.graph.clone(),
            vocab: graph.vocab.clone(),
            max_steps: max_steps.unwrap_or(1000),
        }

    }

    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("GraphANN<MaxSteps={}>", self.max_steps)
    }

    ///    Attempts to find approximate nearest neighbors in graph space
    ///    
    ///    Parameters
    ///    ----------
    ///    query : Query
    ///        Embedding to look for. 
    ///    
    ///    embeddings : NodeEmbeddings
    ///        Set of embeddings defining a distance metric which is used for A*.
    ///    
    ///    k : Int
    ///        Top K items to return from the walk.
    ///    
    ///    seed : Int - Optional
    ///        If provided, uses the provided seed.  Otherwise uses the global seed.
    ///    
    ///    Returns
    ///    -------
    ///    List[(FQNode, f32)] - Can throw exception
    ///        Set of fully qualified nodes and associated scores.
    ///    
    pub fn find(
        &self, 
        query: &Query,
        embeddings: &NodeEmbeddings, 
        k: usize, 
        seed: Option<u64>
    ) -> PyResult<Vec<(FQNode, f32)>> {
        let query_embedding = lookup_embedding(query, embeddings)?;
        let seed = seed.unwrap_or(SEED + 10);
        let ann = algos::graph_ann::Ann::new(k, self.max_steps + k, seed);
        let nodes = ann.find(query_embedding, &(*self.graph), &embeddings.embeddings);
        Ok(convert_node_distance(&self.vocab, nodes))
    }
}

/// Wrapper for a much better ANN solution for embeddings
#[pyclass]
struct EmbAnn {
    ann: Ann
}

#[pymethods]
impl EmbAnn {

    ///    Creates an ANN on a set of node embeddings.  This uses the random projection trees to
    ///    construct a relatively fast ANN lookup.
    ///    
    ///    Parameters
    ///    ----------
    ///    embs : NodeEmbeddings
    ///        Node embedding set for building the ANN 
    ///    
    ///    n_trees : Int
    ///        Number of trees to build.  The more trees, the better the accuracy at the expense of
    ///        memory and compute.
    ///    
    ///    max_nodes_per_leaf : Int
    ///        Determines whether a node should split.  Lower max nodes per leaf creates deeper,
    ///        more accurate trees, at the expense of more memory.
    ///    
    ///    seed : Int - Optional
    ///        If provided, uses this seed for randomization.  Otherwise uses the global seed.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[new]
    pub fn new(
        embs: &NodeEmbeddings, 
        n_trees: usize,
        max_nodes_per_leaf: usize,
        seed: Option<u64>
    ) -> Self {
        let mut ann = Ann::new();
        let seed = seed.unwrap_or(SEED + 10);
        ann.fit(&embs.embeddings, n_trees, max_nodes_per_leaf, seed);

        EmbAnn { ann: ann }

    }

    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("EmbANN<N_Trees={}>", self.ann.num_trees())
    }

    ///    Find the nearest neighbors of a provided embedding using the EmbANN index.
    ///    
    ///    Parameters
    ///    ----------
    ///    embeddings : NodeEmbeddings
    ///        Embeddings used in constructing the trees.
    ///    
    ///    query : Query
    ///        Query of item to look for
    ///    
    ///    Returns
    ///    -------
    ///    List[(FQNode, f32)] - Can throw exception
    ///        List of fully qualified nodes and their associated distances.
    ///    
    pub fn find(
        &self, 
        embeddings: &NodeEmbeddings,
        query: &Query
    ) -> PyResult<Vec<(FQNode, f32)>> {
        let query_embedding = lookup_embedding(query, embeddings)?;
        let nodes = self.ann.predict(&embeddings.embeddings, query_embedding);
        Ok(convert_node_distance(&embeddings.vocab, nodes))
    }

    pub fn find_leaf_indices(
        &self, 
        query: Vec<f32>
    ) -> PyResult<Vec<usize>> {
        let nodes = self.ann.predict_leaf_indices(&query);
        Ok(nodes)
    }

    pub fn find_leaf_paths(
        &self, 
        query: Vec<f32>
    ) -> PyResult<Vec<Vec<usize>>> {
        let nodes = self.ann.predict_leaf_paths(&query);
        Ok(nodes)
    }

    pub fn depth(&self) -> Vec<usize> {
        self.ann.depth()
    }
}


///
/// Wrapper for the Supervised Monte-Carlo Iteration.  It stores the reward maps on the struct.
#[pyclass]
struct Smci {
    graph: Arc<CumCSR>,
    vocab: Arc<Vocab>,
    /// Maps Node 1 -> Node 2 = Reward
    rewards: Vec<(NodeID,NodeID,f32)>
}

#[pymethods]
impl Smci {
    #[new]
    pub fn new(graph: &Graph) -> Self {
        Smci {
            graph: graph.graph.clone(),
            vocab: graph.vocab.clone(),
            rewards: Vec::new()
        }
    }

    pub fn add_reward(
        &mut self, 
        from_node: FQNode, 
        to_node: FQNode, 
        reward: f32
    ) -> PyResult<()> {
        let f_n = get_node_id(self.vocab.deref(), from_node.0, from_node.1)?;
        let t_n = get_node_id(self.vocab.deref(), to_node.0, to_node.1)?;
        self.rewards.push((f_n, t_n, reward));
        Ok(())
    }

    pub fn optimize(
        &self, 
        iterations: usize,
        num_walks: usize,
        alpha: f32,
        discount: f32,
        step_penalty: f32,
        explore_pct: f32,
        restart_prob: f32,
        compression: Option<f32>,
        embeddings: Option<&NodeEmbeddings>,
        seed: Option<u64>
    ) -> PyResult<Graph> {
        let smci = SupervisedMCIteration {
            iterations,
            num_walks,
            alpha,
            discount,
            step_penalty,
            explore_pct,
            restart_prob,
            compression: compression.unwrap_or(1.0),
            seed: seed.unwrap_or(SEED)
        };

        let embs = embeddings.map(|e| {
            let tt = e.vocab.create_translation_table(self.vocab.deref());
            (&e.embeddings, tt)
        });
        let weights = smci.learn(self.graph.deref(), &self.rewards, embs);

        let new_graph = self.graph.clone_with_edges(weights)
            .expect("this is all internal, should just work");

        Ok(Graph {
            graph: Arc::new(new_graph),
            vocab: self.vocab.clone()
        })

    }
}

/// 
#[pyclass]
#[derive(Clone,Debug)]
struct PprRankLearner {
    alpha: f32,
    batch_size: usize,
    dims: usize,
    passes: usize,
    negatives: usize,
    loss: String,
    weight_decay: f32,
    steps: Steps,
    walks: usize,
    k: usize,
    num_features: Option<usize>,
    compression: f32,
    valid_pct: f32,
    beta: f32
}

#[pymethods]
impl PprRankLearner {

    ///    Createsa a PPR Rank Learner.  PPR Rank Learner optimizes similarities based on a nodes
    ///    local neighborhood:
    ///
    ///    1. For each node, compute their top K local neighborhood based on sampling, including
    ///       relative weight.
    ///    2. During optimization, sample N negatives
    ///    3. Optimize a listwise lost which pushes the local neighborhood higher than the randomly
    ///       sampled negatives.
    ///
    ///    This has some interesting dynamics.  Unlike EmbeddingPropagation, this captures
    ///    neighborhoods and local clusters instead of just immediate neighbors; this allows the
    ///    algorithm to better bias toward local topologies rather than basic connections.
    ///    It also is able to use ranking listwise metrics for optimization since the PPR estimate
    ///    gives an ordering to nodes.
    ///    
    ///    Parameters
    ///    ----------
    ///    alpha : Float
    ///        Learning Rate
    ///    
    ///    batch_size : Int
    ///        Batch size
    ///    
    ///    dims : Int
    ///        Dimension of the feature and eventual node embeddings.
    ///    
    ///    passes : Int
    ///        Number of passes to use
    ///    
    ///    steps : Float
    ///        Restart probability.  When steps ~ [0,1), it results in a restart probability.  When
    ///        steps ~ [1,inf], it is treated as a fixed length walk.
    ///    
    ///    walks : Int
    ///        Number of samples to collect.
    ///    
    ///    k : Int
    ///        Top K nearest neighbors according to PPR estimates.
    ///    
    ///    negatives : Int
    ///        Number of random negatives to use.
    ///    
    ///    loss : String - Optional
    ///        Supports 'listnet' and 'listmle'.
    ///    
    ///    compression : Float - Optional
    ///        This will either accentuate or depress the PPR scores for the local neighborhood.
    ///        Higher compressions will accentuate, lower compressions will depress.
    ///    
    ///    beta : Float - Optional
    ///        Determines how much to bias toward community structure versus rarer nodes.
    ///
    ///        Default is 0.8
    ///    
    ///    num_features : Int - Optional
    ///        How many features to use to construct nodes.  If provided, will randomly sample
    ///        features each construction.
    ///
    ///        Default is all
    ///    
    ///    weight_decay : Float - Optional
    ///        If provided, applies weight decar to feature embeddings.
    ///
    ///        Default is 0.
    ///    
    ///    valid_pct : Float - Optional
    ///        If provided, uses a subset of the graph for validation.
    ///    
    ///    Returns
    ///    -------
    ///    Self - Can throw exception
    ///        
    ///    
    #[new]
    fn new(
         // Learning rate
        alpha: f32, 
 
        // Batch size 
        batch_size: usize, 

        // Node embedding size
        dims: usize,

        // Number of passes to run
        passes: usize,

        // Number of steps to take
        steps: f32,

        // Number of walks per node
        walks: usize,

        // Number of neighbors to extract from random walk
        k: usize,

        // Number of  negatives, produced from random walks.  The quality of these deeply
        // depend on the quality of the graph
        negatives: usize,

        // Loss to use
        loss: Option<String>,

        // The PPR scores are either accentuated or depressed by the compression factor
        compression: Option<f32>,
        
        // How much to benefit rarer/popular items
        beta: Option<f32>,

        num_features: Option<usize>,

        weight_decay: Option<f32>, 

        // Percentage of nodes to use for validation
        valid_pct: Option<f32>,

    ) -> PyResult<Self> {

        let steps = if steps >= 1. {
            Steps::Fixed(steps as usize)
        } else if steps > 0. {
            Steps::Probability(steps)
        } else {
            return Err(PyValueError::new_err("Alpha must be between [0, inf)"))
        };

        Ok(PprRankLearner {
            alpha,
            batch_size,
            dims,
            passes,
            steps,
            negatives,
            walks,
            k,
            num_features,
            weight_decay: weight_decay.unwrap_or(0f32),
            loss: loss.unwrap_or_else(||"listnet".into()),
            compression: compression.unwrap_or(1f32),
            valid_pct: valid_pct.unwrap_or(0.1),
            beta: beta.unwrap_or(0.8)
        })
    }
    
    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    ///    Learns the feature embeddings for a given graph.
    ///    
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph to use for construction.
    ///    
    ///    features : FeatureSet
    ///        FeatureSet mapping nodes -> features
    ///    
    ///    feature_embeddings : mut NodeEmbeddings - Optional
    ///        If provided, uses the feature_embeddings passed in as a starting point.  Otherwise,
    ///        creates a new randomly initialized NodeEmbedding.
    ///
    ///        Default is new.
    ///    
    ///    indicator : Bool - Optional
    ///        If provided, shows an indicator.
    ///
    ///        Default is True
    ///    
    ///    seed : Int - Optional
    ///        If provided, uses the seed for all stochastic processes.
    ///
    ///        Default is a fixed global seed.
    ///    
    ///    Returns
    ///    -------
    ///    NodeEmbeddings
    ///        Learned Feature Embeddings.
    ///    
    pub fn learn_features(
        &self, 
        graph: &Graph, 
        features: &mut FeatureSet,
        feature_embeddings: Option<&mut NodeEmbeddings>,
        indicator: Option<bool>,
        seed: Option<u64>
    ) -> NodeEmbeddings {

        let loss = if self.loss == "listnet" {
            PprLoss::ListNet { passive: true, weight_decay: self.weight_decay }
        } else {
            PprLoss::ListMLE { weight_decay: self.weight_decay }
        };

        let ppr_rank = PprRank {
            alpha: self.alpha,
            batch_size: self.batch_size,
            dims: self.dims,
            passes: self.passes,
            steps: self.steps,
            loss: loss,
            negatives: self.negatives,
            num_walks: self.walks,
            beta: self.beta,
            k: self.k,
            num_features: self.num_features,
            compression: self.compression,
            valid_pct: self.valid_pct,
            indicator: indicator.unwrap_or(true),
            seed: seed.unwrap_or(SEED)
        };

        features.features.fill_missing_nodes();

        // Pull out the EmbeddingStore
        let feature_embeddings = feature_embeddings.map(|fes| {
           let mut sfes = EmbeddingStore::new(fes.vocab.len(), 0, EDist::Cosine);
           std::mem::swap(&mut sfes, &mut fes.embeddings);
           sfes
        });

        let feat_embeds = ppr_rank.learn(&*graph.graph, &features.features, feature_embeddings);
        let vocab = features.features.clone_vocab();

        let feature_embeddings = NodeEmbeddings {
            vocab: Arc::new(vocab),
            embeddings: feat_embeds};

        feature_embeddings
    }

}

/// Learns VPCG vectors on a graph.
#[pyclass]
struct VpcgEmbedder {
    max_terms: usize, 
    passes: usize,
    dims: usize,
    alpha: f32,
    err: f32
}

#[pymethods]
impl VpcgEmbedder {
    ///    Creates a VPCGEmbedder.  VPCG uses bipartite graphs with features to learn embeddings
    ///    that attempt to bridge the semantic gap.  It is highly effective on Search and
    ///    Recommendation problems.
    ///
    ///    The graph needs to be a bipartite graph where one side of the graph has a distinct type
    ///    from the other side.
    ///    
    ///    Parameters
    ///    ----------
    ///    max_terms : Int
    ///        Max terms to store on each node during propagation.
    ///    
    ///    passes : Int
    ///        Number of passes to perform.  A good starting point is 10
    ///    
    ///    dims : Int
    ///        Number of dimensions for the final NodeEmbeddings.
    ///    
    ///    alpha : Float - Optional
    ///        If provided, alpha ~ [0,1] allows nodes to reinforce their own features over the propagated
    ///        features.  Higher alphas rely more on propagated features whereas lower alphas
    ///        attempt to preserve the original feature set more.
    ///
    ///        Default is 1.
    ///    
    ///    err : Float - Optional
    ///        Suppresses terms with a weight of less than err.  Default is 1e-5
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[new]
    pub fn new(
        max_terms: usize, 
        passes: usize, 
        dims: usize, 
        alpha: Option<f32>, 
        err: Option<f32>
    ) -> Self {
        VpcgEmbedder { 
            max_terms, 
            passes, 
            dims,
            alpha: alpha.unwrap_or(1f32),
            err: err.unwrap_or(1e-5f32)
        }
    }

    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("VPCGEmbedder<MaxTerms={}, Passes={}, Dims={}, Alpha={}, Err={}>",
                self.max_terms, self.passes, self.dims, self.alpha, self.err)
    }

    ///    Learns VPCG embeddings on the graph.
    ///    
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph to use for bipartite structure
    ///    
    ///    features : FeatureSet
    ///        Features to propagate between nodes
    ///    
    ///    start_node_type : String
    ///        Starting node type for propagation.
    ///    
    ///    Returns
    ///    -------
    ///    NodeEmbeddings
    ///        New NodeEmbeddings capturing the VPCG Embeddings.
    ///    
    pub fn learn(&self, 
        graph: &Graph, 
        features: &mut FeatureSet,
        start_node_type: String
    ) -> NodeEmbeddings  {
        let (mut left, mut right) = (Vec::new(), Vec::new());
        for node_id in 0..graph.graph.len() {
            let nt = graph.vocab.get_node_type(node_id).unwrap();
            if nt.as_ref() == &start_node_type {
                left.push(node_id);
            } else {
                right.push(node_id)
            }
        }

        features.features.fill_missing_nodes();
        let vpcg = crate::algos::vpcg::VPCG {
            max_terms: self.max_terms, 
            iterations: self.passes,
            dims: self.dims,
            alpha: self.alpha,
            err: self.err
        };
        let embs = vpcg.learn(graph.graph.as_ref(), &features.features, (&left, &right));
        let node_embeddings = NodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings:embs 
        };

        node_embeddings 
    }

}

/// Computes embeddings from features using PPR
#[pyclass]
struct PPREmbedder {
    dims: usize,
    num_walks: usize,
    steps: f32,
    beta: f32,
    eps: f32
}

#[pymethods]
impl PPREmbedder {
    ///    Creates a PPREmbedder.  PPREmbedder constructs the local personal page rank for a node
    ///    and then performs a weighted blend of those features to construct a new embedding.
    ///    
    ///    Parameters
    ///    ----------
    ///    dims : Int
    ///        Dimension of the embedding space
    ///    
    ///    num_walks : Int
    ///        Number of walks for sampling the local neighborhood
    ///    
    ///    steps : Float
    ///        if steps ~ [0,1), uses restart probabilities to terminate. if steps ~[1, inf], uses
    ///        fixed length walks.
    ///    
    ///    beta : Float - Optional
    ///        Beta parameter to suppress higher degree nodes.
    ///
    ///        Default is 0.8
    ///    
    ///    eps : Float - Optional
    ///        Suppresses terms under eps.  
    ///
    ///        Default is 1e-5
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[new]
    pub fn new(
        dims: usize,
        num_walks: usize, 
        steps: f32, 
        beta: Option<f32>, 
        eps: Option<f32>
    ) -> Self {
        PPREmbedder { 
            dims,
            num_walks,
            steps,
            beta: beta.unwrap_or(0.8),
            eps: eps.unwrap_or(1e-5)
        }
    }

    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("PPREmbedder<Dims={}, NumWalks={}, Steps={}, Beta={}, EPS={}>",
                self.dims, self.num_walks, self.steps, self.beta, self.eps)
    }

    ///    Constructs Node Embeddings.
    ///    
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph to use.
    ///    
    ///    features : FeatureSet
    ///        FeatureSet with original features for propagation.
    ///    
    ///    seed : Int - Optional
    ///        If provided, uses the random seed.  Otherwise, uses global seed.
    ///    
    ///    Returns
    ///    -------
    ///    NodeEmbeddings - Can throw exception
    ///    
    pub fn learn(&self, 
        graph: &Graph, 
        features: &mut FeatureSet,
        seed: Option<u64>
    ) -> PyResult<NodeEmbeddings> {
        features.features.fill_missing_nodes();

        let steps = if self.steps >= 1. {
            Steps::Fixed(self.steps as usize)
        } else if self.steps > 0. {
            Steps::Probability(self.steps)
        } else {
            return Err(PyValueError::new_err("Alpha must be between [0, inf)"))
        };

        let embedder = PPREmbed {
            dims: self.dims,
            num_walks: self.num_walks,
            steps: steps,
            beta: self.beta,
            eps: self.eps,
            seed: seed.unwrap_or(SEED)
        };

        let embs = embedder.learn(graph.graph.as_ref(), &features.features);
        
        let node_embeddings = NodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings:embs 
        };

        Ok(node_embeddings)
    }

}

/// Computes node embeddings using InstantEmbeddings
#[pyclass]
struct InstantEmbeddings {
    dims: usize,
    hashes: usize,
    estimator: Estimator
}

#[pymethods]
impl InstantEmbeddings {

    ///    Creates an InstantEmbeddings instance.  Instance embeddings construct topological
    ///    embeddings in graphs by using random walks to capturing local neighborhoods.  They're
    ///    fast, scalable, and perform well in practice.
    ///
    ///    This variant uses sampling to construct the local neighborhood, which can be faster when
    ///    nodes exist with high degree counts.
    ///    
    ///    Parameters
    ///    ----------
    ///    dims : Int
    ///        Dimension of the InstantEmbeddings.
    ///    
    ///    hashes : Int
    ///        Number of hashes to use for mapping a Node to embedding space.
    ///    
    ///    num_walks : Int
    ///        Number of walks to perform for sampling InstantEmbeddings.
    ///    
    ///    steps : Float
    ///        If steps ~ [0,1), uses restart probabilities to terminate. if steps ~[1, inf], uses
    ///        fixed length walks.
    ///    
    ///    beta : Float - Optional
    ///        Beta parameter to suppress higher degree nodes.
    ///
    ///        Default is 0.8
    ///    
    ///    seed : Int - Optional
    ///        
    ///    
    ///    Returns
    ///    -------
    ///    Self - Can throw exception
    ///        
    ///    
    #[staticmethod]
    pub fn random_walk(
        dims: usize,
        hashes: usize,
        num_walks: usize, 
        steps: f32, 
        beta: Option<f32>,
        seed: Option<u64>
    ) -> PyResult<Self> {
        let steps = if steps >= 1. {
            Steps::Fixed(steps as usize)
        } else if steps > 0. {
            Steps::Probability(steps)
        } else {
            return Err(PyValueError::new_err("Steps must be between [0, inf)"))
        };

        let ie = InstantEmbeddings { 
            dims,
            hashes,
            estimator: Estimator::RandomWalk {
                walks: num_walks,
                steps,
                beta: beta.unwrap_or(0.8),
                seed: seed.unwrap_or(SEED)
            }
        };
        Ok(ie)
    }

    ///    Creates an InstantEmbedding implementation using a sparse PPR estimation which is
    ///    deterministic.  This is usually faster (assuming a reasonable EPS) when the degree count
    ///    of the nodes are relatively low; when nodes have large numbers of degrees, it can be
    ///    substantially slower than sampling based approaches.
    ///    
    ///    Parameters
    ///    ----------
    ///    dims : Int
    ///        Dimension of the InstantEmbeddings.
    ///    
    ///    hashes : Int
    ///        Number of hashes to use for mapping a Node to embedding space.
    ///    
    ///    steps : Float
    ///        Restart probability for random walks.
    ///    
    ///    eps : Float - Optional
    ///        Tolerable error in PPR estimates.  
    ///
    ///        Default is 1e-5
    ///    
    ///    Returns
    ///    -------
    ///    Self - Can throw exception
    ///        
    ///    
    #[staticmethod]
    pub fn sparse_ppr(
        dims: usize,
        hashes: usize,
        steps: f32, 
        eps: Option<f32>
    ) -> PyResult<Self> {
        if steps <= 0f32 || steps >= 1f32 {
            return Err(PyValueError::new_err("Steps must be between (0, 1)"))
        }

        let ie = InstantEmbeddings { 
            dims,
            hashes,
            estimator: Estimator::SparsePPR {
                p: steps,
                eps: eps.unwrap_or(1e-5)
            }
        };

        Ok(ie)
    }

    ///    Learns Instant Embeddings.
    ///    
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph to learn instant embeddings on.
    ///    
    ///    Returns
    ///    -------
    ///    NodeEmbeddings - Can throw exception
    ///        
    ///    
    pub fn learn(&self, 
        graph: &Graph, 
    ) -> PyResult<NodeEmbeddings> {

        let embedder = IE {
            dims: self.dims,
            hashes: self.hashes,
            estimator: self.estimator
        };

        let embs = embedder.learn(graph.graph.as_ref());
        
        let node_embeddings = NodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings:embs 
        };

        Ok(node_embeddings)
    }

    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("InstantEmbeddings<Dims={}, Hashes={}, Estimator={:?}>",
                self.dims, self.hashes, self.estimator)
    }

}

#[pyclass]
struct TournamentBuilder {
    gb: GraphBuilder,
    degrees: Vec<f32>
}

#[pymethods]
impl TournamentBuilder {
    ///    Creates a new tournament builder.  This will create a special graph which can be used to
    ///    compute Plakett-luce models.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[new]
    pub fn new() -> Self {
        TournamentBuilder {
            gb: GraphBuilder::new(),
            degrees: Vec::new()
        }
    }

    ///    Adds an outcome with a winner and a loser as well as its associated weight.  
    ///    
    ///    Parameters
    ///    ----------
    ///    winner : FQNode
    ///        Winner node.
    ///    
    ///    loser : FQNode
    ///        Loser node
    ///    
    ///    weight : Float
    ///        Relative weight
    ///    
    pub fn add_outcome(
        &mut self,
        winner: FQNode,
        loser: FQNode,
        weight: f32
    ) {
        let mut v_len = self.gb.vocab.len();
        let d = weight / 2f32;
        // Add edge from loser to winner
        self.gb.add_edge( loser.clone(), winner, d, EdgeType::Directed);

        
        while v_len < self.gb.vocab.len() {
            self.degrees.push(1f32);
            v_len += 1;
        }
        
        // Update norm
        let node_id = self.gb.vocab.get_node_id(loser.0, loser.1)
            .expect("Should never fail!");

        self.degrees[node_id] += d;
    }

    ///    Adds a ordered list where teams in earlier positions out competed teams in later
    ///    positions.  For example, [Racer1, Racer3, Racer2], gets contructed as:
    ///    1. Racer1 > Racer3
    ///    2. Racer1 > Racer2
    ///    3. Racer3 > Racer1
    ///    
    ///    Parameters
    ///    ----------
    ///    order : List[FQNode]
    ///        Ordered list of fully qualified nodes where position demarcates ranking in
    ///        comparison to other nodes in the list.
    ///    
    ///    weight : Float
    ///        Overall weight to give list
    ///    
    pub fn add_ranked_outcomes(
        &mut self,
        order: Vec<FQNode>,
        weight: f32
    ) {
        for i in 0..order.len() {
            for j in 1..order.len() {
                self.add_outcome(order[i].clone(), order[j].clone(), weight)
            }
        }
    }


    ///    Builds the tournament.
    ///    
    ///    Returns
    ///    -------
    ///    Tournament - Optional
    ///        Tournament object capturing orderings.
    ///    
    pub fn build(
        &mut self
    ) -> Option<Tournament> {
        if let Some(graph) = self.gb.build_graph() {
            let mut degrees = Vec::with_capacity(0);
            std::mem::swap(&mut degrees, &mut self.degrees);
            let es = EmbeddingStore::new_with_vec(graph.nodes(), 1, EDist::Euclidean, degrees)
                .expect("Should be correct");

            let embs = NodeEmbeddings {
                vocab: graph.vocab.clone(),
                embeddings: es
            };
            Some(Tournament { graph: graph, norms: embs })
        } else {
            None
        }
    }

    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("TournamentBuilder<Nodes={}, Outcomes={}>",
                self.gb.vocab.len(), self.gb.edges.len())
    }

}

#[pyclass]
struct Tournament {
    graph: Graph,
    norms: NodeEmbeddings
}

#[pymethods]
impl Tournament {

    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("Tournament<Nodes={}, Outcomes={}>",
                self.graph.graph.len(), self.graph.graph.edges())
    }

}

#[pyclass]
struct LSR {
    passes: usize
}

#[pymethods]
impl LSR {

    ///    Initializes a Luce Spectral Ranker instance.  Luce Spectral Ranking, or LSR, is a method
    ///    for learning placket-luce models from a given tournament. It's fast, scalable, and can
    ///    recover the original parameters with low error.
    ///    
    ///    Parameters
    ///    ----------
    ///    passes : Int
    ///        
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[new]
    pub fn new(passes: usize) -> Self {
        LSR { passes }
    }
    
    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("LSR<Passes={}>", self.passes)
    }

    ///    Fits a Placket-Luce model to the provided tournament.
    ///    
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph representing the markovian state used by LSR models.  This graph is
    ///        constructed via the TournamentBuilder instance.
    ///    
    ///    team_norms : NodeEmbeddings
    ///        Captures the 
    ///    
    ///    indicator : Bool - Optional
    ///        
    ///    
    ///    Returns
    ///    -------
    ///    NodeEmbeddings 
    ///        Logits for rankings
    ///    
    pub fn learn(
        &self,
        tournament: &Tournament,
        indicator: Option<bool>
    ) -> NodeEmbeddings {
        let g = tournament.graph.graph.as_ref();
        let norms = &tournament.norms.embeddings;
        let lsr = ALSR { passes: self.passes };
        let scores = lsr.compute(g, norms, indicator.unwrap_or(true));
        // Find the page rank of the tournament graph

        let embs = EmbeddingStore::new(g.len(), 1, EDist::Euclidean);
        scores.par_iter().enumerate().for_each(|(node_id, score)| {
            let e1 = embs.get_embedding_mut_hogwild(node_id);
            e1[0] = *score;
        });

        let node_embeddings = NodeEmbeddings {
            vocab: tournament.graph.vocab.clone(),
            embeddings: embs 
        };

        node_embeddings
       
    }
}

#[pyclass]
struct ConnectedComponents {}

#[pymethods]
impl ConnectedComponents {

    ///    Computes the graph, looking for connected components.  This is only guaranteed to work
    ///    correctly for undirected graphs.
    ///    
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph to find connected components in
    ///    
    ///    Returns
    ///    -------
    ///    NodeEmbeddings
    ///        NodeEmbeddings of 1 dimension where the value of the embedding references the
    ///        component id.  All nodes sharing the same component id are connected.
    ///    
    #[staticmethod]
    pub fn learn(graph: &Graph) -> NodeEmbeddings {
        let es = find_connected_components(graph.graph.as_ref());
        NodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings: es
        }
    }
}

#[pyclass]
struct RandomPath {
    rng: XorShiftRng
}

#[pymethods]
impl RandomPath {

    #[new]
    pub fn new(seed: Option<u64>) -> Self {
        let rng = XorShiftRng::seed_from_u64(seed.unwrap_or(SEED + 125));
        RandomPath { rng }
    }
    
    ///    Performs random walks on a graph and returns the full path of nodes visited.
    ///    
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph to find connected components in
    ///
    ///    node : FQNode
    ///        Start Node for the random walks
    ///
    ///    counts : Positive Int
    ///        Number of random walks to run
    ///
    ///    restarts : Float
    ///        restarts ~ (0,1), determines the probability a random walk will terminate with
    ///        lower restarts resulting in longer walks.
    ///    
    ///    weighted : Bool 
    ///        Weighted versus unweighted sampling.
    ///        
    ///    
    ///    Returns
    ///    -------
    ///    List[List[FQNode]]
    ///        Provides a list of random walks with nodes traversed in order until termination.
    ///    
    pub fn rollout<>(
        &mut self,
        graph: &Graph, 
        node: FQNode,
        count: usize,
        restarts: f32, 
        weighted: bool,
    ) -> PyResult<Vec<Vec<FQNode>>> {
        let steps = Steps::from_float(restarts)
            .ok_or_else(|| PyValueError::new_err("restarts must be between [0, inf)"))?;
        //let sampler: Box<dyn Sampler<_>> = if weighted { Box::new(Weighted) } else { Box::new(Unweighted) };
        let g = graph.graph.as_ref();

        let mut outputs: Vec<Vec<NodeID>> = vec![Vec::new(); count];
        let rngs: Vec<_> = (0..count as u64)
            .map(|i| XorShiftRng::seed_from_u64(self.rng.gen::<u64>() + i))
            .collect();

        let vocab = graph.vocab.deref(); 
        let node_id = get_node_id(vocab, node.0, node.1)?;
        outputs.par_iter_mut().zip(rngs.into_par_iter()).for_each(|(walk, mut rng)| {
            walk.push(node_id);
            if weighted {
                rollout(g, steps, &Weighted, node_id, &mut rng, walk);
            } else {
                rollout(g, steps, &Weighted, node_id, &mut rng, walk);
            };
        });

        let it = outputs.into_iter().map(|walk| walk.into_iter().map(|node_id| {
            convert_node_id_to_fqn(vocab, node_id)
        }).collect());

        Ok(it.collect())
    }

}


/// Helper method for looking up an embedding.
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

fn convert_node_distance(
    vocab: &Vocab, 
    dists: Vec<NodeDistance>
) -> Vec<(FQNode, f32)> {
    dists.into_iter()
        .map(|n| {
            let (node_id, dist) = n.to_tup();
            let (node_type, name) = vocab.get_name(node_id)
                .expect("Can't find node id in vocab!");
            (((*node_type).clone(), name.to_string()), dist)
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
    m.add_class::<GraphAnn>()?;
    m.add_class::<EmbAnn>()?;
    m.add_class::<FeatureSet>()?;
    m.add_class::<FeaturePropagator>()?;
    m.add_class::<NodeEmbedder>()?;
    m.add_class::<FeatureAggregator>()?;
    m.add_class::<Query>()?;
    m.add_class::<RandomWalker>()?;
    m.add_class::<BiasedRandomWalker>()?;
    m.add_class::<SparsePPR>()?;
    m.add_class::<NeighborhoodAligner>()?;
    m.add_class::<EmbeddingAligner>()?;
    m.add_class::<PprRankLearner>()?;
    m.add_class::<PageRank>()?;
    m.add_class::<Smci>()?;
    m.add_class::<VpcgEmbedder>()?;
    m.add_class::<PPREmbedder>()?;
    m.add_class::<InstantEmbeddings>()?;
    m.add_class::<LSR>()?;
    m.add_class::<TournamentBuilder>()?;
    m.add_class::<Tournament>()?;
    m.add_class::<ConnectedComponents>()?;
    m.add_class::<ListenerRule>()?;
    m.add_class::<LossWeighting>()?;
    m.add_class::<RandomPath>()?;
    Ok(())
}

