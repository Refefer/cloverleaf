//! Python bindings, built with the `python` feature.
//!
//! The doc comments on the classes below are numpy-style Python docstrings: pyo3 exposes them as
//! `__doc__`.

use std::sync::Arc;
use std::ops::Deref;
use std::collections::HashSet;

use rayon::prelude::*;
use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError,PyIOError,PyKeyError,PyIndexError,PyTypeError,PyRuntimeError};
use rand::prelude::*;
use rand_xorshift::XorShiftRng;

use crate::graph::{CumCSR,Graph as CGraph,NodeID};
use crate::vocab::Vocab;
use crate::sampler::{Weighted,Unweighted};
use crate::distance::{Distance as EDist};
use crate::embeddings::{EmbeddingStore,DimensionMismatch};
use crate::feature_store::FeatureSet as CoreFeatureSet;
use crate::named_graph::NamedGraph;
use crate::feature_store::FeatureNamespace as CoreFeatureNamespace;
use crate::embeddings::NodeEmbeddingsBuilder as CoreNodeEmbeddingsBuilder;
use crate::algos::aggregator::{FeatureAggregator as CoreFeatureAggregator, NodeEmbedder as CoreNodeEmbedder};
use crate::algos::emb_aligner::EmbeddingAligner as CoreEmbeddingAligner;
use crate::embeddings::NodeEmbeddings as CoreNodeEmbeddings;
use crate::io::{read_node_values,EmbeddingLoadError,FeatureFileError,GraphLoadError};

use crate::algos::alignment::{NeighborhoodAligner as NA};
use crate::algos::ann::Ann;
use crate::algos::connected::find_connected_components;
use crate::algos::ep::attention::{AttentionType,MultiHeadedAttention};
use crate::algos::ep::{EmbeddingPropagation,EmbeddingInit,LearningRateSchedule,UpdateFrequency,LossWeighting as EPLW};
use crate::algos::ep::loss::Loss;
use crate::algos::ep::model::{AveragedFeatureModel,AttentionFeatureModel};
use crate::algos::feat_propagation::propagate_features;
use crate::algos::graph_ann::NodeDistance;
use crate::algos::grwr::{Steps as GSteps,GuidedRWR};
use crate::algos::instantembedding::{InstantEmbeddings as IE,Estimator};
use crate::algos::lsr::{LSR as ALSR, TournamentBuilder as CoreTournamentBuilder, Tournament as CoreTournament};
use crate::algos::pprembed::PPREmbed;
use crate::algos::pprrank::{PprRank, Loss as PprLoss};
use crate::algos::grad_utils::optimizer::{OptimizerKind, WeightDecay};
use crate::algos::policy_evaluation::{PolicyEvaluation as PEA};
use crate::algos::reweighter::{Reweighter};
use crate::algos::rwr::{RWR,ppr_estimate,rollouts};
use crate::algos::smci::SupervisedMCIteration;
use crate::algos::utils::{Sample, top_k_scores};
use crate::error::Error;
use crate::named_graph::{GraphBuilder as CoreGraphBuilder, GraphLoadOptions};
use crate::algos::vpcg::{VPCG, FeatureWeight as VFeatureWeight};


impl From<Error> for PyErr {
    fn from(e: Error) -> PyErr {
        match e {
            Error::Io(_) => PyIOError::new_err(e.to_string()),
            Error::UnknownNode { .. } => PyKeyError::new_err(e.to_string()),
            Error::NodeIdOutOfRange { .. } => PyIndexError::new_err(e.to_string()),
            Error::GraphLoad(e) => e.into(),
            Error::EmbeddingLoad(e) => e.into(),
            Error::FeatureFile(e) => e.into(),
            Error::VocabMismatch(_) | Error::DimensionMismatch(_) | Error::Malformed { .. }
                | Error::InvalidArgument(_) => PyValueError::new_err(e.to_string())
        }
    }
}

impl From<GraphLoadError> for PyErr {
    fn from(e: GraphLoadError) -> PyErr {
        match e {
            GraphLoadError::Io(_) => PyIOError::new_err(e.to_string()),
            _ => PyValueError::new_err(e.to_string())
        }
    }
}

impl From<EmbeddingLoadError> for PyErr {
    fn from(e: EmbeddingLoadError) -> PyErr {
        match e {
            EmbeddingLoadError::Io(_) => PyIOError::new_err(e.to_string()),
            EmbeddingLoadError::DuplicateNode(_) => PyKeyError::new_err(e.to_string()),
            EmbeddingLoadError::Parse(_) | EmbeddingLoadError::DimensionMismatch =>
                PyValueError::new_err(e.to_string())
        }
    }
}

impl From<FeatureFileError> for PyErr {
    fn from(e: FeatureFileError) -> PyErr {
        match e {
            FeatureFileError::Io(_) => PyIOError::new_err(e.to_string()),
            FeatureFileError::Malformed(_) => PyValueError::new_err(e.to_string())
        }
    }
}

/// Validates a warm start, then takes its store out of the Python object, which is left with
/// zero-dimensional embeddings: the training loop needs the store by value, and copying it could
/// double the memory used.
fn take_warm_start(
    graph: &NamedGraph,
    features: &mut CoreFeatureSet,
    feature_embeddings: Option<&mut NodeEmbeddings>
) -> PyResult<Option<CoreNodeEmbeddings>> {
    features.prepare_for_learning(graph, feature_embeddings.as_deref().map(|f| &f.inner))?;
    Ok(feature_embeddings.map(|fes| {
        let empty = EmbeddingStore::new(fes.vocab.len(), 0, EDist::Cosine);
        let store = std::mem::replace(&mut fes.embeddings, empty);
        CoreNodeEmbeddings { vocab: fes.vocab.clone(), embeddings: store }
    }))
}

/// Defines a constant seed for use when a seed is not provided.  This is specifically hardcoded to
/// allow for deterministic performance across all algorithms using any stochasticity.
const SEED: u64 = crate::DEFAULT_SEED;

/// Simplifies a lot of the type signatures
type FQNode = (String, String);

/// Maps an iterator of node ids and scores back to their pretty names with optional top K and
/// filtering by node types.
fn convert_scores(
    vocab: &Vocab, 
    scores: impl Iterator<Item=(NodeID, f32)>, 
    k: Option<usize>,
    filtered_node_type: Option<HashSet<String>>
) -> Vec<(FQNode, f32)> {
    let top = match &filtered_node_type {
        Some(nts) => top_k_scores(scores, k, |id| vocab.has_node_type(id, nts)),
        None => top_k_scores(scores, k, |_id| true)
    };
    top.into_iter().map(|(node_id, w)| (convert_node_id_to_fqn(vocab, node_id), w)).collect()
}

fn convert_node_id_to_fqn(
    vocab: &Vocab,
    node_id: NodeID
) -> FQNode {
    vocab.get_name_owned(node_id).expect("node ids come from this vocab")
}


/// Convenience method for getting an internal node id from pretty name
/// Errors unless node ids from `a` index the same nodes in `b`.
fn ensure_aligned(a: &Vocab, b: &Vocab, what: &'static str) -> PyResult<()> {
    Ok(crate::error::ensure_aligned(a, b, what)?)
}

fn dimension_error(e: DimensionMismatch) -> PyErr {
    Error::from(e).into()
}

/// Resolves a Python-style index, where negative values count back from the end.
fn resolve_index(mut idx: isize, len: usize) -> Option<usize> {
    let len = len as isize;
    if idx < 0 {
        idx += len;
    }
    if idx < 0 || idx >= len {
        None
    } else {
        Some(idx as usize)
    }
}

fn get_node_id<A: AsRef<str>, B: AsRef<str>>(
    vocab: &Vocab, 
    node_type: A,
    node_name: B 
) -> PyResult<NodeID> {
    Ok(vocab.node_id(node_type.as_ref(), node_name.as_ref())?)
}


#[derive(Clone)]
enum QueryType {
    Node(String, String),
    Embedding(Vec<f32>),
    NodeID(NodeID)
}

/// Type of Query to issue: a direct node lookup or an embedding
#[pyclass(from_py_object)]
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

    ///    Creates a Query using a referenced NodeID
    ///
    ///    Parameters
    ///    ----------
    ///    node_id :  Integer
    ///        NodeID
    ///
    ///    Returns
    ///    -------
    ///    Query
    #[staticmethod]
    pub fn node_id(
        node_id: NodeID
    ) -> Self {
        Query { qt: QueryType::NodeID(node_id) }
    }
 
}


///    Contains the set of distance metrics used by NodeEmbeddings, typically application
///    dependent.
///
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone,PartialEq)]
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
    inner: NamedGraph
}

impl From<NamedGraph> for Graph {
    fn from(inner: NamedGraph) -> Self {
        Graph { inner }
    }
}

impl Deref for Graph {
    type Target = NamedGraph;
    fn deref(&self) -> &NamedGraph {
        &self.inner
    }
}

impl std::ops::DerefMut for Graph {
    fn deref_mut(&mut self) -> &mut NamedGraph {
        &mut self.inner
    }
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
        self.inner.contains_node(&name.0, &name.1)
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
    #[pyo3(signature = (node, normalized=None))]
    pub fn get_edges(&self, node: FQNode, normalized: Option<bool>) -> PyResult<(Vec<FQNode>, Vec<f32>)> {
        let (edges, weights) = self.inner.get_edges(&node.0, &node.1, normalized.unwrap_or(false))?;
        let names = edges.iter().map(|id| convert_node_id_to_fqn(&self.vocab, *id)).collect();
        Ok((names, weights))
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
    #[pyo3(signature = (path, comp_level=None))]
    pub fn save(&self, path: &str, comp_level: Option<u32>) -> PyResult<()> {
        Ok(self.inner.save(path, comp_level)?)
    }
    
    /**
     * Transposes a graph, swapping directions of the edges and associated weights, returning a new
     * graph.
     *
     *   Returns
     *    -------
     *    Graph
     *      A new graph constructed from the transpose of an existing graph
     **/
    pub fn transpose(&self) -> Graph {
        Graph::from(self.inner.transpose())
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
    #[pyo3(signature = (path, edge_type, chunk_size=None, skip_rows=None, weighted=None, deduplicate=None))]
    pub fn load(
        py: Python<'_>,
        path: &str, 
        edge_type: EdgeType, 
        chunk_size: Option<usize>,
        skip_rows: Option<usize>,
        weighted: Option<bool>,
        deduplicate: Option<bool>
    ) -> PyResult<Self> {
        let d = GraphLoadOptions::default();
        let options = GraphLoadOptions {
            chunk_size: chunk_size.unwrap_or(d.chunk_size),
            skip_rows: skip_rows.unwrap_or(d.skip_rows),
            weighted: weighted.unwrap_or(d.weighted),
            deduplicate: deduplicate.unwrap_or(d.deduplicate)
        };
        py.detach(move || Ok(Graph::from(NamedGraph::load(path, edge_type.into(), &options)?)))
    }

}

/// Basic RP3b walker
#[pyclass]
#[derive(Clone)]
struct RandomWalker {
    restarts: Sample,
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
    #[pyo3(signature = (restarts, walks, beta=None))]
    fn new(restarts: f32, walks: usize, beta: Option<f32>) -> PyResult<Self> {
        // Reject 0 and NaN explicitly: neither is a meaningful restart probability or walk length.
        if !(restarts > 0f32) {
            return Err(PyValueError::new_err("restarts must be in (0, 1) or [1, inf)"))
        }
        Sample::new(restarts)
            .map_err(|_err| PyValueError::new_err("restarts must be in (0, 1) or [1, inf)"))
            .map(|restarts| RandomWalker { restarts, walks, beta })
    }

    /// Simple representation of the RandomWalker
    pub fn __repr__(&self) -> String {
        format!("RandomWalker<restarts={:?}, walks={}, beta={:?}>", self.restarts, self.walks, self.beta)
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
    ///    filter_type : String or List[String] - Optional
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
    #[pyo3(signature = (graph, node, seed=None, k=None, filter_type=None, single_threaded=None, weighted=None))]
    pub fn walk(
        &self, 
        graph: &Graph,
        node: FQNode, 
        seed: Option<u64>, 
        k: Option<usize>, 
        filter_type: Option<&Bound<'_, PyAny>>,
        single_threaded: Option<bool>,
        weighted: Option<bool>
    ) -> PyResult<Vec<(FQNode, f32)>> {
        let node_id = graph.node_id(&node.0, &node.1)?;
        let mut rwr = RWR::new(self.restarts, self.walks);
        rwr.beta = self.beta.unwrap_or(rwr.beta);
        rwr.single_threaded = single_threaded.unwrap_or(rwr.single_threaded);
        rwr.seed = seed.unwrap_or(rwr.seed);

        let results = if weighted.unwrap_or(true) {
            rwr.sample_bfs(graph.graph(), node_id)
        } else {
            rwr.sample(graph.graph(), &Unweighted, node_id)
        };

        let fts = convert_filter_type(filter_type)?;
        Ok(convert_scores(&graph.vocab, results.into_iter(), k, fts))
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
    #[pyo3(signature = (restarts, walks, beta=None, blend=None))]
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
    ///    filter_type : String or List[String] - Optional
    ///        If provided, only returns nodes that match the provided node type.
    ///    
    ///    
    ///    Returns
    ///    -------
    ///    List[(FQNode, Float)] - Can throw exception
    ///        List of fully qualified nodes and their fractional scores.
    ///        
    #[pyo3(signature = (graph, embeddings, node, context, k=None, seed=None, rerank_context=None, filter_type=None))]
    pub fn walk(
        &self, 
        graph: &Graph,
        embeddings: &NodeEmbeddings,
        node: FQNode, 
        context: &Query,
        k: Option<usize>, 
        seed: Option<u64>, 
        rerank_context: Option<&Query>,
        filter_type: Option<&Bound<'_, PyAny>>
    ) -> PyResult<Vec<(FQNode, f32)>> {
        ensure_aligned(&graph.vocab, &embeddings.vocab, "Graph and NodeEmbeddings")?;
        let node_id = get_node_id(graph.vocab.deref(), node.0, node.1)?;
        let g_emb = lookup_embedding(context, embeddings)?;
        
        let mut grwr = GuidedRWR::new(GSteps::from_restarts(self.restarts)?, self.walks);
        grwr.alpha = self.blend.unwrap_or(grwr.alpha);
        grwr.beta = self.beta.unwrap_or(grwr.beta);
        grwr.seed = seed.unwrap_or(grwr.seed);

        let node_embeddings = &embeddings.embeddings;
        let mut results = grwr.sample(graph.graph.as_ref(), 
                                  &Weighted, node_embeddings, node_id, g_emb);
        
        // Reweight results if requested
        if let Some(cn) = rerank_context {
            let c_emb = lookup_embedding(cn, embeddings)?;
            Reweighter::new(self.blend.unwrap_or(0.5))
                .reweight(&mut results, node_embeddings, c_emb);
        }

        let fts = convert_filter_type(filter_type)?;
        Ok(convert_scores(&graph.vocab, results.into_iter(), k, fts))
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
    #[pyo3(signature = (restarts, eps=None))]
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
    ///    filter_type : String or List[String] - Optional
    ///        If provided, filters out nodes that do not match the provided filter_type.
    ///    
    ///    Returns
    ///    -------
    ///    List[(FQNode, f32)] - Can throw exception
    ///        List of fully qualified nodes and their fractional scores
    ///    
    #[pyo3(signature = (graph, node, k=None, filter_type=None))]
    pub fn compute(
        &self, 
        graph: &Graph,
        node: FQNode, 
        k: Option<usize>, 
        filter_type: Option<&Bound<'_, PyAny>>
    ) -> PyResult<Vec<(FQNode, f32)>> {

        let node_id = get_node_id(graph.vocab.deref(), node.0, node.1)?;
        let results = ppr_estimate(graph.graph.as_ref(), node_id, self.restarts, self.eps);

        let fts = convert_filter_type(filter_type)?;

        Ok(convert_scores(&graph.vocab, results.into_iter(), k, fts))
    }

}


/// Type of edge.  Undirected edges internally get converted to two directed edges.
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone,PartialEq)]
pub enum EdgeType {
    Directed,
    Undirected
}

impl From<EdgeType> for crate::graph::EdgeType {
    fn from(e: EdgeType) -> Self {
        match e {
            EdgeType::Directed => crate::graph::EdgeType::Directed,
            EdgeType::Undirected => crate::graph::EdgeType::Undirected
        }
    }
}

/// Allows the user to build a graph incrementally before converting it into a proper CSR graph
#[pyclass]
struct GraphBuilder {
    gb: CoreGraphBuilder
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
        GraphBuilder { gb: CoreGraphBuilder::new() }
    }
    
    /// Simple representation of the GraphBuilder
    pub fn __repr__(&self) -> String {
        format!("GraphBuilder<Nodes={}, Edges={}>", self.gb.nodes(), self.gb.edges())
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
        self.gb.add_edge((&from_node.0, &from_node.1), (&to_node.0, &to_node.1), weight, node_type.into());
    }

    ///    Constructs the graph
    ///    
    ///    Returns
    ///    -------
    ///    Graph - Optional
    ///        Creates a Graph for usage.  If no edges have been specified, returns None.
    ///    
    #[pyo3(signature = (deduplicate=None))]
    pub fn build_graph(&mut self, deduplicate: Option<bool>) -> Option<Graph> {
        self.gb.build(deduplicate.unwrap_or(true)).map(Graph::from)
    }

}

#[pyclass(from_py_object)]
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

/// Optimizer used by the learners (EmbeddingPropagator, PprRankLearner) to update the feature
/// embeddings.
#[pyclass(name = "Optimizer", from_py_object)]
#[derive(Clone)]
struct OptimizerSpec {
    kind: OptimizerKind
}

#[pymethods]
impl OptimizerSpec {

    ///    Adam.  This is the default.
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///    
    #[staticmethod]
    pub fn adam() -> Self {
        OptimizerSpec { kind: OptimizerKind::Adam }
    }

    ///    AdamW: Adam with decoupled weight decay.  At the end of every pass, every feature
    ///    embedding is multiplied by exp(-m * weight_decay), independently of the gradient, where
    ///    m is the learning rate schedule's current value relative to its peak (1 for a constant
    ///    schedule).  The decay per pass is the same for every feature, for batch and epoch
    ///    updates, and at any learning rate.
    ///    
    ///    Parameters
    ///    ----------
    ///    weight_decay : Float - Optional
    ///        Decay rate per pass; must be finite and non-negative.  Default is 0.001, which
    ///        shrinks weights by about 0.1% a pass.
    ///    
    ///    Returns
    ///    -------
    ///    Self - Can throw exception
    ///    
    #[staticmethod]
    #[pyo3(signature = (weight_decay=None))]
    pub fn adamw(weight_decay: Option<f32>) -> PyResult<Self> {
        let rate = weight_decay.unwrap_or(0.001);
        let wd = WeightDecay::new(rate).ok_or_else(|| PyValueError::new_err(
            format!("weight_decay must be finite and non-negative, got {}", rate)))?;
        Ok(OptimizerSpec { kind: OptimizerKind::AdamW(wd) })
    }

    pub fn __repr__(&self) -> String {
        match self.kind {
            OptimizerKind::Adam => "Optimizer.adam()".into(),
            OptimizerKind::AdamW(wd) => format!("Optimizer.adamw(weight_decay={})", wd.rate())
        }
    }
}

/// A python wrapper for the internal ADT used for defining losses
#[pyclass(from_py_object)]
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
    #[pyo3(signature = (gamma, negatives=None))]
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
    ///    weighted_neighbor_sampling : Bool - Optional
    ///        When max_nodes subsamples a neighborhood, picks neighbors in proportion to their
    ///        edge weights instead of uniformly.  A weighted sample is averaged with equal weights,
    ///        so combined with weighted_neighbor_averaging the edge weights are counted once, not
    ///        twice.  Has no effect when the whole neighborhood is used.  Default is False.
    ///
    ///    weighted_neighbor_averaging : Bool - Optional
    ///        Reconstructs the node from the edge-weighted mean of its neighbors rather than the
    ///        plain mean.  Ignored by attention models.  Default is False.
    ///
    ///    max_features : Int/Float - Optional
    ///        Maximum number of features to use to construct a node embedding.  These features
    ///        will be randomly selected every node construction.  If given a float between (0,1),
    ///        will probabalistically drop out features.  Default is all.
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
    ///    optimizer : Optimizer - Optional
    ///        Optimizer for the feature embeddings, e.g. Optimizer.adamw(0.01).
    ///
    ///        Default is Optimizer.adam().
    ///
    ///    lr_schedule : String - Optional
    ///        "cosine" warms the learning rate up from alpha / 100 over the first 20% of updates,
    ///        then decays it along a cosine back to alpha / 100.  "constant" keeps it at alpha,
    ///        as in the EP paper.
    ///
    ///        Default is "cosine".
    ///
    ///    init : String - Optional
    ///        Initialisation of the feature embeddings when none are passed to learn_features.
    ///        "unit" draws each coordinate from U(-1, 1) and scales the embedding to unit length.
    ///        "glorot" draws each coordinate from U(-l, l) with l = sqrt(6 / (features + dims)),
    ///        as in the EP paper.
    ///
    ///        Default is "unit".
    ///
    ///    update_frequency : String - Optional
    ///        "batch" updates the feature embeddings after every batch.  "epoch" still processes
    ///        nodes in batches, but every batch computes its gradients against the same
    ///        embeddings; the gradients are summed over the pass and applied in one update at its
    ///        end.  That lowers the variance of each update but leaves only `passes` updates, so
    ///        it usually wants a larger alpha or more passes.  It also keeps an extra gradient
    ///        table as large as the feature embeddings.
    ///
    ///        Default is "batch".
    ///
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[new]
    #[pyo3(signature = (alpha=None, loss=None, batch_size=None, dims=None, passes=None, seed=None, max_nodes=None, weighted_neighbor_sampling=None, weighted_neighbor_averaging=None, max_features=None, loss_weighting=None, valid_pct=None, hard_negatives=None, indicator=None, attention=None, attention_heads=None, context_window=None, noise=None, optimizer=None, lr_schedule=None, init=None, update_frequency=None))]
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
        max_nodes: Option<f32>,

        // How we sample neighbors
        weighted_neighbor_sampling: Option<bool>,

        // How we blend neighbors during reconstruction
        weighted_neighbor_averaging: Option<bool>,

        // Max features to use for optimization
        max_features: Option<f32>,

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
        noise: Option<f32>,

        // Optimizer for the feature embeddings
        optimizer: Option<OptimizerSpec>,

        // Learning rate schedule: "cosine" or "constant"
        lr_schedule: Option<String>,

        // Feature embedding initialisation: "unit" or "glorot"
        init: Option<String>,

        // When to update the feature embeddings: "batch" or "epoch"
        update_frequency: Option<String>
    ) -> PyResult<Self> {
        let lr_schedule = match lr_schedule.as_deref().unwrap_or("cosine") {
            "cosine" => LearningRateSchedule::CosineWarmup,
            "constant" => LearningRateSchedule::Constant,
            other => return Err(PyValueError::new_err(
                format!("lr_schedule must be \"cosine\" or \"constant\", got {:?}", other)))
        };
        let init = match init.as_deref().unwrap_or("unit") {
            "unit" => EmbeddingInit::UnitNorm,
            "glorot" => EmbeddingInit::GlorotUniform,
            other => return Err(PyValueError::new_err(
                format!("init must be \"unit\" or \"glorot\", got {:?}", other)))
        };
        let update_frequency = match update_frequency.as_deref().unwrap_or("batch") {
            "batch" => UpdateFrequency::Batch,
            "epoch" => UpdateFrequency::Epoch,
            other => return Err(PyValueError::new_err(
                format!("update_frequency must be \"batch\" or \"epoch\", got {:?}", other)))
        };
        let loss_weighting = loss_weighting.map(|lw| lw.loss).unwrap_or(EPLW::None);
        let d = EmbeddingPropagation::default();
        let ep = EmbeddingPropagation {
            alpha: alpha.unwrap_or(d.alpha),
            batch_size: batch_size.unwrap_or(d.batch_size),
            d_model: dims.unwrap_or(d.d_model),
            passes: passes.unwrap_or(d.passes),
            loss: loss.map(|l|l.loss).unwrap_or(d.loss),
            hard_negs: hard_negatives.unwrap_or(d.hard_negs),
            loss_weighting: loss_weighting,
            valid_pct: valid_pct.unwrap_or(d.valid_pct),
            seed: seed.unwrap_or(d.seed),
            indicator: indicator.unwrap_or(d.indicator),
            noise: noise.unwrap_or(d.noise),
            optimizer: optimizer.map(|o| o.kind).unwrap_or(d.optimizer),
            lr_schedule,
            init,
            update_frequency
        };

        let wns = weighted_neighbor_sampling.unwrap_or(false);
        let wna = weighted_neighbor_averaging.unwrap_or(false);
        let max_features = max_features
            .map(|mf| Sample::new(mf))
            .unwrap_or(Ok(Sample::All))
            .map_err(|_err| PyValueError::new_err("max_features must be either None, or (0, N)"))?;

        let max_nodes = max_nodes 
            .map(|mn| Sample::new(mn))
            .unwrap_or(Ok(Sample::All))
            .map_err(|_err| PyValueError::new_err("max_nodes must be either None, or (0, N)"))?;

        let model = if let Some(d_k) = attention {
            let num_heads = attention_heads.unwrap_or(1);
            let at = if let Some(size) = context_window {
                AttentionType::Sliding{window_size: size}
            } else {
                match max_features {
                    Sample::All => AttentionType::Full,
                    _ => AttentionType::Random { num_features: max_features }
                }
            };
            let mha = MultiHeadedAttention::new(num_heads, d_k, at);
            ModelType::Attention(AttentionFeatureModel::new(mha, Sample::All, max_nodes, wns))
        } else {
            ModelType::Averaged(AveragedFeatureModel::new(
                max_features, max_nodes, wns, wna
            ))
        };

        Ok(EmbeddingPropagator{ ep, model })
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
    #[pyo3(signature = (graph, features, feature_embeddings=None))]
    pub fn learn_features(
        &mut self, 
        graph: &Graph, 
        features: &mut FeatureSet,
        feature_embeddings: Option<&mut NodeEmbeddings>
    ) -> PyResult<NodeEmbeddings> {

        let warm_start = take_warm_start(&graph.inner, &mut features.inner, feature_embeddings)?;
        let learned = match &self.model {
            ModelType::Averaged(model) =>
                self.ep.learn_features(&graph.inner, &mut features.inner, warm_start, model),
            ModelType::Attention(model) =>
                self.ep.learn_features(&graph.inner, &mut features.inner, warm_start, model)
        }?;
        Ok(NodeEmbeddings::from(learned))

    }
    
    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.ep)
    }

}

#[pyclass(from_py_object)]
#[derive(Clone,Default)]
pub struct FeatureNamespace(CoreFeatureNamespace);

#[pymethods]
impl FeatureNamespace {
    #[staticmethod]
    pub fn single(ns: String) -> Self {
        FeatureNamespace(CoreFeatureNamespace::single(ns))
    }

    #[staticmethod]
    pub fn node_type() -> Self {
        FeatureNamespace(CoreFeatureNamespace::node_type())
    }

    #[staticmethod]
    pub fn prefix(delim: String) -> PyResult<Self> {
        Ok(FeatureNamespace(CoreFeatureNamespace::prefix(delim)?))
    }

}

/// Defines the FeatureSet class, which allows setting discrete features for a node
#[pyclass]
pub struct FeatureSet {
    inner: CoreFeatureSet
}

impl From<CoreFeatureSet> for FeatureSet {
    fn from(inner: CoreFeatureSet) -> Self {
        FeatureSet { inner }
    }
}

impl Deref for FeatureSet {
    type Target = CoreFeatureSet;
    fn deref(&self) -> &CoreFeatureSet {
        &self.inner
    }
}

impl std::ops::DerefMut for FeatureSet {
    fn deref_mut(&mut self) -> &mut CoreFeatureSet {
        &mut self.inner
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
    ///    Returns
    ///    -------
    ///    Self - Can throw exception
    ///        
    ///    
    #[staticmethod]
    pub fn new_from_graph(graph: &Graph) -> PyResult<Self> {
        Ok(FeatureSet::from(CoreFeatureSet::from_graph(&graph.inner)))
    }

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
    #[pyo3(signature = (path, namespace=None))]
    pub fn new_from_file(
        path: String, 
        namespace: Option<FeatureNamespace>
    ) -> PyResult<Self> {
        Ok(FeatureSet::from(CoreFeatureSet::from_file(&path, &namespace.unwrap_or_default().0)?))
    }

    ///    Sets the features for a Node.
    ///    
    ///    Parameters
    ///    ----------
    ///    node : FQNode
    ///        Fully qualified Node.
    ///    
    ///    features : List[(String, String)]
    ///        Features to set for this node.
    ///    
    ///    Returns
    ///    -------
    ///    () - Can throw exception
    ///        
    ///    
    pub fn set_features(
        &mut self, 
        node: FQNode, 
        features: Vec<(String, String)>
    ) -> PyResult<()> {
        Ok(self.inner.set_features(&node.0, &node.1, features.into_iter())?)
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
    ///    List[(String, String)] - Can throw exception
    ///        Set of features specified for node
    ///    
    pub fn get_features(
        &self, 
        node: FQNode
    ) -> PyResult<Vec<(String,String)>> {
        Ok(self.inner.get_features(&node.0, &node.1)?)
    }

    ///    Loads a file defining fully qualified nodes to features into a feature set.
    ///    
    ///    Parameters
    ///    ----------
    ///    path : String
    ///        Path point to features definitions.
    ///
    ///    f_ns : FeatureNamespace - optional
    ///        How to construct a feature namespace from a given token.  Default is Static("feat")
    ///    
    ///    Returns
    ///    -------
    ///    () - Can throw exception
    ///        
    ///    
    #[pyo3(signature = (path, f_ns=None))]
    pub fn load_into(
        &mut self,
        path: String,
        f_ns: Option<FeatureNamespace>
    ) -> PyResult<()> {
        Ok(self.inner.load_into(&path, &f_ns.unwrap_or_default().0)?)
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
        FeatureSet::from(self.inner.prune_min_count(count))
    }

    ///    Returns the FQN for the feature at feature_id.
    ///    
    ///    Parameters
    ///    ----------
    ///    idx : isize
    ///        Internal index.
    ///    
    ///    Returns
    ///    -------
    ///    (FQNode) - Can throw exception
    ///        
    ///    
    pub fn __getitem__(&self, idx: isize) -> PyResult<FQNode> {
        let idx = resolve_index(idx, self.features.num_features())
            .ok_or_else(|| PyIndexError::new_err("Feature index out of range for FeatureSet!"))?;

        let fqn = self.features.get_pretty_feature(idx);
        Ok(fqn)
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
    #[pyo3(signature = (k, threshold=None, max_iters=None))]
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
    ) -> PyResult<()> {
        ensure_aligned(&graph.vocab, &features.vocab, "Graph and FeatureSet")?;
        propagate_features(
            graph.graph.deref(), 
            &mut features.features, 
            self.max_iters,
            self.k,
            self.threshold);
        Ok(())
    }

}

/// After we train a set of features using EP, we need to use an aggregator to take a set of
/// features and glue them into a new NodeEmbedding

/// This constructs node embeddings based on the AggregatorType and a set of FeatureEmbeddings.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct FeatureAggregator {
    agg: CoreFeatureAggregator
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
        FeatureAggregator { agg: CoreFeatureAggregator::averaged() }
    }

    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("FeatureAggregator<{}>", self.agg)
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
    #[pyo3(signature = (num_heads, d_k, window=None))]
    pub fn Attention(num_heads: usize, d_k: usize, window: Option<usize>) -> Self {
        FeatureAggregator { agg: CoreFeatureAggregator::attention(num_heads, d_k, window) }
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
        FeatureAggregator { agg: CoreFeatureAggregator::weighted(alpha, &fs.inner) }
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
        Ok(self.agg.save(path)?)
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
        Ok(FeatureAggregator { agg: CoreFeatureAggregator::load(&path)? })
    }


}

/// Finally, a node embedder.  This wraps the FeatureAggregator because who doesn't like more
/// indirection?
#[pyclass]
pub struct NodeEmbedder {
    feat_agg: FeatureAggregator,
    embedder: CoreNodeEmbedder
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
        let embedder = CoreNodeEmbedder::new(feat_agg.agg.clone());
        NodeEmbedder { feat_agg, embedder }
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
    ) -> PyResult<NodeEmbeddings> {
        Ok(NodeEmbeddings::from(self.embedder.embed_feature_set(&feat_set.inner, &feature_embeddings.inner)?))
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
    #[pyo3(signature = (features, feature_embeddings, strict=None))]
    pub fn embed_adhoc(
        &self, 
        features: Vec<FQNode>,
        feature_embeddings: &NodeEmbeddings,
        strict: Option<bool>
    ) -> PyResult<Vec<f32>> {
        Ok(self.embedder.embed_adhoc(&features, &feature_embeddings.inner, strict.unwrap_or(true))?)
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
    #[pyo3(signature = (features_set, feature_embeddings, strict=None))]
    pub fn bulk_embed_adhoc(&self, 
        features_set: Vec<Vec<FQNode>>,
        feature_embeddings: &NodeEmbeddings,
        strict: Option<bool>
    ) -> PyResult<Vec<Vec<f32>>> {
        Ok(self.embedder.bulk_embed_adhoc(&features_set, &feature_embeddings.inner, strict.unwrap_or(true))?)
    }

}

/// Struct for defining ALT embeddings
#[pyclass]
struct DistanceEmbedder {
    landmarks: crate::algos::dist::LandmarkSelection,
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
    #[pyo3(signature = (n_landmarks, seed=None))]
    pub fn new(n_landmarks: usize, seed: Option<u64>) -> Self {
        let ls = if let Some(seed) = seed {
            crate::algos::dist::LandmarkSelection::Random(seed)
        } else {
            crate::algos::dist::LandmarkSelection::Degree
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
        NodeEmbeddings::from(CoreNodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings: es
        })
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
    #[pyo3(signature = (k, passes, seed=None))]
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
        NodeEmbeddings::from(CoreNodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings: es
        })
    }
}

#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone,Copy,PartialEq)]
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
    #[pyo3(signature = (t, threshold, memory_size=None, rule=None, seed=None))]
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
        NodeEmbeddings::from(CoreNodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings: es
        })
    }

}

/// Computes the PageRank for all nodes in the graph.
#[pyclass]
struct PageRank {
    iterations: usize,
    damping: f64, 
    eps: f64
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
    #[pyo3(signature = (iterations, damping=None, eps=None))]
    pub fn new(iterations: usize, damping: Option<f64>, eps: Option<f64>) -> Self {
        let d = crate::algos::pagerank::PageRank::with_iterations(iterations);
        PageRank { iterations, damping: damping.unwrap_or(d.damping()), eps: eps.unwrap_or(d.eps()) }
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
    #[pyo3(signature = (graph, indicator=None))]
    pub fn learn(&self, graph: &Graph, indicator: Option<bool>) -> NodeEmbeddings {
        let page_rank = crate::algos::pagerank::PageRank::new(self.iterations, self.damping, self.eps);
        NodeEmbeddings::from(page_rank.learn(&graph.inner, indicator.unwrap_or(true)))
    }

}



/// Wrapper for EmbeddingStore.
#[pyclass]
pub struct NodeEmbeddings {
    inner: CoreNodeEmbeddings
}

impl From<CoreNodeEmbeddings> for NodeEmbeddings {
    fn from(inner: CoreNodeEmbeddings) -> Self {
        NodeEmbeddings { inner }
    }
}

impl Deref for NodeEmbeddings {
    type Target = CoreNodeEmbeddings;
    fn deref(&self) -> &CoreNodeEmbeddings {
        &self.inner
    }
}

impl std::ops::DerefMut for NodeEmbeddings {
    fn deref_mut(&mut self) -> &mut CoreNodeEmbeddings {
        &mut self.inner
    }
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
        NodeEmbeddings::from(CoreNodeEmbeddings::zeros(graph.vocab.clone(), dims, distance.to_edist()))
    }

    #[staticmethod]
    pub fn new_from_list(
        list: Vec<((String, String), Vec<f32>)>, 
        distance: Distance
    ) -> PyResult<Self> {
        let mut builder = CoreNodeEmbeddingsBuilder::new(distance.to_edist());
        for ((node_type, node_name), emb) in list.iter() {
            builder.add_embedding(node_type, node_name, emb).map_err(dimension_error)?;
        }
        let inner = builder.build().unwrap_or_else(||
            CoreNodeEmbeddings::zeros(Arc::new(Vocab::new()), 0, distance.to_edist()));
        Ok(NodeEmbeddings::from(inner))
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
        self.inner.contains(&node.0, &node.1)
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
        Ok(self.inner.get_embedding(&node.0, &node.1)?.to_vec())
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
        Ok(self.inner.set_embedding(&node.0, &node.1, &embedding)?)
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
    ///    filter_type : String or List[String] - Optional
    ///        If provided, filters out nodes that don't match the filter_type
    ///    
    ///    Returns
    ///    -------
    ///    List[(FQNode, f32)]
    ///        Set of fully qualified nodes and distances.
    ///    
    #[pyo3(signature = (emb, k, filter_type=None))]
    pub fn nearest_neighbor(
        &self, 
        emb: Vec<f32>, 
        k: usize,
        filter_type: Option<&Bound<'_, PyAny>>
    ) -> PyResult<Vec<(FQNode, f32)>> {
        let fts = convert_filter_type(filter_type)?;
        let dists = self.inner.nearest_neighbor(&emb, k, fts.as_ref());
        Ok(convert_node_distance(&self.vocab, dists))
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
    pub fn __getitem__(&self, idx: isize) -> PyResult<(FQNode, Vec<f32>)> {
        let idx = resolve_index(idx, self.embeddings.len())
            .ok_or_else(|| PyIndexError::new_err("Index out of range for embedding store!"))?;

        let emb = self.embeddings.get_embedding(idx).to_vec();
        let (nt, nn) = self.vocab.get_name(idx).expect("Already validated bounds, something borked");
        Ok((((*nt).clone(), nn.to_string()), emb))
    }

    ///    L2Norms the embeddings, useful for cosine similarity.
    ///    
    pub fn l2norm(&mut self) {
        self.inner.l2_normalize()
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
    #[pyo3(signature = (path, comp_level=None))]
    pub fn save(&self, path: &str, comp_level: Option<u32>) -> PyResult<()> {
        Ok(self.inner.save(path, comp_level)?)
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
    ///    filter_type : String or List[String] - Optional
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
    #[pyo3(signature = (path, distance=None, filter_type=None, chunk_size=None, skip_rows=None))]
    pub fn load(
        py: Python<'_>,
        path: &str, 
        distance: Option<Distance>, 
        filter_type: Option<&Bound<'_, PyAny>>, 
        chunk_size: Option<usize>,
        skip_rows: Option<usize>
    ) -> PyResult<Self> {
        let fts = convert_filter_type(filter_type)?;
        let distance = distance.unwrap_or(Distance::Cosine).to_edist();
        py.detach(move || {
            let inner = CoreNodeEmbeddings::load(path, distance, fts.as_ref(), chunk_size, skip_rows)?;
            Ok(NodeEmbeddings::from(inner))
        })
    }
}

/// Allows the user to build NodeEmbeddings incrementally.
#[pyclass]
struct NodeEmbeddingsBuilder {
    builder: CoreNodeEmbeddingsBuilder
}

#[pymethods]
impl NodeEmbeddingsBuilder {
    ///    Creates a NodeEmbeddingsBuilder.  Allows progressive addition of nodes and embeddings.
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
        NodeEmbeddingsBuilder { builder: CoreNodeEmbeddingsBuilder::new(dist.to_edist()) }
    }
    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("NodeEmbeddingsBuilder<Nodes={}, Distance={:?}>", self.builder.len(), self.builder.distance())
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
        self.builder.add_embedding(&node.0, &node.1, &embedding).map_err(dimension_error)
    }

    ///    Creates the NodeEmbeddings.
    ///    
    ///    Returns
    ///    -------
    ///    NodeEmbeddings - Optional
    ///        If Embeddings have been added, creates the NodeEmbeddings.  Otherwise returns None.
    ///    
    pub fn build(&mut self) -> Option<NodeEmbeddings> {
        self.builder.build().map(NodeEmbeddings::from)
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

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<FQNode> {
        if slf.cur_idx < slf.vocab.len() {
            let (node_type, name) = slf.vocab.get_name(slf.cur_idx).unwrap();
            let fqn = ((*node_type).clone(), name.to_string());
            slf.cur_idx += 1;
            Some(fqn)
        } else {
            None
        }
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
    #[pyo3(signature = (alpha=None, max_neighbors=None))]
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
    ) -> PyResult<NodeEmbeddings> {
        Ok(NodeEmbeddings::from(self.aligner.align_all(&embeddings.inner, &graph.inner)?))
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
    #[pyo3(signature = (path, embeddings, graph, chunk_size=None, comp_level=None))]
    pub fn align_to_disk(
        &self, 
        path: &str,
        embeddings: &NodeEmbeddings, 
        graph: &Graph,
        chunk_size: Option<usize>,
        comp_level: Option<u32>
    ) -> PyResult<()> {
        Ok(self.aligner.align_to_disk(path, &embeddings.inner, &graph.inner,
                                      chunk_size.unwrap_or(10_000), comp_level)?)
    }


}

/// Aligns an adhoc embedding to a transform embedding state, typically from Neighborhood
/// transforms
#[pyclass]
struct EmbeddingAligner {
    aligner: CoreEmbeddingAligner
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
    #[pyo3(signature = (num_nodes, random_nodes=None, alpha=None, error=None, max_iters=None))]
    pub fn new(
        num_nodes: usize, 
        random_nodes: Option<usize>,
        alpha: Option<f32>,
        error: Option<f32>,
        max_iters: Option<usize>
    ) -> Self {
        let d = CoreEmbeddingAligner::new(num_nodes);
        EmbeddingAligner { aligner: CoreEmbeddingAligner {
            num_nodes,
            random_nodes: random_nodes.unwrap_or(d.random_nodes),
            alpha: alpha.unwrap_or(d.alpha),
            error: error.unwrap_or(d.error),
            max_iters: max_iters.unwrap_or(d.max_iters)
        } }
    }

    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("EmbeddingAligner<NumNodes={}, RandomNodes={}>", self.aligner.num_nodes, self.aligner.random_nodes)
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
    #[pyo3(signature = (orig_embeddings, orig_ann, translated_embeddings, emb, seed=None))]
    pub fn align(&self, 
        orig_embeddings: &NodeEmbeddings, 
        orig_ann: &EmbAnn,
        translated_embeddings: &NodeEmbeddings,
        emb: &Query,
        seed: Option<u64>
    ) -> PyResult<Vec<f32>> {
        let query_embedding = lookup_embedding(emb, translated_embeddings)?;
        Ok(self.aligner.align(&orig_embeddings.inner, &orig_ann.ann, &translated_embeddings.inner,
                              query_embedding, seed.unwrap_or(SEED + 123123)))
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
    #[pyo3(signature = (orig_embeddings, orig_ann, translated_embeddings, queries, seed=None))]
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
    #[pyo3(signature = (graph, max_steps=None))]
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
    #[pyo3(signature = (query, embeddings, k, seed=None))]
    pub fn find(
        &self, 
        query: &Query,
        embeddings: &NodeEmbeddings, 
        k: usize, 
        seed: Option<u64>
    ) -> PyResult<Vec<(FQNode, f32)>> {
        let query_embedding = lookup_embedding(query, embeddings)?;
        let seed = seed.unwrap_or(SEED + 10);
        let ann = crate::algos::graph_ann::Ann::new(k, self.max_steps + k, seed);
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
    ///    test_hp_per_split : Int - Optional
    ///        Number of candidate hyperplanes to pick before selecting a splitting hyperplane.
    ///        More usually results in cleaner splits at the expense of compute.
    ///    
    ///    num_sampled_nodes_split_test : Int - Optional
    ///        When determining the pseudo clusters for splitting during random projections, the
    ///        number of nodes to use to "estimate" each cluster.  More means more accurate pseudo
    ///        clusters at the cost of more compute.
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
    #[pyo3(signature = (embs, n_trees, max_nodes_per_leaf, test_hp_per_split=None, num_sampled_nodes_split_test=None, filter_type=None, seed=None))]
    pub fn new(
        embs: &NodeEmbeddings, 
        n_trees: usize,
        max_nodes_per_leaf: usize,
        test_hp_per_split: Option<usize>,
        num_sampled_nodes_split_test: Option<usize>,
        filter_type: Option<&Bound<'_, PyAny>>,
        seed: Option<u64>
    ) -> PyResult<Self> {
        let mut ann = Ann::new();
        let seed = seed.unwrap_or(SEED + 10);

        // If filter_type is provided, only build an index to reference those nodes.
        let node_ids = convert_filter_type(filter_type)?.map(|hs| embs.vocab.nodes_of_types(&hs));

        ann.fit(
            &embs.embeddings, 
            n_trees, 
            max_nodes_per_leaf, 
            test_hp_per_split,
            num_sampled_nodes_split_test,
            node_ids,
            seed);

        Ok(EmbAnn { ann: ann })

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
    #[pyo3(signature = (embeddings, query, k, min_search_size=None))]
    pub fn find(
        &self, 
        embeddings: &NodeEmbeddings,
        query: &Query,
        k: usize,
        min_search_size: Option<usize>
    ) -> PyResult<Vec<(FQNode, f32)>> {
        let query_embedding = lookup_embedding(query, embeddings)?;
        let nodes = self.ann.predict(&embeddings.embeddings, query_embedding, k, min_search_size);
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

/// Builds a table indexed by graph node id which yields the matching embedding id, if any.

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

    #[pyo3(signature = (iterations, num_walks, alpha, discount, step_penalty, explore_pct, restart_prob, compression=None, embeddings=None, seed=None))]
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

        let graph = NamedGraph::new(self.vocab.clone(), self.graph.clone())?;
        let new_graph = smci.learn_graph(&graph, &self.rewards, embeddings.map(|e| &e.inner));

        Ok(Graph::from(new_graph))

    }
}

/// Implements Bellman expectation equation for graph-based policy evaluation.
/// Treats the graph as an MDP where nodes are states and outgoing edges are actions.
/// Computes the optimal value function V(s) given a per-node reward signal, then reweights
/// edges via softmax to bias trajectories toward higher-value states.
#[pyclass]
struct PolicyEvaluation {
    vocab: Arc<Vocab>,
    gamma: f32,
    iterations: usize,
    eps: f32,
    temperature: f32,
    rewards: Vec<f32>,
}

#[pymethods]
impl PolicyEvaluation {
    ///    Creates a PolicyEvaluation instance seeded from a graph's vocabulary.
    ///
    ///    The graph itself is NOT stored on the returned object.  Pass it back
    ///    to `optimize(graph)` when you're ready to run.  This avoids holding
    ///    an extra reference to the graph, which would force a full copy of
    ///    its edges inside `optimize`.
    ///
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph whose vocabulary and size seed this PolicyEvaluation.  The
    ///        same graph should be passed to `optimize` later.
    ///
    ///    gamma : Float - Optional
    ///        Discount factor ~ [0, 1).  Higher values make the agent plan further
    ///        ahead.  Default is 0.99.
    ///
    ///    iterations : Int - Optional
    ///        Maximum number of Bellman update sweeps.  Default is 100.
    ///
    ///    eps : Float - Optional
    ///        Convergence threshold.  Stops early when L2 change in V is below eps.
    ///        Default is 1e-6.
    ///
    ///    temperature : Float - Optional
    ///        Controls the sharpness of the softmax policy extraction.  Lower values
    ///        produce a near-deterministic policy; higher values keep edges uniform.
    ///        Default is 1.0.
    ///
    ///    Returns
    ///    -------
    ///    Self
    #[new]
    #[pyo3(signature = (graph, gamma=None, iterations=None, eps=None, temperature=None))]
    pub fn new(
        graph: &Graph,
        gamma: Option<f32>,
        iterations: Option<usize>,
        eps: Option<f32>,
        temperature: Option<f32>,
    ) -> Self {
        let d = PEA::default();
        PolicyEvaluation {
            vocab: graph.vocab.clone(),
            gamma: gamma.unwrap_or(d.gamma),
            iterations: iterations.unwrap_or(d.iterations),
            eps: eps.unwrap_or(d.eps),
            temperature: temperature.unwrap_or(d.temperature),
            rewards: vec![0f32; graph.nodes()],
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PolicyEvaluation<gamma={}, iterations={}, eps={}, temperature={}>",
            self.gamma, self.iterations, self.eps, self.temperature
        )
    }

    ///    Sets the reward for a specific node.
    ///
    ///    Parameters
    ///    ----------
    ///    node : FQNode
    ///        Fully qualified node: (NodeType, NodeName)
    ///
    ///    reward : Float
    ///        Reward value for the node.
    ///
    ///    Returns
    ///    -------
    ///    () - Can throw exception
    pub fn set_reward(&mut self, node: FQNode, reward: f32) -> PyResult<()> {
        let node_id = get_node_id(self.vocab.deref(), node.0, node.1)?;
        self.rewards[node_id] = reward;
        Ok(())
    }

    ///    Loads rewards from a tab-separated file.  Format per line:
    ///    node_type<TAB>node_name<TAB>reward
    ///
    ///    Nodes not present in the file retain their current reward (default 0).
    ///
    ///    Parameters
    ///    ----------
    ///    path : String
    ///        Path to the rewards file.
    ///
    ///    Returns
    ///    -------
    ///    () - Can throw exception
    pub fn load_rewards(&mut self, path: &str) -> PyResult<()> {
        for (node_id, reward) in read_node_values(path, &self.vocab)? {
            self.rewards[node_id] = reward;
        }
        Ok(())
    }

    ///    Runs policy evaluation and rewrites the passed graph's edge weights
    ///    in place.
    ///
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph to optimize.  Must have been built from the same vocabulary
    ///        as this PolicyEvaluation.  Edge weights are mutated in place;
    ///        no new Graph is returned.
    ///
    ///    no_copy : Bool - Optional
    ///        When True (default), raises an exception instead of silently
    ///        copying the graph if it has other live references.  This is the
    ///        whole point of the refactor — accidental allocations should be
    ///        loud.  Pass False to opt into a full copy when that's what you
    ///        want.
    ///
    ///    indicator : Bool - Optional
    ///        Show a progress bar while iterating.  Defaults to True.
    ///
    ///    Returns
    ///    -------
    ///    () - Can throw exception
    #[pyo3(signature = (graph, no_copy=None, indicator=None))]
    pub fn optimize(
        &self,
        graph: &mut Graph,
        no_copy: Option<bool>,
        indicator: Option<bool>,
    ) -> PyResult<()> {
        // Identity check: PE's vocab must be the vocab the graph was built with,
        // otherwise `rewards[node_id]` is meaningless against this graph.
        if !Arc::ptr_eq(&self.vocab, &graph.vocab) {
            return Err(PyValueError::new_err(
                "PolicyEvaluation was built from a different graph's vocabulary; \
                 construct a new PolicyEvaluation from this graph before optimizing."
            ));
        }

        if no_copy.unwrap_or(true) && graph.inner.graph_mut().is_none() {
            return Err(PyRuntimeError::new_err(
                "Cannot optimize in place: this graph has other live \
                 references, so running the optimizer would require \
                 making a full copy of it.  Release the other references \
                 (for example, drop any other Python variables pointing \
                 at the same graph) or pass no_copy=False to allow the \
                 copy."
            ));
        }

        let vi = PEA::new(self.gamma, self.iterations, self.eps, self.temperature, indicator.unwrap_or(true));
        Ok(vi.optimize(&mut graph.inner, &self.rewards)?)
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
    steps: Sample,
    walks: usize,
    k: usize,
    num_features: Option<f32>,
    compression: f32,
    valid_pct: f32,
    beta: f32,
    optimizer: OptimizerKind
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
    ///    num_features : f32- Optional
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
    ///    optimizer : Optimizer - Optional
    ///        Optimizer for the feature embeddings, e.g. Optimizer.adamw(0.01).  Unrelated to
    ///        `weight_decay` above, which adds a penalty term to the loss.
    ///
    ///        Default is Optimizer.adam().
    ///    
    ///    Returns
    ///    -------
    ///    Self - Can throw exception
    ///        
    ///    
    #[new]
    #[pyo3(signature = (alpha, batch_size, dims, passes, steps, walks, k, negatives, loss=None, compression=None, beta=None, num_features=None, weight_decay=None, valid_pct=None, optimizer=None))]
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

        num_features: Option<f32>,

        weight_decay: Option<f32>, 

        // Percentage of nodes to use for validation
        valid_pct: Option<f32>,

        // Optimizer for the feature embeddings
        optimizer: Option<OptimizerSpec>,

    ) -> PyResult<Self> {

        let steps = Sample::steps(steps)
            .map_err(|_| PyValueError::new_err("Steps must be between [0, inf)"))?;

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
            beta: beta.unwrap_or(0.8),
            optimizer: optimizer.map(|o| o.kind).unwrap_or(OptimizerKind::Adam)
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
    #[pyo3(signature = (graph, features, feature_embeddings=None, indicator=None, seed=None))]
    pub fn learn_features(
        &self, 
        graph: &Graph, 
        features: &mut FeatureSet,
        feature_embeddings: Option<&mut NodeEmbeddings>,
        indicator: Option<bool>,
        seed: Option<u64>
    ) -> PyResult<NodeEmbeddings> {

        let loss = if self.loss == "listnet" {
            PprLoss::ListNet { passive: true, weight_decay: self.weight_decay }
        } else {
            PprLoss::ListMLE { weight_decay: self.weight_decay }
        };

        let num_features = self.num_features 
            .map(|mf| Sample::new(mf))
            .unwrap_or(Ok(Sample::All))
            .map_err(|_err| PyValueError::new_err("num features must be either None, or (0, N)"))?;

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
            num_features: num_features,
            compression: self.compression,
            valid_pct: self.valid_pct,
            indicator: indicator.unwrap_or(true),
            seed: seed.unwrap_or(SEED),
            optimizer: self.optimizer
        };

        let warm_start = take_warm_start(&graph.inner, &mut features.inner, feature_embeddings)?;
        Ok(NodeEmbeddings::from(ppr_rank.learn_features(&graph.inner, &mut features.inner, warm_start)?))
    }

}

#[pyclass]
/// Method holding sparse embeddings
struct VpcgFeatureMappings {
    /// Vocab used for NodeID
    vocab: Arc<Vocab>,

    /// Mapping from NodeID to a list of (feature_idx, weight)
    feature_mappings: Vec<Vec<(usize, f32)>>
}

#[pymethods]
impl VpcgFeatureMappings {
    ///    Retrieves the feature mappings and weights for a given node.
    ///    
    ///    Parameters
    ///    ----------
    ///    node : FQNode
    ///        Node to retrieve sparse embeddings from.
    ///    
    pub fn get_feature_map(
        &self, 
        node: FQNode
    ) -> PyResult<Vec<(usize, f32)>> {
        let node_id = get_node_id(&self.vocab, node.0, node.1)?;
        Ok(self.feature_mappings[node_id].clone())
    }

    ///    Retrieves the feature mappings and weights for a given NodeID.
    pub fn __getitem__(
        &self, 
        node_id: NodeID 
    ) -> PyResult<Vec<(usize, f32)>> {
        self.feature_mappings.get(node_id).cloned()
            .ok_or_else(|| PyIndexError::new_err("Node index larger than embedding size!"))
    }

}

/// Tells the VPCG algorithm how to weight the initial features
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone,PartialEq)]
pub enum FeatureWeight {
    /// All features are initialized uniformly: 1 / feat
    Uniform,

    /// All features are initialized as 1 / feat_count.ln()
    IDF
}

/// Learns VPCG vectors on a graph.
#[pyclass]
struct VpcgEmbedder {
    max_terms: usize, 
    passes: usize,
    dims: usize,
    alpha: f32,
    err: f32,
    feature_weight: FeatureWeight
}

impl VpcgEmbedder {
    fn split_graph(
        graph: &Graph, 
        start_node_types: HashSet<String>
    ) -> (Vec<NodeID>, Vec<NodeID>) {
        graph.split_by_node_types(&start_node_types)
    }
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
    ///    feature_weight : FeatureWeight - Optional
    ///        Whether to weight each feature uniformly or based on the IDF
    ///    
    ///    Returns
    ///    -------
    ///    Self
    ///        
    ///    
    #[new]
    #[pyo3(signature = (max_terms, passes, dims, alpha=None, err=None, feature_weight=None))]
    pub fn new(
        max_terms: usize, 
        passes: usize, 
        dims: usize, 
        alpha: Option<f32>, 
        err: Option<f32>,
        feature_weight: Option<FeatureWeight>
    ) -> Self {
        VpcgEmbedder { 
            max_terms, 
            passes, 
            dims,
            alpha: alpha.unwrap_or(1f32),
            err: err.unwrap_or(1e-5),
            feature_weight: feature_weight.unwrap_or(FeatureWeight::Uniform) 
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
    ///    start_node_type : String or List[String]
    ///        Starting node type(s) for propagation.
    ///    
    ///    Returns
    ///    -------
    ///    NodeEmbeddings
    ///        New NodeEmbeddings capturing the VPCG Embeddings.
    ///    
    pub fn learn(&self, 
        graph: &Graph, 
        features: &mut FeatureSet,
        start_node_type: &Bound<'_, PyAny>
    ) -> PyResult<NodeEmbeddings> {
        let feature_maps = self.learn_feature_mapping(graph, features, start_node_type)?;
        let embs = VPCG::convert_to_es(features.num_features(), self.dims, feature_maps.feature_mappings);
        let node_embeddings = NodeEmbeddings::from(CoreNodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings: embs 
        });

        Ok(node_embeddings)
    }

    ///    Learns VPCG feature maps on the graph.
    ///    
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph to use for bipartite structure
    ///    
    ///    features : FeatureSet
    ///        Features to propagate between nodes
    ///    
    ///    start_node_type : String or List[String]
    ///        Starting node type(s) for propagation.
    ///    
    ///    Returns
    ///    -------
    ///    VpcgFeatureMappings
    ///        New FeatureMaps.
    ///    
    pub fn learn_feature_mapping(&self, 
        graph: &Graph, 
        features: &mut FeatureSet,
        start_node_type: &Bound<'_, PyAny>
    ) -> PyResult<VpcgFeatureMappings> {
        let start_node_types = single_or_multistring(start_node_type)?;

        // Bisect the graph
        let (left, right) = VpcgEmbedder::split_graph(graph, start_node_types);

        let feature_weight = match self.feature_weight{
            FeatureWeight::Uniform => VFeatureWeight::Uniform,
            FeatureWeight::IDF => VFeatureWeight::IDF,
        };

        // Every node must have at least one feature
        features.features.fill_missing_nodes();
        let vpcg = VPCG {
            max_terms: self.max_terms, 
            iterations: self.passes,
            alpha: self.alpha,
            err: self.err,
            feature_weight: feature_weight 
        };

        let sparse_embeddings = vpcg.learn(
            graph.graph.as_ref(), 
            &features.features, 
            (&left, &right)
        );

        let vfm = VpcgFeatureMappings {
            vocab: graph.vocab.clone(),
            feature_mappings: sparse_embeddings
        };

        Ok(vfm)

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
    #[pyo3(signature = (dims, num_walks, steps, beta=None, eps=None))]
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
        format!("PPREmbedder<Dims={}, NumWalks={}, Steps={:?}, Beta={}, EPS={}>",
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
    #[pyo3(signature = (graph, features, seed=None))]
    pub fn learn(&self, 
        graph: &Graph, 
        features: &mut FeatureSet,
        seed: Option<u64>
    ) -> PyResult<NodeEmbeddings> {
        features.features.fill_missing_nodes();

        let steps = Sample::steps(self.steps)
            .map_err(|_| PyValueError::new_err("Steps must be between [0, inf)"))?;

        let embedder = PPREmbed {
            dims: self.dims,
            num_walks: self.num_walks,
            steps: steps,
            beta: self.beta,
            eps: self.eps,
            seed: seed.unwrap_or(SEED)
        };

        let embs = embedder.learn(graph.graph.as_ref(), &features.features);
        
        let node_embeddings = NodeEmbeddings::from(CoreNodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings:embs 
        });

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
    #[pyo3(signature = (dims, hashes, num_walks, steps, beta=None, seed=None))]
    pub fn random_walk(
        dims: usize,
        hashes: usize,
        num_walks: usize, 
        steps: f32, 
        beta: Option<f32>,
        seed: Option<u64>
    ) -> PyResult<Self> {
        let steps = Sample::steps(steps)
            .map_err(|_| PyValueError::new_err("Steps must be between [0, inf)"))?;

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
    #[pyo3(signature = (dims, hashes, steps, eps=None))]
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
        
        let node_embeddings = NodeEmbeddings::from(CoreNodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings:embs 
        });

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
    tb: CoreTournamentBuilder
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
        TournamentBuilder { tb: CoreTournamentBuilder::new() }
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
        self.tb.add_outcome((&winner.0, &winner.1), (&loser.0, &loser.1), weight);
    }

    ///    Adds a ordered list where teams in earlier positions out competed teams in later
    ///    positions.  For example, [Racer1, Racer3, Racer2], gets contructed as:
    ///    1. Racer1 > Racer3
    ///    2. Racer1 > Racer2
    ///    3. Racer3 > Racer2
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
        self.tb.add_ranked_outcomes(&order, weight);
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
        self.tb.build().map(|inner| Tournament { inner })
    }

    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        format!("TournamentBuilder<Nodes={}, Outcomes={}>", self.tb.nodes(), self.tb.edges())
    }

}

#[pyclass]
struct Tournament {
    inner: CoreTournament
}

#[pymethods]
impl Tournament {

    /// Simple Python representation 
    pub fn __repr__(&self) -> String {
        let g = self.inner.graph();
        format!("Tournament<Nodes={}, Outcomes={}>", g.nodes(), g.edges())
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
    #[pyo3(signature = (tournament, indicator=None))]
    pub fn learn(
        &self,
        tournament: &Tournament,
        indicator: Option<bool>
    ) -> NodeEmbeddings {
        let lsr = ALSR { passes: self.passes };
        NodeEmbeddings::from(lsr.learn(&tournament.inner, indicator.unwrap_or(true)))
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
        NodeEmbeddings::from(CoreNodeEmbeddings {
            vocab: graph.vocab.clone(),
            embeddings: es
        })
    }
    
    ///    Creates a subgraph of the largest K connected components.  Only guaranteed to work
    ///    for undirected graphs or bidirectionally edged graphs
    ///    
    ///    Parameters
    ///    ----------
    ///    graph : Graph
    ///        Graph to find connected components in
    ///
    ///    k : Int
    ///        Graph to find connected components in
    ///    
    ///    Returns
    ///    -------
    ///    Graph
    ///        Returns a new graph with only the top K connected components
    ///    
    #[staticmethod]
    pub fn prune_largest_components(graph: &Graph, k: usize) -> Option<Graph> {
        graph.inner.prune_largest_components(k).map(Graph::from)
    }

}

#[pyclass]
struct RandomPath {
    rng: XorShiftRng
}

#[pymethods]
impl RandomPath {

    #[new]
    #[pyo3(signature = (seed=None))]
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
        let steps = Sample::new(restarts)
            .map_err(|_s| PyValueError::new_err("restarts must be between [0, inf)"))?;

        //let sampler: Box<dyn Sampler<_>> = if weighted { Box::new(Weighted) } else { Box::new(Unweighted) };
        let g = graph.graph.as_ref();

        let rngs: Vec<_> = (0..count as u64)
            .map(|i| XorShiftRng::seed_from_u64(self.rng.gen::<u64>() + i))
            .collect();

        let vocab = graph.vocab.deref(); 
        let node_id = get_node_id(vocab, node.0, node.1)?;
        let outputs = rollouts(g, steps, weighted, node_id, rngs);

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
        QueryType::Node(nt, nn) => Ok(embeddings.inner.get_embedding(nt, nn)?),
        QueryType::Embedding(ref emb) => Ok(emb),
        QueryType::NodeID(node_id) => Ok(embeddings.inner.get_embedding_by_id(*node_id)?)
    }
}

fn convert_node_distance(
    vocab: &Vocab, 
    dists: Vec<NodeDistance>
) -> Vec<(FQNode, f32)> {
    dists.into_iter().map(|n| {
        let (node_id, dist) = n.to_tup();
        (convert_node_id_to_fqn(vocab, *node_id), dist)
    }).collect()
}

fn single_or_multistring(xs: &Bound<'_, PyAny>) -> PyResult<HashSet<String>> {
    // Try to extract a single string first.
    let mut hs = HashSet::new();
    if let Ok(s) = xs.extract::<String>() {
        hs.insert(s);
        Ok(hs)
    } else if let Ok(v) = xs.extract::<Vec<String>>() {
        v.into_iter().for_each(|s| { hs.insert(s); });
        Ok(hs)
    } else if let Ok(v) = xs.extract::<HashSet<String>>() {
        Ok(v)
    } else {
        Err(PyTypeError::new_err("Expected a string or a list of strings"))
    }
}

fn convert_filter_type(
    filter_type: Option<&Bound<'_, PyAny>>
) -> PyResult<Option<HashSet<String>>> {
    filter_type.map(|pa| single_or_multistring(pa)).transpose()
}

// Training shares embedding stores across rayon threads without locks, so keep the GIL on
// free-threaded interpreters rather than letting Python threads race on the same objects.
#[pymodule(gil_used = true)]
fn cloverleaf(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    m.add_class::<OptimizerSpec>()?;
    m.add_class::<GraphAnn>()?;
    m.add_class::<EmbAnn>()?;
    m.add_class::<FeatureNamespace>()?;
    m.add_class::<FeatureSet>()?;
    m.add_class::<FeaturePropagator>()?;
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
    m.add_class::<VpcgFeatureMappings>()?;
    m.add_class::<FeatureWeight>()?;
    m.add_class::<PPREmbedder>()?;
    m.add_class::<InstantEmbeddings>()?;
    m.add_class::<LSR>()?;
    m.add_class::<TournamentBuilder>()?;
    m.add_class::<Tournament>()?;
    m.add_class::<ConnectedComponents>()?;
    m.add_class::<ListenerRule>()?;
    m.add_class::<LossWeighting>()?;
    m.add_class::<RandomPath>()?;
    m.add_class::<NodeEmbeddingsBuilder>()?;
    m.add_class::<NodeEmbedder>()?;
    m.add_class::<PolicyEvaluation>()?;
    Ok(())
}

#[cfg(test)]
mod binding_tests {
    //! Only exercise code paths that don't touch the Python C API: with the `extension-module`
    //! feature, libpython symbols are unresolved when linking test binaries.
    use super::*;

    /// Python indexing: -len..len is valid, everything else is out of range.
    #[test]
    fn test_resolve_index() {
        assert_eq!(resolve_index(0, 3), Some(0));
        assert_eq!(resolve_index(-1, 3), Some(2));
        assert_eq!(resolve_index(-3, 3), Some(0));
        assert_eq!(resolve_index(3, 3), None);
        assert_eq!(resolve_index(-4, 3), None);
        assert_eq!(resolve_index(-1, 0), None);
    }
}
