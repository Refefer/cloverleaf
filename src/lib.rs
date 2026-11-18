//! Cloverleaf: graph algorithms and graph embeddings that scale to hundreds of millions of nodes.
//!
//! The crate is usable from Rust directly, with the same capabilities as the Python package.  The
//! Python bindings are built with the `python` feature, which maturin turns on; Rust users don't
//! need it or a Python install.
//!
//! The Python classes map onto these types:
//!
//! | Python | Rust |
//! |---|---|
//! | `Graph`, `GraphBuilder` | [`NamedGraph`], [`GraphBuilder`] |
//! | `FeatureSet`, `FeatureNamespace` | [`FeatureSet`], [`FeatureNamespace`] |
//! | `NodeEmbeddings`, `NodeEmbeddingsBuilder` | [`NodeEmbeddings`], [`NodeEmbeddingsBuilder`] |
//! | `FeatureAggregator`, `NodeEmbedder` | [`algos::aggregator::FeatureAggregator`], [`algos::aggregator::NodeEmbedder`] |
//! | `EmbeddingPropagator`, `PprRankLearner` | [`algos::ep::EmbeddingPropagation::learn_features`], [`algos::pprrank::PprRank::learn_features`] |
//! | `NeighborhoodAligner`, `EmbeddingAligner` | [`algos::alignment::NeighborhoodAligner`], [`algos::emb_aligner::EmbeddingAligner`] |
//! | `TournamentBuilder`, `LSR` | [`algos::lsr::TournamentBuilder`], [`algos::lsr::LSR::learn`] |
//! | `PageRank`, `PolicyEvaluation` | [`algos::pagerank::PageRank::learn`], [`algos::policy_evaluation::PolicyEvaluation::optimize`] |
//!
//! The other algorithms live under [`algos`].  They take the graph itself ([`NamedGraph::graph`])
//! and return results indexed by node id; [`Vocab`] maps ids back to names, and
//! [`algos::utils::top_k_scores`] ranks scores the way the Python walkers do.  Where Python
//! looks a node up by name, Rust returns [`Error::UnknownNode`]; combining structures built on
//! different vocabularies is an [`Error::VocabMismatch`] rather than silently misindexed.
//!
//! See `examples/embed.rs` for an end to end example.

/// Graph storage and the [`graph::Graph`] trait that algorithms run against.
pub mod graph;

/// Graphs with named nodes, and building them edge by edge.
pub mod named_graph;

/// The error type for name lookups and for combining structures.
pub mod error;

/// The algorithms.
pub mod algos;

/// Maps node types and names to internal ids and back.
pub mod vocab;

/// Node and feature embeddings.
pub mod embeddings;

/// Distance functions over embeddings.
pub mod distance;

/// Mapping from nodes to features.
pub mod feature_store;

/// Reading and writing graphs, embeddings and feature files.
pub mod io;

/// Weighted and unweighted sampling of a node's neighbours.
pub mod sampler;

mod autograd;

/// Simple bitset
mod bitset;

/// This interface allows us to update embeddings (and other structures) in multiple threads
/// without having to gain exclusive write access.  Do _not_ clone hogwild structures as they
/// will still point to the underlying data
mod hogwild;

/// Who doesn't like progress bars?
mod progress;

// rustdoc would compile the bindings' numpy-style docstrings as Rust doc-tests.
#[cfg(all(feature = "python", not(doctest)))]
mod python;

/// Seed used when none is given, so runs are deterministic by default.
pub const DEFAULT_SEED: u64 = 20222022;

pub use crate::graph::{CSR, CumCSR, EdgeType, Graph, NodeID};
pub use crate::named_graph::{NamedGraph, GraphBuilder, GraphLoadOptions};
pub use crate::vocab::Vocab;
pub use crate::embeddings::{EmbeddingStore, NodeEmbeddings, NodeEmbeddingsBuilder};
pub use crate::distance::Distance;
pub use crate::feature_store::{FeatureStore, FeatureSet, FeatureNamespace};
pub use crate::error::Error;
