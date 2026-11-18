//! The error type for operations that look nodes up by name or combine structures built on
//! different vocabularies.  The file loaders have their own, more specific errors
//! ([`GraphLoadError`](crate::io::GraphLoadError), [`EmbeddingLoadError`](crate::io::EmbeddingLoadError),
//! [`FeatureFileError`](crate::io::FeatureFileError)), which convert into this one.
use std::fmt;

use crate::embeddings::DimensionMismatch;
use crate::io::{EmbeddingLoadError, FeatureFileError, GraphLoadError};

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),

    /// A node isn't in the vocabulary.
    UnknownNode { node_type: String, name: String },

    /// A node id is past the end of the structure it indexes.
    NodeIdOutOfRange { node_id: usize, len: usize },

    /// Two structures don't give the same nodes the same ids.  Names what was being combined.
    VocabMismatch(&'static str),

    /// An embedding has the wrong number of dimensions.
    DimensionMismatch(DimensionMismatch),

    /// A malformed line in an input file.  `line` is 1-based.
    Malformed { line: usize, message: String },

    /// An argument outside its allowed range.
    InvalidArgument(String),

    GraphLoad(GraphLoadError),
    EmbeddingLoad(EmbeddingLoadError),
    FeatureFile(FeatureFileError)
}

impl Error {
    pub(crate) fn unknown_node(node_type: &str, name: &str) -> Self {
        Error::UnknownNode { node_type: node_type.to_string(), name: name.to_string() }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "{:?}", e),
            Error::UnknownNode { node_type, name } =>
                write!(f, "Node '{}:{}' does not exist!", node_type, name),
            Error::NodeIdOutOfRange { node_id, len } =>
                write!(f, "Node id {} is out of range for {} nodes", node_id, len),
            Error::VocabMismatch(what) => write!(f,
                "{} must have the same nodes in the same order (build one from the other)", what),
            Error::DimensionMismatch(e) =>
                write!(f, "Embedding has {} dimensions, expected {}", e.actual, e.expected),
            Error::Malformed { line, message } => write!(f, "Line {}: {}", line, message),
            Error::InvalidArgument(message) => write!(f, "{}", message),
            Error::GraphLoad(e) => write!(f, "{}", e),
            Error::EmbeddingLoad(e) => write!(f, "{}", e),
            Error::FeatureFile(e) => write!(f, "{}", e)
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self { Error::Io(e) }
}

impl From<DimensionMismatch> for Error {
    fn from(e: DimensionMismatch) -> Self { Error::DimensionMismatch(e) }
}

impl From<GraphLoadError> for Error {
    fn from(e: GraphLoadError) -> Self { Error::GraphLoad(e) }
}

impl From<EmbeddingLoadError> for Error {
    fn from(e: EmbeddingLoadError) -> Self { Error::EmbeddingLoad(e) }
}

impl From<FeatureFileError> for Error {
    fn from(e: FeatureFileError) -> Self { Error::FeatureFile(e) }
}

/// Errors unless node ids from `a` index the same nodes in `b`.
pub fn ensure_aligned(
    a: &crate::vocab::Vocab,
    b: &crate::vocab::Vocab,
    what: &'static str
) -> Result<(), Error> {
    if a.is_aligned_with(b) { Ok(()) } else { Err(Error::VocabMismatch(what)) }
}
