//! The main Embedding class.  This defines both distance metrics as well as access to the
//! embeddings.
use rayon::prelude::*;
use rand::prelude::*;

use crate::graph::NodeID;
use crate::bitset::AtomicBitSet;
use crate::hogwild::Hogwild;
use crate::algos::graph_ann::{TopK,NodeDistance};
use crate::distance::Distance;
use crate::error::Error;
use crate::io::{EmbeddingLoadError, EmbeddingReader, EmbeddingWriter};
use crate::vocab::Vocab;
use std::collections::HashSet;
use std::sync::Arc;

/// Entity allows for adhoc embeddings versus looking up by NodeID within the embedding set
#[derive(Clone,Copy,Debug)]
pub enum Entity<'a> {
    /// Use the embedding defined at NodeID
    Node(NodeID),

    /// Use an adhoc embedding
    Embedding(&'a [f32])
}

/// What to look an embedding up by in a [`NodeEmbeddings`]: a node by name or by id, or an
/// embedding given directly.  [`NodeEmbeddings::resolve`] turns it into the embedding.
#[derive(Clone,Debug,PartialEq)]
pub enum Query {
    /// A node by (node type, name).
    Node(String, String),

    /// A node by id.
    NodeId(NodeID),

    /// An embedding, used as given.
    Embedding(Vec<f32>)
}

/// The core Embedding Store used everywhere.
#[derive(Clone)]
pub struct EmbeddingStore {
    /// Dimensions of each embedding
    dims: usize,

    /// The embeddings are a congiuous vector, wrapped in a Hogwild algorithm so they can be
    /// updated in parallal.
    embeddings: Hogwild<Vec<f32>>,

    /// Bitfield measuring if an embedding has been set
    bitfield: AtomicBitSet,

    /// distance metric to use
    distance: Distance,

    /// Number of nodes in the Embedding Store
    nodes: usize
}

impl EmbeddingStore {
    pub fn new(nodes: usize, dims: usize, distance: Distance) -> Self {
        EmbeddingStore {
            dims,
            distance,
            bitfield: AtomicBitSet::new(nodes),
            embeddings: Hogwild::new(vec![0.; nodes * dims]),
            nodes
        }
    }

    pub fn new_with_vec(
        nodes: usize, 
        dims: usize, 
        distance: Distance,
        vec: Vec<f32>
    ) -> Option<Self> {
        if vec.len() != nodes * dims {
            None
        } else {
            //
            let bitfield = AtomicBitSet::new(nodes);
            (0..nodes).for_each(|node_id| bitfield.set_bit(node_id));

            let es = EmbeddingStore {
                dims,
                distance,
                bitfield: bitfield,
                embeddings: Hogwild::new(vec),
                nodes
            };
            Some(es)
        }
    }


    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn is_set(&self, node_id: NodeID) -> bool {
        self.bitfield.is_set(node_id)
    }

    pub fn len(&self) -> usize {
        self.nodes
    }

    pub fn distance(&self) -> Distance {
        self.distance
    }

    /// Sets a node's embedding.  A wrong-length embedding is rejected without writing anything.
    pub fn set_embedding(
        &mut self, 
        node_id: NodeID, 
        embedding: &[f32]
    ) -> Result<(), DimensionMismatch> {
        if embedding.len() != self.dims {
            return Err(DimensionMismatch { expected: self.dims, actual: embedding.len() })
        }
        self.get_embedding_mut(node_id).copy_from_slice(embedding);
        self.bitfield.set_bit(node_id);
        Ok(())
    }

    pub fn get_embedding(&self, node_id: NodeID) -> &[f32] {
        let start = node_id * self.dims;
        &self.embeddings[start..start+self.dims]
    }

    pub fn get_embedding_mut(&mut self, node_id: NodeID) -> &mut [f32] {
        let start = node_id * self.dims;
        self.bitfield.set_bit(node_id);
        &mut self.embeddings[start..start+self.dims]
    }

    pub fn get_embedding_mut_hogwild(&self, node_id: NodeID) -> &mut [f32] {
        let start = node_id * self.dims;
        self.bitfield.set_bit(node_id);
        &mut self.embeddings.get()[start..start+self.dims]
    }

    pub fn set_bit(&mut self, node_id: NodeID) {
        self.bitfield.set_bit(node_id);
    }

    fn extract_vec<'a>(&'a self, n: &Entity<'a>) -> &'a [f32] {
        match n {
            Entity::Node(node_id) => self.get_embedding(*node_id),
            Entity::Embedding(vec) => vec
        }
    }

    pub fn compute_distance<'a>(&self, n1: &Entity<'a>, n2: &Entity<'a>) -> f32 {
        let e1 = self.extract_vec(n1);
        let e2 = self.extract_vec(n2);

        self.distance.compute(e1, e2)
    }

    pub fn score_all<'a>(
        &self, 
        q: &Entity<'a>
    ) -> EmbeddingStore {
        let es = EmbeddingStore::new(self.len(), 1, self.distance.clone());

        let query_emb = self.extract_vec(q);
        (0..self.len()).into_par_iter().for_each(|node_id| {
            let e2 = self.get_embedding(node_id);
            es.get_embedding_mut_hogwild(node_id)[0] = self.distance.compute(query_emb, e2);
        });
        es
    }

    pub fn nearest_neighbor<'a,F>(
        &self, 
        q: &Entity<'a>, 
        k: usize,
        filter: F
    ) -> Vec<NodeDistance>  
        where F: Sync + Fn(NodeID) -> bool 
    {
        let query_emb = self.extract_vec(q);
        (0..self.len()).into_par_iter().filter(|node_id| filter(*node_id)).map(|node_id| {
            let node_emb = self.get_embedding(node_id);
            (node_id, self.distance.compute(query_emb, node_emb))
        }).fold(|| TopK::new(k), |mut acc, (node_id, dist)| {
            acc.push(node_id, dist);
            acc
        }).reduce(|| TopK::new(k),|mut tk1, tk2| {
            tk1.extend(tk2);
            tk1
        }).into_sorted()
    }

}

/// An embedding whose length doesn't match the dimensions it is meant for.
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct DimensionMismatch {
    pub expected: usize,
    pub actual: usize
}

/// Row-major embeddings built up one node at a time, as a single flat buffer so building millions
/// of nodes costs no per-node allocation and converts to an EmbeddingStore without copying.  The
/// first embedding fixes the dimensions.
#[derive(Debug,Default)]
pub struct EmbeddingBuffer {
    dims: Option<usize>,
    nodes: usize,
    data: Vec<f32>
}

impl EmbeddingBuffer {
    pub fn new() -> Self {
        EmbeddingBuffer::default()
    }

    pub fn len(&self) -> usize {
        self.nodes
    }

    /// Checks an embedding length against the dimensions fixed so far.
    pub fn check(&self, len: usize) -> Result<(), DimensionMismatch> {
        match self.dims {
            Some(dims) if dims != len => Err(DimensionMismatch { expected: dims, actual: len }),
            _ => Ok(())
        }
    }

    /// Sets an existing node or appends the next one.  Node ids must be assigned densely, as a
    /// Vocab does, so `node_id` may be at most `len()`.
    pub fn set(&mut self, node_id: NodeID, embedding: &[f32]) -> Result<(), DimensionMismatch> {
        self.check(embedding.len())?;
        let dims = *self.dims.get_or_insert(embedding.len());
        if node_id < self.nodes {
            self.data[node_id * dims..(node_id + 1) * dims].copy_from_slice(embedding);
        } else {
            assert_eq!(node_id, self.nodes, "node ids must be added densely");
            self.data.extend_from_slice(embedding);
            self.nodes += 1;
        }
        Ok(())
    }

    /// Converts into an EmbeddingStore, reusing the buffer.  An empty buffer has no dimensions.
    pub fn into_store(self, distance: Distance) -> EmbeddingStore {
        let dims = self.dims.unwrap_or(0);
        EmbeddingStore::new_with_vec(self.nodes, dims, distance, self.data)
            .expect("buffer holds nodes * dims values")
    }
}

/// Embeddings for named nodes: an [`EmbeddingStore`] indexed by the ids of a [`Vocab`].
#[derive(Clone)]
pub struct NodeEmbeddings {
    pub(crate) vocab: Arc<Vocab>,
    pub(crate) embeddings: EmbeddingStore
}

impl NodeEmbeddings {
    /// Distance the Python bindings load embeddings with when none is given.
    pub const DEFAULT_DISTANCE: Distance = Distance::Cosine;

    /// Pairs a store with the vocab whose ids index it.
    pub fn new(vocab: Arc<Vocab>, embeddings: EmbeddingStore) -> Result<Self, Error> {
        if vocab.len() != embeddings.len() {
            return Err(Error::InvalidArgument(format!(
                "vocab has {} nodes but the embeddings have {}", vocab.len(), embeddings.len())))
        }
        Ok(NodeEmbeddings { vocab, embeddings })
    }

    /// Zeroed embeddings for every node in `vocab`.
    pub fn zeros(vocab: Arc<Vocab>, dims: usize, distance: Distance) -> Self {
        let embeddings = EmbeddingStore::new(vocab.len(), dims, distance);
        NodeEmbeddings { vocab, embeddings }
    }

    /// Embeddings from `((node type, name), embedding)` pairs, ids in the order first listed; a
    /// node listed again takes the later embedding.  Every embedding must have as many
    /// dimensions as the first, else an [`Error::DimensionMismatch`].  An empty list gives no
    /// nodes and 0 dimensions.
    pub fn from_list<I, A, B, E>(list: I, distance: Distance) -> Result<Self, Error>
    where
        I: IntoIterator<Item = ((A, B), E)>,
        A: AsRef<str>,
        B: AsRef<str>,
        E: AsRef<[f32]>
    {
        let mut builder = NodeEmbeddingsBuilder::new(distance);
        for ((node_type, name), embedding) in list {
            builder.add_embedding(node_type.as_ref(), name.as_ref(), embedding.as_ref())?;
        }
        Ok(builder.build().unwrap_or_else(|| NodeEmbeddings::zeros(Arc::new(Vocab::new()), 0, distance)))
    }

    /// One-dimensional embeddings holding a score per node, as PageRank and LSR produce.
    pub fn from_scores(vocab: Arc<Vocab>, scores: &[f64]) -> Result<Self, Error> {
        let values = scores.iter().map(|s| *s as f32).collect();
        let embeddings = EmbeddingStore::new_with_vec(scores.len(), 1, Distance::Euclidean, values)
            .expect("one value per node");
        NodeEmbeddings::new(vocab, embeddings)
    }

    /// Loads a `node_type<TAB>name<TAB>[e1,e2,...]` file, keeping only `node_types` when given.
    pub fn load(
        path: &str,
        distance: Distance,
        node_types: Option<&HashSet<String>>,
        chunk_size: Option<usize>,
        skip_rows: Option<usize>
    ) -> Result<Self, EmbeddingLoadError> {
        let (vocab, embeddings) = EmbeddingReader::load(
            path, distance, &node_types, chunk_size, skip_rows)?;
        Ok(NodeEmbeddings { vocab: Arc::new(vocab), embeddings })
    }

    /// Writes the embeddings in the format `load` reads, gzipped when the path ends in `.gz`.
    pub fn save(&self, path: &str, comp_level: Option<u32>) -> std::io::Result<()> {
        let mut writer = EmbeddingWriter::new(path, self.vocab.as_ref(), comp_level)?;
        writer.stream((0..self.len()).map(|node_id| (node_id, self.embeddings.get_embedding(node_id))))?;
        writer.finish()
    }

    pub fn vocab(&self) -> &Arc<Vocab> {
        &self.vocab
    }

    pub fn store(&self) -> &EmbeddingStore {
        &self.embeddings
    }

    pub fn store_mut(&mut self) -> &mut EmbeddingStore {
        &mut self.embeddings
    }

    pub fn into_parts(self) -> (Arc<Vocab>, EmbeddingStore) {
        (self.vocab, self.embeddings)
    }

    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    pub fn dims(&self) -> usize {
        self.embeddings.dims()
    }

    pub fn distance(&self) -> Distance {
        self.embeddings.distance()
    }

    pub fn node_id(&self, node_type: &str, name: &str) -> Result<NodeID, Error> {
        self.vocab.node_id(node_type, name)
    }

    pub fn contains(&self, node_type: &str, name: &str) -> bool {
        self.vocab.get_node_id(node_type, name).is_some()
    }

    pub fn get_embedding(&self, node_type: &str, name: &str) -> Result<&[f32], Error> {
        Ok(self.embeddings.get_embedding(self.node_id(node_type, name)?))
    }

    /// The node at `node_id` and its embedding, if there is one.
    pub fn get(&self, node_id: NodeID) -> Option<((String, String), &[f32])> {
        let name = self.vocab.get_name_owned(node_id)?;
        (node_id < self.len()).then(|| (name, self.embeddings.get_embedding(node_id)))
    }

    /// The embedding at `node_id`, erroring when it's out of range.
    pub fn get_embedding_by_id(&self, node_id: NodeID) -> Result<&[f32], Error> {
        if node_id < self.len() {
            Ok(self.embeddings.get_embedding(node_id))
        } else {
            Err(Error::NodeIdOutOfRange { node_id, len: self.len() })
        }
    }

    pub fn set_embedding(&mut self, node_type: &str, name: &str, embedding: &[f32]) -> Result<(), Error> {
        let node_id = self.node_id(node_type, name)?;
        Ok(self.embeddings.set_embedding(node_id, embedding)?)
    }

    /// The embedding `query` refers to: a node's here, or the query's own embedding, which isn't
    /// checked against these embeddings' dimensions.  Errors with [`Error::UnknownNode`] or
    /// [`Error::NodeIdOutOfRange`] when the node isn't here.
    pub fn resolve<'a>(&'a self, query: &'a Query) -> Result<&'a [f32], Error> {
        match query {
            Query::Node(node_type, name) => self.get_embedding(node_type, name),
            Query::NodeId(node_id) => self.get_embedding_by_id(*node_id),
            Query::Embedding(embedding) => Ok(embedding)
        }
    }

    /// The distance between what two queries refer to, by these embeddings' distance.
    pub fn compute_distance(&self, a: &Query, b: &Query) -> Result<f32, Error> {
        Ok(self.distance().compute(self.resolve(a)?, self.resolve(b)?))
    }

    /// The `k` nodes closest to `embedding`, closest first, optionally only those whose type is
    /// in `node_types`.
    pub fn nearest_neighbor(
        &self,
        embedding: &[f32],
        k: usize,
        node_types: Option<&HashSet<String>>
    ) -> Vec<NodeDistance> {
        let query = Entity::Embedding(embedding);
        match node_types {
            Some(nts) => self.embeddings.nearest_neighbor(&query, k, |id| self.vocab.has_node_type(id, nts)),
            None => self.embeddings.nearest_neighbor(&query, k, |_id| true)
        }
    }

    /// Scales every embedding to unit length.  All-zero embeddings are left as they are.
    pub fn l2_normalize(&mut self) {
        let es = &self.embeddings;
        (0..es.len()).into_par_iter().for_each(|idx| {
            let e = es.get_embedding_mut_hogwild(idx);
            let norm = e.iter().map(|ei| ei.powf(2.)).sum::<f32>().sqrt();
            if norm > 0f32 {
                e.iter_mut().for_each(|ei| *ei /= norm);
            }
        });
    }
}

/// Builds [`NodeEmbeddings`] one named node at a time.  The first embedding fixes the dimensions.
pub struct NodeEmbeddingsBuilder {
    vocab: Vocab,
    distance: Distance,
    embeddings: EmbeddingBuffer
}

impl NodeEmbeddingsBuilder {
    pub fn new(distance: Distance) -> Self {
        NodeEmbeddingsBuilder { vocab: Vocab::new(), distance, embeddings: EmbeddingBuffer::new() }
    }

    pub fn distance(&self) -> Distance {
        self.distance
    }

    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Adds or replaces a node's embedding.  A rejected embedding leaves the builder unchanged.
    pub fn add_embedding(
        &mut self,
        node_type: &str,
        name: &str,
        embedding: &[f32]
    ) -> Result<(), DimensionMismatch> {
        // Check before inserting, so a rejected node doesn't stay in the vocab.
        self.embeddings.check(embedding.len())?;
        let node_id = self.vocab.get_or_insert(node_type, name);
        self.embeddings.set(node_id, embedding)
    }

    /// Takes what's been added, leaving the builder empty.  None if nothing was added.
    pub fn build(&mut self) -> Option<NodeEmbeddings> {
        if self.embeddings.len() == 0 {
            return None
        }
        // The flat buffer becomes the store without a copy.
        let vocab = std::mem::replace(&mut self.vocab, Vocab::new());
        let embeddings = std::mem::take(&mut self.embeddings).into_store(self.distance);
        Some(NodeEmbeddings { vocab: Arc::new(vocab), embeddings })
    }
}

/// Randomize embeddings.  
pub fn randomize_embedding_store(es: &mut EmbeddingStore, rng: &mut impl Rng) {
    for idx in 0..es.len() {
        let e = es.get_embedding_mut(idx);
        let mut norm = 0f32;
        e.iter_mut().for_each(|ei| {
            *ei = 2f32 * rng.gen::<f32>() - 1f32;
            norm += ei.powf(2f32);
        });
        norm = norm.sqrt();
        e.iter_mut().for_each(|ei| *ei /= norm);
    }
}

#[cfg(test)]
mod embedding_tests {
    use super::*;

    #[test]
    fn test_hogwild_writes_mark_embedding_set() {
        let es = EmbeddingStore::new(40, 2, Distance::Euclidean);
        assert_eq!(es.is_set(35), false);
        es.get_embedding_mut_hogwild(35)[0] = 1.;
        assert_eq!(es.is_set(35), true);
        assert_eq!(es.is_set(34), false);
    }

    #[test]
    fn test_embeddings() {
        let mut es = EmbeddingStore::new(100, 2, Distance::Euclidean);

        assert_eq!(es.is_set(0), false);
        es.set_embedding(0, &[0., 1.]).unwrap();
        assert_eq!(es.is_set(0), true);

        assert_eq!(es.is_set(35), false);
        es.set_embedding(35, &[2., 3.]).unwrap();
        assert_eq!(es.is_set(35), true);

        es.set_embedding(1, &[1., 2.]).unwrap();

        assert_eq!(es.get_embedding(0), &[0., 1.]);
        assert_eq!(es.get_embedding(1), &[1., 2.]);
        assert_eq!(es.get_embedding(35), &[2., 3.]);

        assert_eq!(es.compute_distance(&Entity::Node(0), &Entity::Node(1)), 2f32.sqrt());
        assert_eq!(es.compute_distance(&Entity::Node(0), &Entity::Node(35)), 8f32.sqrt());
    }

    #[test]
    fn test_distances() {
        let alt_d = Distance::ALT.compute(&[1., 2., 1.], &[3., 2., 4.]);
        assert_eq!(alt_d, 3.);

        let cosine_d = Distance::Cosine.compute(&[1., 2., 1.], &[3., 2., 4.]);
        let dot = 3. + 4. + 4.;
        let n1 = (1f32 + 4. + 1.).sqrt();
        let n2 = (9f32 + 4. + 16.).sqrt();
        let cosine_score = dot / (n1 * n2);
        assert_eq!(cosine_d, -cosine_score + 1.);

        let euclidean_d = Distance::Euclidean.compute(&[1., 2., 1.], &[3., 2., 4.]);
        assert_eq!(euclidean_d, (4f32 + 0. + 9.).sqrt());

        let hamming_d = Distance::Hamming.compute(&[1., 2., 1.], &[3., 2., 4.]);
        assert_eq!(hamming_d, 2./3.);

        let overlap_d = Distance::Jaccard.compute(&[1., 2., -1.], &[2., 4., 5.]);
        assert_eq!(overlap_d, 1. - 1. / 4.);
    }


    /// Nodes rejected by the filter must not be returned, even if fewer than k pass.
    #[test]
    fn test_nearest_neighbor_excludes_filtered_nodes() {
        let es = EmbeddingStore::new(3, 1, Distance::Euclidean);
        for node in 0..3 {
            es.get_embedding_mut_hogwild(node)[0] = node as f32;
        }
        let q = [0f32];
        let res = es.nearest_neighbor(&Entity::Embedding(&q), 2, |node_id| node_id == 2);
        let ids: Vec<_> = res.iter().map(|nd| nd.to_tup_cloned().0).collect();
        assert_eq!(ids, vec![2]);
    }

    /// A wrong-length embedding must not be partially written.
    #[test]
    fn test_set_embedding_rejects_wrong_length() {
        let mut es = EmbeddingStore::new(2, 3, Distance::Euclidean);
        assert_eq!(es.set_embedding(0, &[1.]), Err(DimensionMismatch { expected: 3, actual: 1 }));
        assert!(es.set_embedding(1, &[1., 2., 3., 4.]).is_err());
        assert_eq!(es.get_embedding(0), &[0., 0., 0.]);
        assert_eq!(es.get_embedding(1), &[0., 0., 0.]);
        assert!(!es.is_set(0) && !es.is_set(1));
    }

    /// The first embedding fixes the dimensions; later ones must match, and a rejected
    /// embedding leaves the buffer unchanged.
    #[test]
    fn test_embedding_buffer_enforces_dims() {
        let mut buf = EmbeddingBuffer::new();
        buf.set(0, &[1., 2., 3.]).unwrap();
        assert_eq!(buf.set(1, &[1., 2., 3., 4., 5.]),
                   Err(DimensionMismatch { expected: 3, actual: 5 }));
        assert_eq!(buf.set(1, &[1., 2.]), Err(DimensionMismatch { expected: 3, actual: 2 }));
        assert_eq!(buf.len(), 1);
        buf.set(1, &[4., 5., 6.]).unwrap();
        buf.set(0, &[7., 8., 9.]).unwrap();

        let es = buf.into_store(Distance::Euclidean);
        assert_eq!((es.len(), es.dims()), (2, 3));
        assert_eq!(es.get_embedding(0), &[7., 8., 9.]);
        assert_eq!(es.get_embedding(1), &[4., 5., 6.]);
    }

    #[test]
    fn test_queries_resolve_by_name_id_or_embedding() {
        let mut b = NodeEmbeddingsBuilder::new(Distance::Euclidean);
        b.add_embedding("n", "a", &[0., 0.]).unwrap();
        b.add_embedding("n", "b", &[3., 4.]).unwrap();
        let ne = b.build().unwrap();

        let by_name = Query::Node("n".into(), "b".into());
        assert_eq!(ne.resolve(&by_name).unwrap(), &[3., 4.]);
        assert_eq!(ne.resolve(&Query::NodeId(0)).unwrap(), &[0., 0.]);
        assert_eq!(ne.resolve(&Query::Embedding(vec![1.])).unwrap(), &[1.]);
        assert!(matches!(ne.resolve(&Query::Node("n".into(), "zzz".into())), Err(Error::UnknownNode { .. })));
        assert!(matches!(ne.resolve(&Query::NodeId(2)), Err(Error::NodeIdOutOfRange { node_id: 2, len: 2 })));

        assert_eq!(ne.compute_distance(&Query::NodeId(0), &by_name).unwrap(), 5.);
        assert_eq!(ne.compute_distance(&Query::Embedding(vec![3., 0.]), &by_name).unwrap(), 4.);
        assert!(ne.compute_distance(&by_name, &Query::NodeId(7)).is_err());
    }

    #[test]
    fn test_empty_embedding_buffer_builds_empty_store() {
        let es = EmbeddingBuffer::new().into_store(Distance::Cosine);
        assert_eq!((es.len(), es.dims()), (0, 0));
    }
}
