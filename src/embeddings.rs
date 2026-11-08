//! The main Embedding class.  This defines both distance metrics as well as access to the
//! embeddings.
use rayon::prelude::*;
use rand::prelude::*;

use crate::graph::NodeID;
use crate::bitset::AtomicBitSet;
use crate::hogwild::Hogwild;
use crate::algos::graph_ann::{TopK,NodeDistance};
use crate::distance::Distance;

/// Entity allows for adhoc embeddings versus looking up by NodeID within the embedding set
#[derive(Clone,Copy,Debug)]
pub enum Entity<'a> {
    /// Use the embedding defined at NodeID
    Node(NodeID),

    /// Use an adhoc embedding
    Embedding(&'a [f32])
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
    fn test_empty_embedding_buffer_builds_empty_store() {
        let es = EmbeddingBuffer::new().into_store(Distance::Cosine);
        assert_eq!((es.len(), es.dims()), (0, 0));
    }
}
