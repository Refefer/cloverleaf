
use float_ord::FloatOrd;
use crate::graph::NodeID;

pub enum Distance {
    ALT,
    Cosine,
    Euclidean
}

impl Distance {
    fn compute(&self, e1: &[f32], e2: &[f32]) -> f32 {
        match &self {
            Distance::ALT => e1.iter().zip(e2.iter())
                .map(|(ei, ej)| (*ei - *ej).abs())
                .max_by_key(|v| FloatOrd(*v)).unwrap_or(0.),

            Distance::Cosine => {
                let mut d1 = 0.;
                let mut d2 = 0.;
                let dot = e1.iter().zip(e2.iter()).map(|(ei, ej)| {
                    d1 += ei.powf(2.);
                    d2 += ej.powf(2.);
                    ei * ej
                }).sum::<f32>();
                -(dot / (d1*d2)) + 1.
            },

            Distance::Euclidean => {
                e1.iter().zip(e2.iter()).map(|(ei, ej)| {
                    (*ei - *ej).powf(2.)
                }).sum::<f32>().sqrt()
            }
        }
    }
}

pub struct EmbeddingStore {
    nodes: usize,
    dims: usize,
    embeddings: Vec<f32>,
    bitfield: Vec<u32>,
    distance: Distance
}

impl EmbeddingStore {
    pub fn new(nodes: usize, dims: usize, distance: Distance) -> Self {
        EmbeddingStore {
            nodes,
            dims,
            distance,
            bitfield: vec![0; (nodes / 4) + 1],
            embeddings: vec![0.; nodes * dims]
        }
    }

    fn get_bit_idx(&self, node_id: &NodeID) -> (usize, u32) {
        let field_offset = node_id / 32;
        let bit_offset = node_id % 32;
        (field_offset, 1u32 << bit_offset)
    }

    pub fn is_set(&self, node_id: NodeID) -> bool {
        let (fo, bm) = self.get_bit_idx(&node_id);
        (self.bitfield[fo] & bm) > 0
    }

    pub fn set_embedding(&mut self, node_id: NodeID, embedding: &[f32]) {
        self.get_embedding_mut(node_id).iter_mut().zip(embedding.iter()).for_each(|(ei, wi)| {
            *ei = *wi;
        });
        let (fo, bm) = self.get_bit_idx(&node_id);
        self.bitfield[fo] |= bm
    }

    fn get_embedding(&self, node_id: NodeID) -> &[f32] {
        let start = node_id * self.dims;
        &self.embeddings[start..start+self.dims]
    }

    fn get_embedding_mut(&mut self, node_id: NodeID) -> &mut [f32] {
        let start = node_id * self.dims;
        &mut self.embeddings[start..start+self.dims]
    }

    pub fn compute_distance(&self, n1: NodeID, n2: NodeID) -> f32 {
        let e1 = self.get_embedding(n1);
        let e2 = self.get_embedding(n2);
        self.distance.compute(e1, e2)
    }
}

#[cfg(test)]
mod embedding_tests {
    use super::*;
    use float_ord::FloatOrd;

    #[test]
    fn test_embeddings() {
        let mut es = EmbeddingStore::new(100, 2, Distance::Euclidean);

        assert_eq!(es.is_set(0), false);
        es.set_embedding(0, &[0., 1.]);
        assert_eq!(es.is_set(0), true);

        assert_eq!(es.is_set(35), false);
        es.set_embedding(35, &[2., 3.]);
        assert_eq!(es.is_set(35), true);

        es.set_embedding(1, &[1., 2.]);

        assert_eq!(es.get_embedding(0), &[0., 1.]);
        assert_eq!(es.get_embedding(1), &[1., 2.]);
        assert_eq!(es.get_embedding(35), &[2., 3.]);

        assert_eq!(es.compute_distance(0, 1), 2f32.sqrt());
        assert_eq!(es.compute_distance(0, 35), 8f32.sqrt());
    }

}
