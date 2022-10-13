use float_ord::FloatOrd;
use crate::graph::NodeID;
use crate::bitset::BitSet;
use crate::hogwild::Hogwild;

pub enum Distance {
    ALT,
    Cosine,
    Euclidean,
    Hamming,
    Jaccard
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
                let cosine_score = dot / (d1.sqrt() * d2.sqrt());
                -cosine_score + 1.
            },

            Distance::Euclidean => {
                e1.iter().zip(e2.iter()).map(|(ei, ej)| {
                    (*ei - *ej).powf(2.)
                }).sum::<f32>().sqrt()
            },

            Distance::Hamming => {
                let not_matches = e1.iter().zip(e2.iter()).map(|(ei, ej)| {
                    if *ei != *ej { 1f32 } else {0f32}
                }).sum::<f32>();
                not_matches / e1.len() as f32
            },
            Distance::Jaccard => {
                let mut idx1 = 0;
                let mut idx2 = 0;
                let mut matches = 0;
                while idx1 < e1.len() && idx2 < e2.len() && e1[idx1] >= 0. && e2[idx2] >= 0. {
                    let v1 = e1[idx1];
                    let v2 = e2[idx2];
                    if v1 == v2 {
                        matches += 1;
                        idx1 += 1;
                        idx2 += 1;
                    } else if v1 < v2 {
                        idx1 += 1;
                    } else {
                        idx2 += 1;
                    }
                }
                while idx1 < e1.len() && e1[idx1] >= 0. {
                    idx1 +=1;
                }
                while idx2 < e2.len() && e2[idx2] >= 0. {
                    idx2 +=1;
                }

                let total_sets = matches + (idx1 - matches) + (idx2 - matches);
                1. - matches as f32 / total_sets as f32
                //( - matches) as f32 / e1.len() as f32
            }
        }
    }
}

pub struct EmbeddingStore {
    dims: usize,
    embeddings: Hogwild<Vec<f32>>,
    bitfield: BitSet,
    distance: Distance,
    nodes: usize
}

impl EmbeddingStore {
    pub fn new(nodes: usize, dims: usize, distance: Distance) -> Self {
        EmbeddingStore {
            dims,
            distance,
            bitfield: BitSet::new(nodes),
            embeddings: Hogwild::new(vec![0.; nodes * dims]),
            nodes
        }
    }

    pub fn is_set(&self, node_id: NodeID) -> bool {
        self.bitfield.is_set(node_id)
    }

    pub fn len(&self) -> usize {
        self.nodes
    }

    pub fn set_embedding(&mut self, node_id: NodeID, embedding: &[f32]) {
        self.get_embedding_mut(node_id).iter_mut().zip(embedding.iter()).for_each(|(ei, wi)| {
            *ei = *wi;
        });
        self.bitfield.set_bit(node_id);
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
        &mut self.embeddings.get()[start..start+self.dims]
    }

    pub fn set_bit(&mut self, node_id: NodeID) {
        self.bitfield.set_bit(node_id);
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

}
