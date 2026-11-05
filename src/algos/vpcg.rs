use std::fmt::Write;

use rayon::prelude::*;
use hashbrown::HashMap;
use float_ord::FloatOrd;

use crate::progress::CLProgressBar;
use crate::graph::{Graph,NodeID,CDFtoP};
use crate::feature_store::FeatureStore;
use crate::hogwild::Hogwild;
use crate::algos::utils::FeatureHasher;
use crate::embeddings::EmbeddingStore;
use crate::distance::Distance;

type SparseEmbeddings = Vec<Vec<(usize, f32)>>;

pub enum FeatureWeight {
    Uniform,
    IDF
}

pub struct VPCG {

    // Maximum number of terms to retain
    pub max_terms: usize,

    // Blend coefficient
    pub alpha: f32,

    // Suppress terms under err
    pub err: f32,

    // Number of passes to run
    pub iterations: usize,

    // How to weight the initial features
    pub feature_weight: FeatureWeight
}

impl VPCG {

    pub fn learn<G: Graph + Send + Sync>(
        &self,
        graph: &G,
        features: &FeatureStore,
        mask: (&[NodeID], &[NodeID]),
    ) -> SparseEmbeddings {
        
        let counts = features.count_features();
        let propagations: SparseEmbeddings = (0..graph.len())
            .into_par_iter()
            .map(|node_id| {
                let mut fs = Vec::with_capacity(self.max_terms);
                let feats = features.get_features(node_id);
                let term_map: HashMap<_, _> = feats.iter().map(|fi| {
                    if matches!(self.feature_weight, FeatureWeight::Uniform) {
                        (*fi, 1. / (feats.len() as f32).sqrt())
                    } else {
                        (*fi, 1. / (1f32 + counts[*fi] as f32).ln())
                    }
                }).collect();
                norm_filter_feat(&term_map, self.max_terms, self.err)
                    .for_each(|f| fs.push(f));
                fs
            })
            .collect();

        let propagations = Hogwild::new(propagations);

        let (left, right) = mask;

        let work = ((self.iterations * (left.len()+right.len())) as f32 / 2f32) as u64;
        let pb = CLProgressBar::new(work, true);
        for iter in 0..self.iterations {
            pb.update_message(|msg| {
                msg.clear();
                write!(msg, "Pass {}/{}", iter, self.iterations)
                    .expect("Error writing out indicator message!");
            });

            // Take turns propagating
            let node_set = if iter % 2 == 0 { &left } else { &right };

            // For each node, grab the edges and propagate from them to it
            node_set.par_chunks(128).for_each(|chunk| {
                chunk.par_iter().for_each(|node_id| {
                    
                    // Sum up features
                    let mut term_map = HashMap::new();

                    // If alpha, copy existing features into the term_map
                    if self.alpha < 1f32 {
                        let feats = &propagations.get()[*node_id];
                        feats.iter().for_each(|(feat, score)| {
                            let e = term_map.entry(*feat).or_insert(0f32);
                            *e += (1f32 - self.alpha) * score;
                        });
                    }

                    // Main propagation
                    let (edges, weights) = graph.get_edges(*node_id);
                    edges.iter().zip(CDFtoP::new(weights)).for_each(|(edge, p)| {
                        let feats = &propagations.get()[*edge];
                        feats.iter().for_each(|(feat, score)| {
                            let e = term_map.entry(*feat).or_insert(0f32);
                            *e += self.alpha * p * score;
                        });
                    });

                    // L2 norm and filter
                    let node_feats = &mut propagations.get()[*node_id];
                    node_feats.clear();
                    norm_filter_feat(&term_map, self.max_terms, self.err).for_each(|ts| node_feats.push(ts));
                });
                pb.inc(chunk.len() as u64);
            });
        }

        // Convert it to an embedding store
        pb.finish();

        let props = propagations.into_inner().expect("Shouldn't have any other references!");
        props
    }

    fn build_hash_table(
        num_features: usize,
        dims: usize
    ) -> Vec<[(i8, usize); 3]> {
        let mut hash_lookups = vec![[(1i8, 0usize); 3]; num_features];
        let hasher = FeatureHasher::new(dims);
        hash_lookups.par_iter_mut().enumerate().for_each(|(feat_idx, hashes)| {
            hashes.par_iter_mut().enumerate().for_each(|(hash_num, out)| {
                *out = hasher.hash(feat_idx, hash_num);
            });
        });
        hash_lookups
    }

    pub fn convert_to_es(
        num_features: usize, 
        dims: usize,
        embeddings: SparseEmbeddings
    ) -> EmbeddingStore {
        let hash_table = VPCG::build_hash_table(num_features, dims);
        let embs = EmbeddingStore::new(embeddings.len(), dims, Distance::Cosine);
        embeddings.into_par_iter().enumerate().for_each(|(node_id, sparse_emb)| {
            let emb = embs.get_embedding_mut_hogwild(node_id);
            if sparse_emb.len() == 0 {
                panic!("node missing features!");
            }
            for (feat, weight) in sparse_emb {
                for (sign, dim) in hash_table[feat] {
                    emb[dim] += sign as f32 * weight;
                }
            }
        });
        embs
    }

}

fn l2norm(
    term_map: &HashMap<usize, f32>
) -> Vec<(usize, f32)> {
    let norm = term_map.par_values().map(|s| s.powf(2.)).sum::<f32>().sqrt();
    term_map.into_par_iter().map(|(feature, score)| {
        (*feature, score / norm)
    }).collect()
}

fn norm_filter_feat<'a>(
    term_map: &HashMap<usize, f32>,
    max_terms: usize,
    err: f32
) -> impl Iterator<Item=(usize, f32)> {
    let mut results = l2norm(term_map);
    results.par_sort_unstable_by_key(|(_, s)| FloatOrd(-*s));
    results.into_iter().take(max_terms)
        .filter(move |(_t, s)| *s > err)
}


#[cfg(test)]
mod vpcg_tests {
    use super::*;
    use crate::graph::{CSR, CumCSR};

    fn feature_ids(props: &SparseEmbeddings, node: NodeID) -> Vec<usize> {
        let mut ids: Vec<_> = props[node].iter().map(|(f, _)| *f).collect();
        ids.sort();
        ids
    }

    /// Left/right sides are interleaved node ids, so the position in `left` differs from the node
    /// id.  Only the left side should be updated on the first pass.
    #[test]
    fn test_learn_updates_masked_node_ids() {
        // Bipartite: left = {1, 3}, right = {0, 2}, fully connected across sides.
        let mut edges = Vec::new();
        for l in [1usize, 3] {
            for r in [0usize, 2] {
                edges.push((l, r, 1f32));
                edges.push((r, l, 1f32));
            }
        }
        let graph = CumCSR::convert(CSR::construct_from_edges(edges, false));

        let mut fs = FeatureStore::new(4);
        for node in 0..4 {
            fs.set_features(node, [("f", node.to_string())].into_iter());
        }
        // Feature ids line up with node ids by construction.
        for node in 0..4 {
            assert_eq!(fs.get_features(node), &[node]);
        }

        let vpcg = VPCG {
            max_terms: 10,
            alpha: 1.0,
            err: 0.0,
            iterations: 1,
            feature_weight: FeatureWeight::Uniform
        };

        let left = [1usize, 3];
        let right = [0usize, 2];
        let props = vpcg.learn(&graph, &fs, (&left, &right));

        // Pass 0 propagates into the left side from its (right side) neighbors.
        assert_eq!(feature_ids(&props, 1), vec![0, 2]);
        assert_eq!(feature_ids(&props, 3), vec![0, 2]);

        // Right side is untouched on pass 0.
        assert_eq!(feature_ids(&props, 0), vec![0]);
        assert_eq!(feature_ids(&props, 2), vec![2]);
    }
}
