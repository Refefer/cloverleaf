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
            node_set.par_iter().enumerate().chunks(128).for_each(|chunk| {
                chunk.par_iter().for_each(|(node_id, _mask)| {
                    
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

