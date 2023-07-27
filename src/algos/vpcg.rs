use std::fmt::Write;

use rayon::prelude::*;
use hashbrown::HashMap;
use float_ord::FloatOrd;

use crate::progress::CLProgressBar;
use crate::graph::{Graph,NodeID,CDFtoP};
use crate::feature_store::FeatureStore;
use crate::hogwild::Hogwild;

type SparseEmbeddings = Vec<Vec<(usize, f32)>>;

pub struct VPCG {
    pub max_terms: usize,
    pub iterations: usize
}

impl VPCG {

    pub fn learn<G: Graph + Send + Sync>(
        &self,
        graph: &G,
        features: &FeatureStore,
        mask: (&[NodeID], &[NodeID]),
    ) -> SparseEmbeddings {
        
        let mut propagations: SparseEmbeddings = (0..graph.len())
            .map(|node_id| {
                let mut fs = Vec::with_capacity(self.max_terms);
                let feats = features.get_features(node_id);
                let weight = 1. / (feats.len() as f32).sqrt();
                feats.iter().for_each(|f| fs.push((*f, weight)));
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

            let node_set = if iter % 2 == 0 {
                &left
            } else {
                &right
            };

            // For each node, grab the edges and propagate from them to it
            node_set.par_iter().enumerate().for_each(|(node_id, mask)| {
                // Sum up features
                let mut term_map = HashMap::new();
                let (edges, weights) = graph.get_edges(node_id);
                edges.iter().zip(CDFtoP::new(weights)).for_each(|(edge, p)| {
                    let feats = &propagations.get()[*edge];
                    for (feat, score) in feats {
                        let e = term_map.entry(feat).or_insert(0f32);
                        *e += p * score;
                    }
                });

                // L2 norm
                let norm = term_map.par_values().map(|s| s.powf(2.)).sum::<f32>().sqrt();
                let mut results = term_map.into_par_iter().map(|(feature, score)| {
                    (*feature, score / norm)
                }).collect::<Vec<_>>();
                
                // Sort and add to propagations
                results.par_sort_unstable_by_key(|(_, s)| FloatOrd(-*s));
                let node_feats = &mut propagations.get()[node_id];
                node_feats.clear();

                results.into_iter().take(self.max_terms).for_each(|ts| node_feats.push(ts));
                pb.inc(1);
            });
        }

        pb.finish();
        propagations.into_inner().expect("Shouldn't have any other references!")
    }

}

#[cfg(test)]
mod vpcg_tests {
    use super::*;
    use crate::graph::CSR;

    fn build_edges() -> Vec<(usize, usize, f32)> {
        vec![
            (0, 1, 1.),
            (0, 2, 1.),
            (1, 0, 1.),
            (1, 2, 3.),
            (2, 1, 3.),
            (2, 0, 2.5),
            (2, 3, 2.5),
            (3, 2, 2.5),
            (3, 4, 0.5),
            (4, 3, 0.5),
            (4, 5, 0.5),
            (4, 6, 0.5),
            (5, 4, 0.5),
            (5, 6, 0.5),
            (6, 4, 0.5),
            (6, 5, 0.5),
        ]
    }

    #[test]
    fn run_test() {
        let graph = CSR::construct_from_edges(build_edges());
        let es = construct_slpa_embedding(&graph, 10, 0.3, 12345123);
        for node_id in 0..graph.len() {
            println!("{}: {:?}", node_id, es.get_embedding(node_id));
        }
        //panic!();
    }
}
