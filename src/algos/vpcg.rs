use std::collections::HashSet;
use std::fmt::Write;
use std::sync::Arc;

use rayon::prelude::*;
use hashbrown::HashMap;
use float_ord::FloatOrd;

use crate::progress::CLProgressBar;
use crate::graph::{Graph,NodeID,CDFtoP};
use crate::feature_store::FeatureStore;
use crate::hogwild::Hogwild;
use crate::algos::utils::FeatureHasher;
use crate::embeddings::{EmbeddingStore, NodeEmbeddings};
use crate::distance::Distance;
use crate::error::Error;
use crate::feature_store::FeatureSet;
use crate::named_graph::NamedGraph;
use crate::vocab::Vocab;

type SparseEmbeddings = Vec<Vec<(usize, f32)>>;

/// How VPCG weighs each node's own features before propagating.  Either way a node's weights are
/// then scaled to unit length.
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub enum FeatureWeight {
    /// Every feature on a node weighs the same.
    #[default]
    Uniform,

    /// A feature weighs 1 / ln(1 + the number of nodes it's on), so rare features count more.
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
    /// Blend coefficient the Python bindings use when none is given: only propagated features.
    pub const DEFAULT_ALPHA: f32 = 1.0;

    /// Weight under which the Python bindings drop a term when none is given.
    pub const DEFAULT_ERR: f32 = 1e-5;

    /// Keeps up to `max_terms` features per node over `iterations` passes, with the defaults the
    /// Python bindings use: [`DEFAULT_ALPHA`](Self::DEFAULT_ALPHA),
    /// [`DEFAULT_ERR`](Self::DEFAULT_ERR) and [`FeatureWeight::Uniform`].
    pub fn new(max_terms: usize, iterations: usize) -> Self {
        VPCG {
            max_terms,
            alpha: VPCG::DEFAULT_ALPHA,
            err: VPCG::DEFAULT_ERR,
            iterations,
            feature_weight: FeatureWeight::default()
        }
    }

    /// Learns feature mappings for every node of a bipartite `graph`: passes alternate between
    /// updating the nodes whose type is in `start_types` (first) and the rest.  `features` must
    /// index the graph's nodes, else an [`Error::VocabMismatch`]; featureless nodes are given a
    /// feature of their own first.
    pub fn learn_named(
        &self,
        graph: &NamedGraph,
        features: &mut FeatureSet,
        start_types: &HashSet<String>
    ) -> Result<VpcgFeatureMappings, Error> {
        features.prepare_for_learning(graph, None)?;
        let (left, right) = graph.split_by_node_types(start_types);
        let maps = self.learn(graph.graph(), features.store(), (&left, &right));
        VpcgFeatureMappings::new(graph.vocab().clone(), maps)
    }

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

    /// Hashes sparse feature mappings, indexed by node id, into `dims`-dimensional embeddings
    /// with cosine distance.  Feature ids must be under `num_features`.  Errors with
    /// [`Error::InvalidArgument`] when `dims` is 0, a feature id is out of range or a node has no
    /// features, which would leave it a zero embedding that cosine distance can't compare.
    pub fn convert_to_es(
        num_features: usize, 
        dims: usize,
        embeddings: SparseEmbeddings
    ) -> Result<EmbeddingStore, Error> {
        if dims == 0 {
            return Err(Error::InvalidArgument("dims must be at least 1".into()))
        }
        if let Some(node_id) = embeddings.iter().position(|e| e.is_empty()) {
            return Err(Error::InvalidArgument(format!("node {} has no features to embed", node_id)))
        }
        if let Some(feat) = embeddings.iter().flatten().map(|(f, _)| *f).find(|f| *f >= num_features) {
            return Err(Error::InvalidArgument(format!(
                "feature id {} is out of range for {} features", feat, num_features)))
        }
        let hash_table = VPCG::build_hash_table(num_features, dims);
        let embs = EmbeddingStore::new(embeddings.len(), dims, Distance::Cosine);
        embeddings.into_par_iter().enumerate().for_each(|(node_id, sparse_emb)| {
            let emb = embs.get_embedding_mut_hogwild(node_id);
            for (feat, weight) in sparse_emb {
                for (sign, dim) in hash_table[feat] {
                    emb[dim] += sign as f32 * weight;
                }
            }
        });
        Ok(embs)
    }

}

/// What [`VPCG::learn_named`] learns: each node's propagated features, as (feature id, weight)
/// pairs that are heaviest first and have unit length together.  Feature ids index the
/// [`FeatureSet`]'s features.
#[derive(Clone,Debug)]
pub struct VpcgFeatureMappings {
    vocab: Arc<Vocab>,
    maps: SparseEmbeddings
}

impl VpcgFeatureMappings {
    /// Pairs mappings indexed by node id, as [`VPCG::learn`] returns them, with the vocab that
    /// names the nodes.  [`Error::InvalidArgument`] unless there's one mapping per node.
    pub fn new(vocab: Arc<Vocab>, maps: Vec<Vec<(usize, f32)>>) -> Result<Self, Error> {
        if vocab.len() != maps.len() {
            return Err(Error::InvalidArgument(format!(
                "vocab has {} nodes but there are {} feature mappings", vocab.len(), maps.len())))
        }
        Ok(VpcgFeatureMappings { vocab, maps })
    }

    pub fn vocab(&self) -> &Arc<Vocab> {
        &self.vocab
    }

    /// Every node's mapping, indexed by node id.
    pub fn maps(&self) -> &[Vec<(usize, f32)>] {
        &self.maps
    }

    pub fn len(&self) -> usize {
        self.maps.len()
    }

    /// A node's mapping, or [`Error::UnknownNode`].
    pub fn get(&self, node_type: &str, name: &str) -> Result<&[(usize, f32)], Error> {
        Ok(&self.maps[self.vocab.node_id(node_type, name)?])
    }

    /// The mapping of the node at `node_id`, or [`Error::NodeIdOutOfRange`].
    pub fn get_by_id(&self, node_id: NodeID) -> Result<&[(usize, f32)], Error> {
        self.maps.get(node_id).map(|m| m.as_slice())
            .ok_or(Error::NodeIdOutOfRange { node_id, len: self.maps.len() })
    }

    /// Hashes the mappings into `dims`-dimensional embeddings with cosine distance, freeing them
    /// as it goes.  Errors with [`Error::InvalidArgument`] when `dims` is 0 or a node was left
    /// without features (no neighbours to propagate from, every weight under `err`, or a
    /// `max_terms` of 0), as its embedding would be all zeros.
    pub fn into_embeddings(self, dims: usize) -> Result<NodeEmbeddings, Error> {
        if let Some(node_id) = self.maps.iter().position(|m| m.is_empty()) {
            let (node_type, name) = self.vocab.get_name(node_id).expect("one mapping per node");
            return Err(Error::InvalidArgument(format!(
                "VPCG left node '{}:{}' without features, so it can't be embedded", node_type, name)))
        }
        // Each feature hashes on its own, so the table only needs the features that are used.
        let num_features = self.maps.iter().flatten().map(|(f, _)| f + 1).max().unwrap_or(0);
        let es = VPCG::convert_to_es(num_features, dims, self.maps)?;
        NodeEmbeddings::new(self.vocab, es)
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

    /// Users u1, u2 and items i1..i3: u1 - i1, u1 - i2, u2 - i2, u2 - i3.  Only items have
    /// features.
    fn user_items(item_type: &str) -> (NamedGraph, FeatureSet) {
        let mut gb = crate::named_graph::GraphBuilder::new();
        for (u, i) in [("u1", "i1"), ("u1", "i2"), ("u2", "i2"), ("u2", "i3")] {
            gb.add_edge(("user", u), (item_type, i), 1., crate::graph::EdgeType::Undirected);
        }
        let g = gb.build(true).unwrap();
        let mut fs = FeatureSet::from_graph(&g);
        for i in ["i1", "i2", "i3"] {
            fs.set_features(item_type, i, [("w", i), ("w", "shared")].into_iter()).unwrap();
        }
        (g, fs)
    }

    fn sorted(map: &[(usize, f32)]) -> Vec<(usize, f32)> {
        let mut map = map.to_vec();
        map.sort_by_key(|(f, _)| *f);
        map
    }

    #[test]
    fn test_new_uses_binding_defaults() {
        let vpcg = VPCG::new(7, 3);
        assert_eq!((vpcg.max_terms, vpcg.iterations, vpcg.alpha, vpcg.err), (7, 3, 1.0, 1e-5));
        assert_eq!(vpcg.feature_weight, FeatureWeight::Uniform);
    }

    #[test]
    fn test_learn_named_matches_learn() {
        let (g, mut fs) = user_items("item");
        let vpcg = VPCG { err: 0., ..VPCG::new(10, 2) };
        let users = HashSet::from(["user".to_string()]);
        let mappings = vpcg.learn_named(&g, &mut fs, &users).unwrap();
        assert!(Arc::ptr_eq(mappings.vocab(), g.vocab()));
        assert_eq!(mappings.len(), g.nodes());

        // Users got a feature of their own first.
        assert_eq!(fs.num_features(), 4 + 2);
        let (left, right) = g.split_by_node_types(&users);
        let expected = vpcg.learn(g.graph(), fs.store(), (&left, &right));
        for node_id in 0..g.nodes() {
            assert_eq!(sorted(mappings.get_by_id(node_id).unwrap()), sorted(&expected[node_id]));
        }

        // u1 takes on its items' features: i1, i2 and the shared one.
        let u1: Vec<_> = sorted(mappings.get("user", "u1").unwrap()).into_iter().map(|(f, _)| f).collect();
        let names: Vec<_> = u1.iter().map(|f| fs.store().get_pretty_feature(*f).1).collect();
        assert_eq!(names.len(), 3);
        for name in ["i1", "i2", "shared"] {
            assert!(names.iter().any(|n| n == name), "{:?}", names);
        }
        let norm: f32 = mappings.get("user", "u1").unwrap().iter().map(|(_, w)| w * w).sum();
        assert!((norm - 1.).abs() < 1e-5);

        assert!(matches!(mappings.get("user", "zzz"), Err(Error::UnknownNode { .. })));
        assert!(matches!(mappings.get_by_id(5), Err(Error::NodeIdOutOfRange { node_id: 5, len: 5 })));
    }

    /// Features over another graph's nodes are rejected before they're changed.
    #[test]
    fn test_learn_named_rejects_misaligned_features() {
        let (g, _) = user_items("item");
        let (_, mut other) = user_items("product");
        let before = other.num_features();
        let users = HashSet::from(["user".to_string()]);
        assert!(matches!(VPCG::new(10, 2).learn_named(&g, &mut other, &users), Err(Error::VocabMismatch(_))));
        assert_eq!(other.num_features(), before);
    }

    /// The hash table sized by the features used embeds the same as one sized by all of them.
    #[test]
    fn test_into_embeddings_matches_convert_to_es() {
        let (g, mut fs) = user_items("item");
        let users = HashSet::from(["user".to_string()]);
        let mappings = VPCG::new(10, 3).learn_named(&g, &mut fs, &users).unwrap();
        let expected = VPCG::convert_to_es(fs.num_features() + 10, 16, mappings.maps().to_vec()).unwrap();
        let ne = mappings.into_embeddings(16).unwrap();
        assert!(Arc::ptr_eq(ne.vocab(), g.vocab()));
        assert_eq!((ne.len(), ne.dims()), (5, 16));
        assert!(matches!(ne.distance(), Distance::Cosine));
        for node_id in 0..ne.len() {
            assert_eq!(ne.get_embedding_by_id(node_id).unwrap(), expected.get_embedding(node_id));
        }
    }

    #[test]
    fn test_featureless_nodes_and_zero_dims_are_errors() {
        let (g, mut fs) = user_items("item");
        let users = HashSet::from(["user".to_string()]);
        // Keeping no terms leaves every node without features.
        let empty = VPCG::new(0, 1).learn_named(&g, &mut fs, &users).unwrap();
        assert!(matches!(empty.clone().into_embeddings(8), Err(Error::InvalidArgument(_))));
        assert!(matches!(VPCG::convert_to_es(6, 8, empty.maps().to_vec()), Err(Error::InvalidArgument(_))));

        let mappings = VPCG::new(10, 1).learn_named(&g, &mut fs, &users).unwrap();
        assert!(matches!(mappings.clone().into_embeddings(0), Err(Error::InvalidArgument(_))));
        // Feature ids must be in range.
        assert!(matches!(VPCG::convert_to_es(1, 8, mappings.maps().to_vec()), Err(Error::InvalidArgument(_))));
        assert!(matches!(VpcgFeatureMappings::new(g.vocab().clone(), vec![]), Err(Error::InvalidArgument(_))));
    }
}
