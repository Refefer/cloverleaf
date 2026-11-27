//! FastRP: node embeddings from very sparse random projections, smoothed over the graph
//! (Chen et al., "Fast and Accurate Network Embeddings via Very Sparse Random Projection",
//! CIKM 2019).  Nothing is trained: each node starts from a random vector, and its embedding is a
//! weighted sum of that vector's propagations through the graph,
//!
//! ```text
//!     N_0 = deg^β · R            (a very sparse random vector per node)
//!     N_i = P · N_{i-1}          (P the row-normalised adjacency: an average over neighbours)
//!     E   = Σ_i weights[i] · N_i / |N_i|
//! ```
//!
//! With a [`FeatureSet`], part of each node's starting vector is the mean of random vectors for
//! its features instead, so nodes that share features start close, as in Neo4j's extension with
//! node properties.  Random vectors are derived from hashes of the node's and features' names, so
//! they don't depend on node order, and the same seed gives the same embeddings on any build.
use std::fmt::Write;

use rayon::prelude::*;

use crate::distance::Distance;
use crate::embeddings::{EmbeddingStore, NodeEmbeddings};
use crate::error::{ensure_aligned, Error};
use crate::feature_store::{FeatureSet, FeatureStore};
use crate::graph::{CDFtoP, Graph, NodeID};
use crate::named_graph::NamedGraph;
use crate::progress::CLProgressBar;
use crate::vocab::Vocab;

/// FastRP's settings.  [`FastRP::new`] fills in the defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct FastRP {
    /// Dimensions of the embeddings.
    pub dims: usize,

    /// `weights[0]` weighs each node's own starting vector and `weights[i]` its propagation `i`
    /// hops out, so there are `weights.len() - 1` propagations.  Each term is scaled to unit
    /// length before it's weighed.
    pub weights: Vec<f32>,

    /// β: each starting vector is scaled by its node's degree to the power β.  Negative values
    /// quiet high-degree nodes in their neighbours' embeddings, positive ones amplify them; 0
    /// leaves every node the same.
    pub normalization_strength: f32,

    /// s ≥ 1: each entry of a random vector is +1 or -1 with probability 1 / (2s) each, else 0
    /// (Achlioptas' and Li's very sparse projections).  The paper uses 3; 1 is a dense ±1
    /// projection.
    pub sparsity: f32,

    /// Seed for the random vectors.
    pub seed: u64,

    /// Shows a progress bar.
    pub indicator: bool
}

impl FastRP {
    /// Neo4j's default weights: three propagations, weighing the second and third hop.
    pub const DEFAULT_WEIGHTS: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

    /// The paper's sparsity.
    pub const DEFAULT_SPARSITY: f32 = 3.0;

    pub const DEFAULT_SEED: u64 = 20191103;

    /// FastRP with `dims` dimensions, [`DEFAULT_WEIGHTS`](Self::DEFAULT_WEIGHTS), no degree
    /// normalisation, [`DEFAULT_SPARSITY`](Self::DEFAULT_SPARSITY),
    /// [`DEFAULT_SEED`](Self::DEFAULT_SEED) and a progress bar.
    pub fn new(dims: usize) -> Self {
        FastRP {
            dims,
            weights: FastRP::DEFAULT_WEIGHTS.to_vec(),
            normalization_strength: 0.0,
            sparsity: FastRP::DEFAULT_SPARSITY,
            seed: FastRP::DEFAULT_SEED,
            indicator: true
        }
    }

    /// [`Error::InvalidArgument`] unless `dims` is at least 1, `weights` is non-empty, finite
    /// and not all 0, `normalization_strength` is finite and `sparsity` is finite and at least 1.
    pub fn validate(&self) -> Result<(), Error> {
        let invalid = |m: &str| Err(Error::InvalidArgument(m.to_string()));
        if self.dims == 0 {
            return invalid("dims must be at least 1")
        }
        if self.weights.is_empty() {
            return invalid("weights must have at least one entry")
        }
        if self.weights.iter().any(|w| !w.is_finite()) {
            return invalid("weights must be finite")
        }
        if self.weights.iter().all(|w| *w == 0.0) {
            return invalid("weights can't all be 0: every embedding would be zeros")
        }
        if !self.normalization_strength.is_finite() {
            return invalid("normalization_strength must be finite")
        }
        if !(self.sparsity.is_finite() && self.sparsity >= 1.0) {
            return invalid("sparsity must be finite and at least 1")
        }
        Ok(())
    }

    /// Embeds every node of `graph` from random vectors for the nodes alone (structure only).
    pub fn learn(&self, graph: &NamedGraph) -> Result<NodeEmbeddings, Error> {
        self.validate()?;
        let init = Initial::new(graph.vocab(), None, 0);
        self.into_embeddings(graph, self.compute(graph.graph(), &init))
    }

    /// Embeds every node of `graph`, starting each from a random vector for the node in its first
    /// `dims - feature_dims` dimensions and the mean of random vectors for its `features` in the
    /// last `feature_dims`; with `feature_dims == dims` only the features count.  The two blocks
    /// are each scaled to unit length, so they weigh the same whatever their sizes, and a node
    /// without features starts with zeros in the feature block.  `features` must index the
    /// graph's nodes, else an [`Error::VocabMismatch`]; `feature_dims` must be from 1 to `dims`.
    pub fn learn_with_features(
        &self,
        graph: &NamedGraph,
        features: &FeatureSet,
        feature_dims: usize
    ) -> Result<NodeEmbeddings, Error> {
        self.validate()?;
        ensure_aligned(graph.vocab(), features.vocab(), "Graph and FeatureSet")?;
        if feature_dims == 0 || feature_dims > self.dims {
            return Err(Error::InvalidArgument(format!(
                "feature_dims must be from 1 to dims ({}), got {}", self.dims, feature_dims)))
        }
        let init = Initial::new(graph.vocab(), Some(features.store()), feature_dims);
        self.into_embeddings(graph, self.compute(graph.graph(), &init))
    }

    fn into_embeddings(&self, graph: &NamedGraph, embeddings: Vec<f32>) -> Result<NodeEmbeddings, Error> {
        let es = EmbeddingStore::new_with_vec(graph.nodes(), self.dims, Distance::Cosine, embeddings)
            .expect("one embedding per node");
        NodeEmbeddings::new(graph.vocab().clone(), es)
    }

    /// The embeddings, `dims` floats per node in node id order.  Holds three such buffers at
    /// most (the current propagation, the next and the running sum), two when only the last
    /// weight is non-zero.
    fn compute<G: Graph + Sync>(&self, graph: &G, init: &Initial) -> Vec<f32> {
        let d = self.dims;
        let n = graph.len();
        let steps = self.weights.len() - 1;
        let pb = CLProgressBar::new(((steps + 1) * n) as u64, self.indicator);
        pb.update_message(|msg| { msg.clear(); msg.push_str("Random vectors"); });

        let mut current = vec![0f32; n * d];
        current.par_chunks_mut(d).enumerate().for_each(|(node, row)| {
            init.fill(node, row, d, self.sparsity, self.seed);
            let degree = graph.degree(node);
            if self.normalization_strength != 0.0 && degree > 0 {
                let scale = (degree as f32).powf(self.normalization_strength);
                row.iter_mut().for_each(|x| *x *= scale);
            }
        });
        pb.inc(n as u64);

        // Only the last term counts: it's the result, so no running sum is needed.
        let last_only = self.weights[..steps].iter().all(|w| *w == 0.0);
        let mut sum = if last_only { Vec::new() } else { vec![0f32; n * d] };
        if !last_only {
            add_normalised(&mut sum, &current, d, self.weights[0]);
        }
        let mut next = if steps > 0 { vec![0f32; n * d] } else { Vec::new() };
        for step in 1..=steps {
            pb.update_message(|msg| {
                msg.clear();
                write!(msg, "Propagation {}/{}", step, steps).expect("writing to a String");
            });
            next.par_chunks_mut(d).enumerate().for_each(|(node, row)| {
                row.iter_mut().for_each(|x| *x = 0.0);
                let (edges, weights) = graph.get_edges(node);
                for (neighbour, p) in edges.iter().zip(CDFtoP::new(weights)) {
                    let from = &current[neighbour * d..(neighbour + 1) * d];
                    row.iter_mut().zip(from).for_each(|(x, y)| *x += p * y);
                }
            });
            std::mem::swap(&mut current, &mut next);
            if !last_only {
                add_normalised(&mut sum, &current, d, self.weights[step]);
            }
            pb.inc(n as u64);
        }
        pb.finish();

        if last_only {
            // `current` becomes the result in place.
            let w = self.weights[steps];
            current.par_chunks_mut(d).for_each(|row| {
                let norm = l2(row);
                let scale = if norm > 0.0 { w / norm } else { 0.0 };
                row.iter_mut().for_each(|x| *x *= scale);
            });
            current
        } else {
            sum
        }
    }
}

/// `sum += weight * row / |row|` for each node's row; zero rows add nothing.
fn add_normalised(sum: &mut [f32], term: &[f32], d: usize, weight: f32) {
    if weight == 0.0 {
        return
    }
    sum.par_chunks_mut(d).zip(term.par_chunks(d)).for_each(|(s, t)| {
        let norm = l2(t);
        if norm > 0.0 {
            let scale = weight / norm;
            s.iter_mut().zip(t).for_each(|(s, t)| *s += scale * t);
        }
    });
}

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// What each node's starting vector is made of: hashes of the node names and, for the feature
/// block, of the feature names.
struct Initial<'a> {
    node_keys: Vec<u64>,
    features: Option<(&'a FeatureStore, Vec<u64>)>,
    feature_dims: usize
}

impl<'a> Initial<'a> {
    fn new(vocab: &Vocab, features: Option<&'a FeatureStore>, feature_dims: usize) -> Self {
        let node_keys = (0..vocab.len()).into_par_iter()
            .map(|node| {
                let (node_type, name) = vocab.get_name(node).expect("node ids index the vocab");
                name_key(&node_type, name)
            })
            .collect();
        let features = features.map(|fs| {
            let fv = fs.get_vocab();
            let keys = (0..fv.len()).into_par_iter()
                .map(|f| {
                    let (ft, name) = fv.get_name(f).expect("feature ids index the feature vocab");
                    // Features and nodes with the same name get different vectors.
                    mix(name_key(&ft, name) ^ FEATURE_SALT)
                })
                .collect();
            (fs, keys)
        });
        Initial { node_keys, features, feature_dims }
    }

    /// Writes `node`'s starting vector into `row` (zeros on entry): the id block, then the
    /// feature block, each of unit length (or zeros).
    fn fill(&self, node: NodeID, row: &mut [f32], d: usize, sparsity: f32, seed: u64) {
        let (id_block, feature_block) = row.split_at_mut(d - self.feature_dims);
        if !id_block.is_empty() {
            add_random_vector(id_block, self.node_keys[node], seed, sparsity);
            normalise(id_block);
        }
        if let Some((fs, keys)) = &self.features {
            for f in fs.get_features(node) {
                add_random_vector(feature_block, keys[*f], seed, sparsity);
            }
            normalise(feature_block);
        }
    }
}

fn normalise(v: &mut [f32]) {
    let norm = l2(v);
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

const FEATURE_SALT: u64 = 0x5fa7_0f3a_7c1e_9d2b;

/// Adds the random vector for `key`: each entry is +1 or -1 with probability 1 / (2 sparsity)
/// each, else 0, from a hash of the key, the seed and the entry's position.
fn add_random_vector(out: &mut [f32], key: u64, seed: u64, sparsity: f32) {
    // h < nonzero (probability 1 / s) gives a non-zero entry, its lowest bit the sign.
    let nonzero = if sparsity <= 1.0 { u64::MAX } else { (u64::MAX as f64 / sparsity as f64) as u64 };
    let base = mix(key ^ mix(seed));
    for (dim, x) in out.iter_mut().enumerate() {
        let h = mix(base.wrapping_add((dim as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)));
        if h < nonzero || nonzero == u64::MAX {
            *x += if h & 1 == 1 { 1.0 } else { -1.0 };
        }
    }
}

/// FNV-1a over the node (or feature) type and name: stable across builds and platforms, unlike
/// the std and ahash hashers.
fn name_key(node_type: &str, name: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in node_type.bytes().chain(std::iter::once(0xff)).chain(name.bytes()) {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    mix(h)
}

/// SplitMix64's finaliser.
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod fastrp_tests {
    use super::*;
    use crate::graph::EdgeType;
    use crate::named_graph::GraphBuilder;

    fn quiet(dims: usize) -> FastRP {
        FastRP { indicator: false, ..FastRP::new(dims) }
    }

    fn build(edges: &[(&str, &str)]) -> NamedGraph {
        let mut gb = GraphBuilder::new();
        for (a, b) in edges {
            gb.add_edge(("n", *a), ("n", *b), 1., EdgeType::Undirected);
        }
        gb.build(true).unwrap()
    }

    /// A hub `h` with leaves `a`, `b`, `c`, plus a path `c - x - y`.
    fn star() -> NamedGraph {
        build(&[("h", "a"), ("h", "b"), ("h", "c"), ("c", "x"), ("x", "y")])
    }

    fn emb<'a>(e: &'a NodeEmbeddings, name: &str) -> &'a [f32] {
        e.get_embedding("n", name).unwrap()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() / (l2(a) * l2(b))
    }

    #[test]
    fn test_new_uses_defaults() {
        let f = FastRP::new(32);
        assert_eq!(f.weights, vec![0.0, 0.0, 1.0, 1.0]);
        assert_eq!((f.normalization_strength, f.sparsity, f.seed), (0.0, 3.0, FastRP::DEFAULT_SEED));
        assert!(f.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_bad_settings() {
        let bad = [
            FastRP { dims: 0, ..quiet(8) },
            FastRP { weights: vec![], ..quiet(8) },
            FastRP { weights: vec![0.0, 0.0], ..quiet(8) },
            FastRP { weights: vec![1.0, f32::NAN], ..quiet(8) },
            FastRP { normalization_strength: f32::INFINITY, ..quiet(8) },
            FastRP { sparsity: 0.5, ..quiet(8) },
            FastRP { sparsity: f32::NAN, ..quiet(8) },
        ];
        for f in bad {
            assert!(matches!(f.learn(&star()), Err(Error::InvalidArgument(_))), "{:?}", f);
        }
    }

    #[test]
    fn test_same_seed_same_embeddings_and_independent_of_node_order() {
        let a = quiet(64).learn(&star()).unwrap();
        let b = quiet(64).learn(&star()).unwrap();
        // The same edges added in another order give the nodes other ids.
        let reordered = build(&[("x", "y"), ("c", "x"), ("h", "c"), ("h", "b"), ("h", "a")]);
        let c = quiet(64).learn(&reordered).unwrap();
        assert_ne!(a.node_id("n", "y").unwrap(), c.node_id("n", "y").unwrap());
        for name in ["h", "a", "b", "c", "x", "y"] {
            assert_eq!(emb(&a, name), emb(&b, name));
            for (x, y) in emb(&a, name).iter().zip(emb(&c, name)) {
                assert!((x - y).abs() < 1e-5, "{}", name);
            }
        }
        let other = FastRP { seed: 7, ..quiet(64) }.learn(&star()).unwrap();
        assert_ne!(emb(&a, "h"), emb(&other, "h"));
    }

    #[test]
    fn test_structurally_equivalent_nodes_match() {
        // Leaves a and b only touch the hub: without their own vectors they're identical.
        let e = FastRP { weights: vec![0.0, 1.0, 1.0], ..quiet(64) }.learn(&star()).unwrap();
        assert_eq!(emb(&e, "a"), emb(&e, "b"));
        // With their own vectors they differ.
        let e = FastRP { weights: vec![1.0, 1.0], ..quiet(64) }.learn(&star()).unwrap();
        assert_ne!(emb(&e, "a"), emb(&e, "b"));
    }

    #[test]
    fn test_one_hop_is_the_normalised_neighbour_mean() {
        let g = star();
        let own = FastRP { weights: vec![1.0], ..quiet(48) }.learn(&g).unwrap();
        let hop = FastRP { weights: vec![0.0, 1.0], ..quiet(48) }.learn(&g).unwrap();
        // c's neighbours are h and x, with equal weight.
        let mean: Vec<f32> = emb(&own, "h").iter().zip(emb(&own, "x")).map(|(a, b)| (a + b) / 2.).collect();
        let norm = l2(&mean);
        for (x, y) in emb(&hop, "c").iter().zip(&mean) {
            assert!((x - y / norm).abs() < 1e-5);
        }
        // Each term has unit length, so these do too.
        assert!((l2(emb(&own, "a")) - 1.0).abs() < 1e-5);
        assert!((l2(emb(&hop, "c")) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_weights_scale_their_terms() {
        let g = star();
        let both = FastRP { weights: vec![2.0, 0.5], ..quiet(32) }.learn(&g).unwrap();
        let own = FastRP { weights: vec![1.0], ..quiet(32) }.learn(&g).unwrap();
        let hop = FastRP { weights: vec![0.0, 1.0], ..quiet(32) }.learn(&g).unwrap();
        for name in ["h", "c", "y"] {
            for ((x, a), b) in emb(&both, name).iter().zip(emb(&own, name)).zip(emb(&hop, name)) {
                assert!((x - (2.0 * a + 0.5 * b)).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_random_vectors_are_sparse_and_nearly_orthogonal() {
        let g = star();
        let own = FastRP { weights: vec![1.0], ..quiet(1024) }.learn(&g).unwrap();
        // About a third of the entries are non-zero, all of the same magnitude.
        let nonzero: Vec<f32> = emb(&own, "a").iter().copied().filter(|x| *x != 0.0).collect();
        assert!(nonzero.len() > 280 && nonzero.len() < 400, "{}", nonzero.len());
        assert!(nonzero.iter().all(|x| (x.abs() - nonzero[0].abs()).abs() < 1e-6));
        let positive = nonzero.iter().filter(|x| **x > 0.0).count();
        assert!(positive > nonzero.len() / 3 && positive < 2 * nonzero.len() / 3);
        assert!(cosine(emb(&own, "a"), emb(&own, "b")).abs() < 0.2);
        let dense = FastRP { weights: vec![1.0], sparsity: 1.0, ..quiet(64) }.learn(&g).unwrap();
        assert!(emb(&dense, "a").iter().all(|x| *x != 0.0));
    }

    #[test]
    fn test_normalization_strength_quiets_hubs() {
        // y's only neighbour is x, whose neighbours are c and y: no effect on a one-hop term
        // from a single neighbour, but c (degree 2) weighs x's term against y (degree 1).
        let g = star();
        let own = FastRP { weights: vec![1.0], ..quiet(256) }.learn(&g).unwrap();
        let plain = FastRP { weights: vec![0.0, 1.0], ..quiet(256) }.learn(&g).unwrap();
        let quiet_hubs = FastRP { weights: vec![0.0, 1.0], normalization_strength: -1.0, ..quiet(256) }
            .learn(&g).unwrap();
        let towards = |e: &NodeEmbeddings, name| cosine(emb(e, "x"), emb(&own, name));
        assert!((towards(&plain, "c") - towards(&plain, "y")).abs() < 0.1);
        assert!(towards(&quiet_hubs, "y") > towards(&quiet_hubs, "c") + 0.2);
    }

    #[test]
    fn test_features_bring_nodes_together() {
        let g = build(&[("p", "q"), ("r", "s")]);
        let mut fs = FeatureSet::from_graph(&g);
        for (node, words) in [("p", ["red", "big"]), ("r", ["red", "big"]), ("q", ["blue", "small"])] {
            fs.set_features("n", node, words.iter().map(|w| ("w", *w))).unwrap();
        }
        let f = FastRP { weights: vec![1.0], ..quiet(512) };
        let only = f.learn_with_features(&g, &fs, 512).unwrap();
        assert!((cosine(emb(&only, "p"), emb(&only, "r")) - 1.0).abs() < 1e-5);
        assert!(cosine(emb(&only, "p"), emb(&only, "q")).abs() < 0.2);
        // s has no features: zeros.
        assert!(emb(&only, "s").iter().all(|x| *x == 0.0));

        // Half and half: the id blocks differ and the feature blocks match.
        let half = f.learn_with_features(&g, &fs, 256).unwrap();
        for (x, y) in emb(&half, "p")[256..].iter().zip(&emb(&half, "r")[256..]) {
            assert!((x - y).abs() < 1e-6);
        }
        assert!(cosine(&emb(&half, "p")[..256], &emb(&half, "r")[..256]).abs() < 0.3);
        // Each block has unit length, so the vector has length sqrt(2) before the scaling to 1.
        assert!((l2(&emb(&half, "p")[..256]) - 0.5f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_learn_with_features_checks_inputs() {
        let g = star();
        let fs = FeatureSet::from_graph(&g);
        assert!(matches!(quiet(8).learn_with_features(&g, &fs, 0), Err(Error::InvalidArgument(_))));
        assert!(matches!(quiet(8).learn_with_features(&g, &fs, 9), Err(Error::InvalidArgument(_))));
        let other = FeatureSet::from_graph(&build(&[("a", "b")]));
        assert!(matches!(quiet(8).learn_with_features(&g, &other, 4), Err(Error::VocabMismatch(_))));
    }

    #[test]
    fn test_sinks_in_directed_graphs_keep_their_own_term() {
        let mut gb = GraphBuilder::new();
        gb.add_edge(("n", "a"), ("n", "b"), 1., EdgeType::Directed);
        let g = gb.build(true).unwrap();
        let e = FastRP { weights: vec![1.0, 1.0], ..quiet(16) }.learn(&g).unwrap();
        // b has no out-edges: its propagation is zeros and adds nothing, and nothing is NaN.
        assert!(emb(&e, "b").iter().all(|x| x.is_finite()));
        assert!((l2(emb(&e, "b")) - 1.0).abs() < 1e-5);
        let hop_only = FastRP { weights: vec![0.0, 1.0], ..quiet(16) }.learn(&g).unwrap();
        assert!(emb(&hop_only, "b").iter().all(|x| *x == 0.0));
    }

    const PINNED_KEY: u64 = 1944415843554842546;
    const PINNED_VECTOR: [f32; 16] =
        [0., 0., 0., -1., 0., 0., 1., 0., 1., -1., 0., 1., -1., 0., 0., 0.];

    #[test]
    fn test_name_key_is_stable() {
        // Pinned so a change to the hashing, which changes every embedding, is deliberate.
        assert_eq!(name_key("n", "a"), name_key("n", "a"));
        assert_ne!(name_key("n", "a"), name_key("na", ""));
        assert_eq!(mix(0), 0);
        assert_eq!(name_key("n", "a"), PINNED_KEY);
        let mut v = [0f32; 16];
        add_random_vector(&mut v, PINNED_KEY, FastRP::DEFAULT_SEED, 3.0);
        assert_eq!(v, PINNED_VECTOR);
    }
}
