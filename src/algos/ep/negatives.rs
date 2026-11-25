//! Negative sampling options for Embedding Propagation beyond uniform random negatives:
//! restricting each anchor's negatives by node type ([`NegativeTypes`]) and hard negatives
//! ([`HardNegatives`]), either from random walks or the nearest nodes in a per-batch pool.
//!
//! These match the GPU trainer's `negative_types`, `hard_negatives` and `negative_pool` options.
use std::fmt;
use std::str::FromStr;

use rand::prelude::*;
use rand_distr::{Distribution, Uniform};
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;

use crate::embeddings::EmbeddingStore;
use crate::error::Error;
use crate::feature_store::FeatureStore;
use crate::geometry::Geometry;
use crate::graph::{Graph as CGraph, NodeID};
use crate::vocab::Vocab;
use crate::algos::grad_utils::node_sampler::{NegativeSource, NodeSampler, RandomWalkHardSampler};
use super::model::Model;

/// Which node types each anchor type's negatives may be, as `"query=product;product=query"`:
/// `;` separates anchor types, `,` the types each may draw.  Anchor types left out may draw any
/// type.  Within an anchor's allowed types, each type is equally likely, however rare, then a
/// node of that type uniformly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NegativeTypes(Vec<(String, Vec<String>)>);

impl NegativeTypes {
    /// The anchor types and, for each, the node types its negatives may be.
    pub fn mapping(&self) -> &[(String, Vec<String>)] {
        &self.0
    }
}

impl FromStr for NegativeTypes {
    type Err = Error;

    fn from_str(spec: &str) -> Result<Self, Error> {
        let mut mapping: Vec<(String, Vec<String>)> = Vec::new();
        for part in spec.split(';').map(str::trim).filter(|p| !p.is_empty()) {
            let bad = || Error::InvalidArgument(format!(
                "negative_types: expected ANCHOR=TYPE[,TYPE...] separated by ';', got {:?}", part));
            let (anchor, allowed) = part.split_once('=').ok_or_else(bad)?;
            let anchor = anchor.trim();
            let types: Vec<String> = allowed.split(',').map(str::trim).filter(|t| !t.is_empty())
                .map(String::from).collect();
            if anchor.is_empty() || types.is_empty() { return Err(bad()) }
            if mapping.iter().any(|(a, _)| a == anchor) {
                return Err(Error::InvalidArgument(format!("negative_types: anchor type {:?} given twice", anchor)))
            }
            mapping.push((anchor.to_string(), types));
        }
        if mapping.is_empty() {
            return Err(Error::InvalidArgument("negative_types is empty".into()))
        }
        Ok(NegativeTypes(mapping))
    }
}

impl fmt::Display for NegativeTypes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.0.iter().map(|(a, t)| format!("{}={}", a, t.join(","))).collect();
        write!(f, "{}", parts.join(";"))
    }
}

/// Nearest-node hard negatives: each batch draws `pool` nodes (by type, with
/// [`NegativeTypes`]), embeds them, and each anchor adds the `k` pool nodes nearest its
/// reconstruction, other than itself, its own neighbours (pushing those away would fight the
/// positive term) and types it may not draw.  They are used alongside the loss's random
/// negatives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NearestNegatives {
    k: usize,
    pool: usize
}

impl NearestNegatives {
    /// `k` nearest of a `pool`-node pool per batch; needs 1 <= k <= pool.
    pub fn new(k: usize, pool: usize) -> Result<Self, Error> {
        if k == 0 || pool < k {
            return Err(Error::InvalidArgument(format!(
                "nearest hard negatives need 1 <= k <= pool, got k {} and pool {}", k, pool)))
        }
        Ok(NearestNegatives { k, pool })
    }

    pub fn k(&self) -> usize { self.k }

    pub fn pool(&self) -> usize { self.pool }
}

/// Hard negatives used alongside the loss's random negatives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HardNegatives {
    /// Random negatives only.
    #[default]
    None,

    /// Up to this many of the random negatives replaced by nodes found by short random walks
    /// from the anchor that aren't its neighbours.
    RandomWalk(usize),

    /// The nearest nodes of a per-batch pool, added to the random negatives.
    Nearest(NearestNegatives)
}

impl HardNegatives {
    /// Negatives the loss scores per anchor beyond the loss's own random ones.
    pub fn extra(&self) -> usize {
        match self {
            HardNegatives::Nearest(n) => n.k,
            _ => 0
        }
    }
}

/// [`NegativeTypes`] resolved against a graph's node types, with the nodes of each type
/// negatives may be.
pub(crate) struct TypedNodes<'a> {
    vocab: &'a Vocab,
    /// Per anchor node type: the types its negatives may be (None: any).
    allowed: Vec<Option<Vec<usize>>>,
    /// Per node type: its nodes, when some anchor may draw it.
    members: Vec<Vec<NodeID>>,
    /// The types some anchor may draw.
    used: Vec<usize>
}

impl <'a> TypedNodes<'a> {
    /// Negatives may be any node of an allowed type, validation nodes included: a node that is
    /// never a negative is only ever pulled towards its neighbours.
    pub(crate) fn new(spec: &NegativeTypes, vocab: &'a Vocab) -> Result<Self, Error> {
        let names = vocab.node_types();
        let index = |t: &str| vocab.node_type_index(t).ok_or_else(|| Error::InvalidArgument(format!(
            "negative_types names node type {:?}, which isn't in the graph (it has {})", t, names.join(", "))));
        let mut allowed = vec![None; names.len()];
        for (anchor, types) in spec.mapping() {
            let mut ids = types.iter().map(|t| index(t)).collect::<Result<Vec<_>, _>>()?;
            ids.sort_unstable();
            ids.dedup();
            allowed[index(anchor)?] = Some(ids);
        }
        let mut used = vec![false; names.len()];
        for a in allowed.iter() {
            match a {
                Some(ids) => ids.iter().for_each(|t| used[*t] = true),
                None => used.iter_mut().for_each(|u| *u = true)
            }
        }
        let mut members = vec![Vec::new(); names.len()];
        for node in 0..vocab.len() {
            let t = vocab.node_type_id(node).expect("nodes are in the vocab");
            if used[t] { members[t].push(node) }
        }
        // Types without nodes can't be drawn: an anchor left with none can't train.
        for (anchor, a) in allowed.iter_mut().enumerate() {
            if let Some(ids) = a {
                ids.retain(|t| !members[*t].is_empty());
                if ids.is_empty() {
                    return Err(Error::InvalidArgument(format!(
                        "negative_types: anchor type {:?} has no nodes of any type it may draw",
                        names[anchor])))
                }
            }
        }
        let used = (0..names.len()).filter(|t| used[*t] && !members[*t].is_empty()).collect();
        Ok(TypedNodes { vocab, allowed, members, used })
    }

    fn type_of(&self, node: NodeID) -> usize {
        self.vocab.node_type_id(node).expect("graph nodes are in the vocab")
    }

    fn may_draw(&self, anchor_type: usize, node_type: usize) -> bool {
        match &self.allowed[anchor_type] {
            Some(ids) => ids.binary_search(&node_type).is_ok(),
            None => true
        }
    }

    /// One node of a type `anchor` may draw, other than `anchor` when anything else exists.
    fn draw<R: Rng>(&self, anchor: NodeID, anchor_type: usize, rng: &mut R) -> NodeID {
        let candidates = |rng: &mut R| -> NegativeSource {
            match &self.allowed[anchor_type] {
                Some(ids) => NegativeSource::Nodes(&self.members[ids[Uniform::new(0, ids.len()).sample(rng)]]),
                None => NegativeSource::All(self.vocab.len())
            }
        };
        for _ in 0..64 {
            let node = candidates(rng).sample(rng);
            if node != anchor { return node }
        }
        candidates(rng).sample(rng)
    }
}

/// Random negatives restricted by node type.
pub(crate) struct TypedSampler<'a, 'b> {
    types: &'b TypedNodes<'a>
}

impl <'a, 'b> NodeSampler for TypedSampler<'a, 'b> {
    fn sample_negatives<R: Rng>(
        &self,
        _graph: &impl CGraph,
        anchor: NodeID,
        negatives: &mut Vec<NodeID>,
        num_negs: usize,
        rng: &mut R
    ) {
        let t = self.types.type_of(anchor);
        while negatives.len() < num_negs {
            negatives.push(self.types.draw(anchor, t, rng));
        }
    }
}

/// The random-negative sampler EP uses: uniform over every node (with optional random-walk hard
/// negatives) or restricted by node type.
pub(crate) enum RandomNegatives<'a, 'b> {
    Uniform(RandomWalkHardSampler<'static>),
    Typed(TypedSampler<'a, 'b>)
}

impl <'a, 'b> RandomNegatives<'a, 'b> {
    pub(crate) fn new(hard: &HardNegatives, types: Option<&'b TypedNodes<'a>>, num_nodes: usize) -> Self {
        match types {
            Some(types) => RandomNegatives::Typed(TypedSampler { types }),
            None => {
                let walks = if let HardNegatives::RandomWalk(n) = hard { *n } else { 0 };
                RandomNegatives::Uniform(RandomWalkHardSampler::new(walks, NegativeSource::All(num_nodes)))
            }
        }
    }
}

impl <'a, 'b> NodeSampler for RandomNegatives<'a, 'b> {
    fn sample_negatives<R: Rng>(
        &self,
        graph: &impl CGraph,
        anchor: NodeID,
        negatives: &mut Vec<NodeID>,
        num_negs: usize,
        rng: &mut R
    ) {
        match self {
            RandomNegatives::Uniform(s) => s.sample_negatives(graph, anchor, negatives, num_negs, rng),
            RandomNegatives::Typed(s) => s.sample_negatives(graph, anchor, negatives, num_negs, rng)
        }
    }
}

/// A batch's pool of embedded candidate nodes for nearest hard negatives.
pub(crate) struct HardPool<'a, 'b> {
    k: usize,
    nodes: Vec<NodeID>,
    /// The pool's embeddings in the model's space, row-major.
    embeddings: Vec<f32>,
    dims: usize,
    types: Option<&'b TypedNodes<'a>>
}

impl <'a, 'b> HardPool<'a, 'b> {
    /// Draws the pool (by type in equal shares with `types`, else uniformly from the graph's
    /// `num_nodes` nodes) and embeds it with the current feature embeddings, in `geometry`'s
    /// space.
    pub(crate) fn build<M: Model>(
        nearest: &NearestNegatives,
        types: Option<&'b TypedNodes<'a>>,
        num_nodes: usize,
        features: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        model: &M,
        geometry: &Geometry,
        seed: u64
    ) -> Self {
        let mut rng = XorShiftRng::seed_from_u64(seed);
        let nodes: Vec<NodeID> = match types {
            Some(t) => {
                let share = (nearest.pool + t.used.len() - 1) / t.used.len();
                t.used.iter().flat_map(|ty| {
                    let m = &t.members[*ty];
                    let dist = Uniform::new(0, m.len());
                    (0..share).map(|_| m[dist.sample(&mut rng)]).collect::<Vec<_>>()
                }).take(nearest.pool).collect()
            },
            None => {
                let dist = Uniform::new(0, num_nodes);
                (0..nearest.pool).map(|_| dist.sample(&mut rng)).collect()
            }
        };
        let rows: Vec<Vec<f32>> = nodes.par_iter().enumerate().map(|(i, node)| {
            let mut rng = XorShiftRng::seed_from_u64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let params = model.params();
            let (_, emb) = model.construct_node_embedding(&params, *node, 1f32, features, feature_embeddings, &mut rng);
            let mut v = emb.value().to_vec();
            geometry.to_space(&mut v);
            v
        }).collect();
        let dims = rows.first().map(|r| r.len()).unwrap_or(0);
        HardPool { k: nearest.k, nodes, embeddings: rows.concat(), dims, types }
    }

    /// Appends the `k` pool nodes nearest `target` (an embedding in the space) to `out`: not
    /// `anchor`, its neighbours, or types it may not draw.  Fewer when the pool has fewer.
    pub(crate) fn nearest(&self, graph: &impl CGraph, anchor: NodeID, target: &[f32], geometry: &Geometry,
                          out: &mut Vec<NodeID>) {
        let mut neighbours = graph.get_edges(anchor).0.to_vec();
        neighbours.sort_unstable();
        let anchor_type = self.types.map(|t| t.type_of(anchor));
        let mut scored: Vec<(f32, NodeID)> = self.nodes.iter().enumerate().filter_map(|(i, &node)| {
            if node == anchor || neighbours.binary_search(&node).is_ok() { return None }
            if let (Some(t), Some(at)) = (self.types, anchor_type) {
                if !t.may_draw(at, t.type_of(node)) { return None }
            }
            let d = geometry.distance(target, &self.embeddings[i * self.dims..(i + 1) * self.dims]);
            if d.is_finite() { Some((d, node)) } else { None }
        }).collect();
        let k = self.k.min(scored.len());
        if k == 0 { return }
        if k < scored.len() {
            scored.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
        }
        out.extend(scored[..k].iter().map(|(_, n)| *n));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_negative_types_parse() {
        let t: NegativeTypes = "query=product; product=product,query".parse().unwrap();
        assert_eq!(t.mapping(), &[("query".to_string(), vec!["product".to_string()]),
                                  ("product".to_string(), vec!["product".to_string(), "query".to_string()])]);
        assert_eq!(t.to_string(), "query=product;product=product,query");
        assert_eq!(t.to_string().parse::<NegativeTypes>().unwrap(), t);
        for bad in ["", ";", "query", "query=", "=product", "query=product;query=query"] {
            assert!(bad.parse::<NegativeTypes>().is_err(), "{:?} should be rejected", bad);
        }
    }

    #[test]
    fn test_nearest_negatives_bounds() {
        assert!(NearestNegatives::new(0, 10).is_err());
        assert!(NearestNegatives::new(11, 10).is_err());
        let n = NearestNegatives::new(3, 10).unwrap();
        assert_eq!((n.k(), n.pool()), (3, 10));
        assert_eq!(HardNegatives::Nearest(n).extra(), 3);
        assert_eq!(HardNegatives::RandomWalk(3).extra(), 0);
    }

    fn typed_vocab() -> Vocab {
        let mut v = Vocab::new();
        for i in 0..6 { v.get_or_insert("query", format!("q{}", i)); }
        for i in 0..4 { v.get_or_insert("product", format!("p{}", i)); }
        v.get_or_insert("brand", "b0");
        v
    }

    #[test]
    fn test_typed_nodes_draw_only_allowed_types() {
        let vocab = typed_vocab();
        let spec: NegativeTypes = "query=product;product=query".parse().unwrap();
        let types = TypedNodes::new(&spec, &vocab).unwrap();
        let sampler = RandomNegatives::new(&HardNegatives::None, Some(&types), vocab.len());
        let edges = (0..vocab.len() - 1).map(|i| (i, i + 1, 1.)).collect();
        let graph = crate::graph::CumCSR::convert(crate::graph::CSR::construct_from_edges(edges, false));
        let mut rng = XorShiftRng::seed_from_u64(3);
        let kind = |n: NodeID| vocab.get_node_type(n).unwrap().as_str().to_string();
        for anchor in 0..vocab.len() {
            let mut negs = Vec::new();
            sampler.sample_negatives(&graph, anchor, &mut negs, 50, &mut rng);
            assert_eq!(negs.len(), 50);
            let want = match kind(anchor).as_str() { "query" => vec!["product"], "product" => vec!["query"],
                                                    _ => vec!["query", "product", "brand"] };
            assert!(negs.iter().all(|n| *n != anchor && want.contains(&kind(*n).as_str())),
                    "anchor {} drew {:?}", anchor, negs);
        }
        // Unknown types and anchors left with nothing to draw are errors.
        assert!(TypedNodes::new(&"query=shoe".parse().unwrap(), &vocab).is_err());
    }

    /// The pool's nearest nodes, leaving out the anchor, its neighbours and types it may not draw.
    #[test]
    fn test_hard_pool_nearest_skips_anchor_neighbours_and_types() {
        let vocab = typed_vocab();
        let spec: NegativeTypes = "query=product".parse().unwrap();
        let types = TypedNodes::new(&spec, &vocab).unwrap();
        // Query 0 neighbours product 6; products 6..10 and query 1 sit at increasing distances.
        let graph = crate::graph::CumCSR::convert(crate::graph::CSR::construct_from_edges(
            vec![(0, 6, 1.), (6, 0, 1.), (10, 10, 1.)], false));
        let nodes = vec![0, 1, 6, 7, 8, 9];
        let embeddings: Vec<f32> = [0., 0.1, 0.2, 0.3, 0.4, 0.5].iter().flat_map(|x| [*x, 0.]).collect();
        let pool = HardPool { k: 2, nodes, embeddings, dims: 2, types: Some(&types) };
        let mut out = Vec::new();
        pool.nearest(&graph, 0, &[0., 0.], &Geometry::Euclidean, &mut out);
        out.sort();
        assert_eq!(out, vec![7, 8], "not itself (0), a query (1) or its neighbour (6)");
        // Without types the nearest query counts.
        let pool = HardPool { types: None, ..pool };
        let mut out = Vec::new();
        pool.nearest(&graph, 0, &[0., 0.], &Geometry::Euclidean, &mut out);
        out.sort();
        assert_eq!(out, vec![1, 7]);
    }

    /// Uniform negatives come from every node, not only the ones being trained: validation
    /// nodes that are never negatives drift to the Poincaré ball's origin, near everything.
    #[test]
    fn test_uniform_negatives_cover_every_node() {
        let graph = crate::graph::CumCSR::convert(crate::graph::CSR::construct_from_edges(
            (0..9).map(|i| (i, i + 1, 1.)).collect(), false));
        let sampler = RandomNegatives::new(&HardNegatives::None, None, 10);
        let mut rng = XorShiftRng::seed_from_u64(5);
        let mut negs = Vec::new();
        sampler.sample_negatives(&graph, 0, &mut negs, 500, &mut rng);
        let mut seen: Vec<NodeID> = negs.clone();
        seen.sort();
        seen.dedup();
        assert_eq!(seen, (1..10).collect::<Vec<_>>(), "every node but the anchor");
    }
}
