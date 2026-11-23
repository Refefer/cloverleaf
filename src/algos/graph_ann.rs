//! This module defines a mediocre way of approximating nearest neighbors through random graph
//! traversals and greedy hill climbing toward embeddings which minimize the distance.  It's fine,
//! works better than nothing, but leaves a lot to be desired and is entirely dependent on the
//! connectedness of the graph.  Meh.
use std::cmp::{Eq,PartialEq,Ordering,Reverse};
use std::collections::BinaryHeap;

use hashbrown::HashSet;
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rand_distr::{Distribution,Uniform};
use float_ord::FloatOrd;

use crate::graph::{Graph as CGraph,NodeID};
use crate::embeddings::{EmbeddingStore,Entity,NodeEmbeddings,Query};
use crate::error::{Error,ensure_aligned};
use crate::named_graph::NamedGraph;

/// Seed for [`GraphAnn::find`]'s random starting nodes when none is given.
pub const GRAPH_ANN_SEED: u64 = crate::DEFAULT_SEED + 10;

/// Defines a distance metric which we can use with heaps.  Lower == better

#[derive(Copy, Clone, Debug)]
pub struct DistanceFromEntity<A>(pub f32, pub A);

impl <A> DistanceFromEntity<A> {
    pub fn to_tup(&self) -> (&A, f32) {
        (&self.1, self.0)
    }

    pub fn new(distance: f32, entity: A) -> DistanceFromEntity<A> {
        DistanceFromEntity(distance, entity)
    }
}

impl <A: Clone> DistanceFromEntity<A> {
    pub fn to_tup_cloned(&self) -> (A, f32) {
        (self.1.clone(), self.0)
    }
}

// Min Heap, so reverse order
impl <A: Ord> Ord for DistanceFromEntity<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        FloatOrd(other.0).cmp(&FloatOrd(self.0))
            .then_with(|| other.1.cmp(&self.1))
    }
}

impl <A: Ord> PartialOrd for DistanceFromEntity<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl <A: PartialEq> PartialEq for DistanceFromEntity<A> {
    fn eq(&self, other: &Self) -> bool {
        FloatOrd(self.0) == FloatOrd(other.0) && self.1 == other.1
    }
}

impl <A: Eq> Eq for DistanceFromEntity<A> {}

pub type NodeDistance = DistanceFromEntity<NodeID>;

/// Struct which tracks the top K nodes according to some distance.  Useful outside of ANN as well.
pub struct TopK {
    heap: BinaryHeap<Reverse<NodeDistance>>,
    k: usize
}

impl TopK {
    pub fn new(k: usize) -> Self {
        TopK {
            k: k,
            heap: BinaryHeap::with_capacity(k+1)
        }
    }

    pub fn push(&mut self, node_id: NodeID, score: f32) {
        self.push_nd(Reverse(NodeDistance::new(score, node_id)));
    }

    fn push_nd(&mut self, nd: Reverse<NodeDistance>) {
        self.heap.push(nd);
        if self.heap.len() > self.k {
            self.heap.pop();
        }
    }

    pub fn into_sorted(self) -> Vec<NodeDistance> {
        let mut results: Vec<NodeDistance> = self.heap.into_iter()
            .map(|n| n.0).collect();
        results.sort_by_key(|n| FloatOrd(n.0));
        results
    }

    pub fn extend(&mut self, other: TopK) {
        other.heap.into_iter().for_each(|nd| {
            self.push_nd(nd);
        });
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }
}

/// This Ann hill climbs from random starting nodes within the graph.  if the graph isn't fully
/// connected, good luck.  Depending on the smoothness of the embeddings amongst neighbors, has a
/// habit of running into local minimas.  It's fine, just not anything special.
#[derive(Debug)]
pub struct Ann {
    k: usize,
    max_steps: usize,
    seed: u64
}

impl Ann {
    pub fn new(k: usize, max_steps: usize, seed: u64) -> Self {
        Ann {k, max_steps, seed}
    }

    pub fn find<G: CGraph + Send + Sync>(
        &self, 
        query: &[f32],
        graph: &G, 
        embeddings: &EmbeddingStore,
    ) -> Vec<NodeDistance> {
        let mut rng = XorShiftRng::seed_from_u64(self.seed);
        hill_climb(
            Entity::Embedding(query), 
            graph,
            embeddings,
            self.k,
            self.max_steps,
            &mut rng)
    }
    
}

/// [`Ann`] over a named graph: finds nodes close to a query by hill climbing through the graph,
/// by the distances of embeddings that index the graph's nodes.
#[derive(Clone)]
pub struct GraphAnn {
    graph: NamedGraph,
    max_steps: usize
}

impl GraphAnn {
    /// Steps the Python bindings explore when not told otherwise.
    pub const DEFAULT_MAX_STEPS: usize = 1000;

    /// Searches `graph` for `max_steps` steps, plus one per neighbour asked for.
    pub fn new(graph: &NamedGraph, max_steps: usize) -> Self {
        GraphAnn { graph: graph.clone(), max_steps }
    }

    pub fn graph(&self) -> &NamedGraph {
        &self.graph
    }

    pub fn max_steps(&self) -> usize {
        self.max_steps
    }

    /// The `k` nodes found closest to `query`, closest first, by `embeddings`' distance.  The
    /// query is looked up in `embeddings`, which must index the graph's nodes, else an
    /// [`Error::VocabMismatch`].
    pub fn find(
        &self,
        query: &Query,
        embeddings: &NodeEmbeddings,
        k: usize,
        seed: u64
    ) -> Result<Vec<NodeDistance>, Error> {
        ensure_aligned(self.graph.vocab(), embeddings.vocab(), "Graph and NodeEmbeddings")?;
        let query = embeddings.resolve(query)?;
        let ann = Ann::new(k, self.max_steps + k, seed);
        Ok(ann.find(query, self.graph.graph(), embeddings.store()))
    }
}

// This hill climbs.  We start with a node and compute the embeddings for each node.  We greedily
// explore the edges where the distance is minmized.  We return the best nodes after performing the
// search `max_steps` times.
fn hill_climb<'a, G: CGraph, R: Rng>(
    needle: Entity<'a>, 
    graph: &G, 
    es: &EmbeddingStore,
    k: usize,
    mut max_steps: usize,
    rng: &mut R
) -> Vec<NodeDistance> {
    let distribution = Uniform::new(0, graph.len());

    let mut heap = BinaryHeap::new();
    let mut best = TopK::new(k);
    let mut seen = HashSet::new();

    while max_steps > 0 {

        // Find a starting node, randomly selected
        heap.clear();
        let start_node = distribution.sample(rng);
        seen.insert(start_node.clone());
        let start_d = es.compute_distance(&needle, &Entity::Node(start_node.clone()));
        let start = NodeDistance::new(start_d, start_node);
        heap.push(start.clone());

        loop {
            // Hardcoded restart rate for the time being
            if rng.gen::<f32>() < 0.05f32 {
                break
            }

            let cur_node = heap.pop().expect("Shouldn't be empty!");
            best.push(cur_node.1, cur_node.0);
            // Get edges, compute distances between them and needle, add to the heap
            for edge in graph.get_edges(cur_node.1).0.iter() {
                if !seen.contains(edge) {
                    seen.insert(*edge);
                    let dist = es.compute_distance(&needle, &Entity::Node(*edge));
                    heap.push(NodeDistance::new(dist, *edge));
                }
            }

            max_steps -= 1;
            if max_steps == 0 || heap.len() == 0 {
                break
            }
        }

    }
    best.into_sorted()
}

#[cfg(test)]
mod ann_tests {
    use super::*;

    fn build_star_edges() -> Vec<(usize, usize, f32)> {
        let mut edges = Vec::new();
        let max = 100;
        for ni in 0..max {
            for no in (ni+1)..max {
                edges.push((ni, no, 1f32));
                edges.push((no, ni, 1f32));
            }
        }
        edges
    }

    fn named_clique() -> (NamedGraph, NodeEmbeddings) {
        let mut gb = crate::named_graph::GraphBuilder::new();
        for (a, b, _) in build_star_edges().into_iter().filter(|(a, b, _)| a < b && *b < 10) {
            gb.add_edge(("n", &a.to_string()), ("n", &b.to_string()), 1., crate::graph::EdgeType::Undirected);
        }
        let g = gb.build(true).unwrap();
        let mut emb = crate::embeddings::NodeEmbeddingsBuilder::new(crate::distance::Distance::Euclidean);
        for (_, name) in g.vocab().iter() {
            emb.add_embedding("n", name, &[name.parse::<f32>().unwrap()]).unwrap();
        }
        (g, emb.build().unwrap())
    }

    #[test]
    fn test_graph_ann_matches_ann() {
        let (g, emb) = named_clique();
        let gann = GraphAnn::new(&g, 20);
        let query = Query::Embedding(vec![3.2]);
        let found = gann.find(&query, &emb, 3, GRAPH_ANN_SEED).unwrap();
        let expected = Ann::new(3, 23, GRAPH_ANN_SEED).find(&[3.2], g.graph(), emb.store());
        assert_eq!(found, expected);
        assert_eq!(found.len(), 3);

        let by_node = gann.find(&Query::Node("n".into(), "3".into()), &emb, 3, GRAPH_ANN_SEED).unwrap();
        assert_eq!(by_node, Ann::new(3, 23, GRAPH_ANN_SEED).find(&[3.], g.graph(), emb.store()));
        assert!(matches!(gann.find(&Query::NodeId(100), &emb, 3, 1), Err(Error::NodeIdOutOfRange { .. })));
    }

    /// Embeddings must index the graph's nodes: these list them in another order.
    #[test]
    fn test_graph_ann_rejects_misaligned_embeddings() {
        let (g, _) = named_clique();
        let mut names: Vec<_> = g.vocab().iter().map(|(_, n)| n.to_string()).collect();
        names.reverse();
        let mut emb = crate::embeddings::NodeEmbeddingsBuilder::new(crate::distance::Distance::Euclidean);
        for name in names.iter() {
            emb.add_embedding("n", name, &[0.]).unwrap();
        }
        let reordered = emb.build().unwrap();
        let query = Query::Embedding(vec![0.]);
        assert!(matches!(GraphAnn::new(&g, 10).find(&query, &reordered, 2, 1), Err(Error::VocabMismatch(_))));
    }

    #[test]
    fn test_top_k() {
        let mut top_k = TopK::new(3);
        top_k.push(1, 0.1);
        top_k.push(2, 0.2);
        top_k.push(3, 0.3);
        top_k.push(4, 0.15);
        top_k.push(5, 0.01);

        let results = top_k.into_sorted();

        println!("results: {:?}", results);
        assert_eq!(results[0], NodeDistance::new(0.01, 5));
        assert_eq!(results[1], NodeDistance::new(0.1, 1));
        assert_eq!(results[2], NodeDistance::new(0.15, 4));
    }
}
