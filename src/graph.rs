//! This is the core graph library.
//! While we define a number of different representations, including mutability, in practice we're
//! only using CSR variants which also assumes static construction.  We have a couplte of tricks
//! defined in here to allow for swapping of edges while minimizing the amount of memory we have to
//! copy.

use rayon::prelude::ParallelSliceMut;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};

pub type NodeID = usize;

pub trait Graph {
    /// Get number of nodes in graph
    fn len(&self) -> usize;
    
    /// Get number of edges in graph
    fn edges(&self) -> usize;

    /// Get degree of node in graph
    fn degree(&self, idx: NodeID) -> usize;

    /// Get edges and corresponding weights
    fn get_edges(&self, idx: NodeID) -> (&[NodeID], &[f32]);
    
    /// Get edge offset in graph
    fn get_edge_range(&self, idx: NodeID) -> (usize, usize);
    
}

/// trait which allows graphs to be updated
pub trait ModifiableGraph {
    /// Get edges and corresponding weights
    fn modify_edges(&mut self, idx: NodeID) -> (&mut [NodeID], &mut [f32]);
}

/// trait which allows graphs to be updated in parallel.  Implementors must expose per-node
/// edge and weight slices that live in disjoint regions of flat backing buffers.
pub trait ParallelModifiableGraph: ModifiableGraph {
    /// Parallel iterator yielding `(&mut edges, &mut weights)` per node, in node order.
    fn par_iter_mut(&mut self) -> CSRParIterMut<'_>;
}

/// Trait which transposes a graph's adjacency list
pub trait Transpose {
    type Output: Graph;

    /// Transpose function to a new Graph
    fn transpose(&self) -> Self::Output;
}

/// Used for trait bounds.  Confirms the underlying weights for each node
/// sum to 1, as in a transition matrix.
pub trait NormalizedGraph: Graph {}

/// Used for trait bounds.  Confirms the weights are normalized transition
/// matrix, optimized in cumulative distribution function.
pub trait CDFGraph: Graph {}

/// Compressed Sparse Row Format.  We use this for graphs since adjancency
/// lists tend to use more memory.
#[derive(Clone)]
pub struct CSR {
    rows: Vec<NodeID>,
    columns: Vec<NodeID>,
    weights: Vec<f32>
}

impl CSR {
    pub fn construct_from_edges(
        mut edges: Vec<(NodeID, NodeID, f32)>, 
        deduplicate: bool
    ) -> Self {

        if deduplicate {
            CSR::deduplicate_edges(&mut edges);
        }

        // Determine the number of rows in the adjacency graph
        let max_node = edges.iter().map(|(from_node, to_node, _)| {
            *from_node.max(to_node)
        }).max().unwrap_or(0);

        // Figure out how many out edges per node
        let mut rows = vec![0; max_node+2];
        edges.iter().for_each(|(from_node, _to_node, _w)| {
            rows[*from_node + 1] += 1;
        });

        // Convert to row offset format
        let mut offset = 0;
        rows.iter_mut().skip(1).for_each(|count| {
            offset += *count;
            *count = offset;
        });

        // Insert columns and weights
        let mut counts  = vec![0; max_node+1];
        let mut columns = vec![0; edges.len()];
        let mut data    = vec![0f32; edges.len()];
        edges.into_iter().for_each(|(from_node, to_node, weight)| {
            let idx = rows[from_node] + counts[from_node];
            columns[idx] = to_node;
            data[idx] = weight;
            counts[from_node] += 1;
        });

        CSR { rows, columns, weights: data }
    }

    /// Parallel mutable access to each node's edge and weight slices.
    pub fn par_iter_mut(&mut self) -> CSRParIterMut<'_> {
        CSRParIterMut {
            rows: &self.rows,
            columns: &mut self.columns,
            weights: &mut self.weights,
        }
    }

    fn deduplicate_edges(
        edges: &mut Vec<(NodeID, NodeID, f32)>
    ) -> () {
        edges.par_sort_by_key(|(f_n, t_n, _)| (*f_n, *t_n));
        let mut cur_record = 0;
        let mut idx = 1;
        while idx < edges.len() {
            let (f_n, t_n, w) = edges[idx];
            let c_r = edges[cur_record];
            // Same edge, add the weights.
            if f_n == c_r.0 && t_n == c_r.1 {
                (&mut edges[cur_record]).2 += w;
            } else {
                // Different record, move it
                cur_record += 1;
                edges[cur_record] = edges[idx];
            }
            idx += 1;
        }
        edges.truncate(cur_record + 1);

    }

}

impl Graph for CSR {
    // Get number of nodes in graph
    fn len(&self) -> usize {
        self.rows.len() - 1
    }
    
    // Get number of nodes in graph
    fn edges(&self) -> usize {
        self.weights.len()
    }

    // Get degree of node in graph
    fn degree(&self, idx: NodeID) -> usize {
        self.rows[idx+1] - self.rows[idx]
    }

    // Get edges and corresponding weights
    fn get_edges(&self, idx: NodeID) -> (&[NodeID], &[f32]) {
        let (start, stop) = self.get_edge_range(idx);
        (&self.columns[start..stop], &self.weights[start..stop])
    }

    // get edge range
    fn get_edge_range(&self, idx: NodeID) -> (usize, usize) {
        let start = self.rows[idx];
        let stop  = self.rows[idx+1];
        (start, stop)
    }
    
}

impl ModifiableGraph for CSR {
    // Get edges and corresponding weights
    fn modify_edges(&mut self, idx: NodeID) -> (&mut [NodeID], &mut [f32]) {
        let (start, stop) = self.get_edge_range(idx);
        (&mut self.columns[start..stop], &mut self.weights[start..stop])
    }

}

impl ParallelModifiableGraph for CSR {
    fn par_iter_mut(&mut self) -> CSRParIterMut<'_> {
        CSR::par_iter_mut(self)
    }
}

/// Zero-allocation parallel iterator over each node's mutable edge and weight slices.
/// Splits via rayon are O(1) — we carve the flat CSR buffers with split_at_mut.
pub struct CSRParIterMut<'a> {
    rows: &'a [NodeID],
    columns: &'a mut [NodeID],
    weights: &'a mut [f32],
}

impl<'a> ParallelIterator for CSRParIterMut<'a> {
    type Item = (&'a mut [NodeID], &'a mut [f32]);

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(IndexedParallelIterator::len(self))
    }
}

impl<'a> IndexedParallelIterator for CSRParIterMut<'a> {
    fn len(&self) -> usize {
        self.rows.len().saturating_sub(1)
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(CSRProducer {
            rows: self.rows,
            columns: self.columns,
            weights: self.weights,
        })
    }
}

/// Producer used by rayon to drive parallel splits of a CSRParIterMut.
struct CSRProducer<'a> {
    rows: &'a [NodeID],
    columns: &'a mut [NodeID],
    weights: &'a mut [f32],
}

impl<'a> Producer for CSRProducer<'a> {
    type Item = (&'a mut [NodeID], &'a mut [f32]);
    type IntoIter = CSRSeqIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        CSRSeqIter {
            rows: self.rows,
            cols: self.columns,
            wgts: self.weights,
        }
    }

    fn split_at(self, mid: usize) -> (Self, Self) {
        // Invariant: columns.len() == weights.len() == rows.last() - rows.first().
        // After splits rows[0] may be non-zero, so subtract base to get the local boundary.
        let base = self.rows[0];
        let boundary = self.rows[mid] - base;
        let (cols_l, cols_r) = self.columns.split_at_mut(boundary);
        let (wgts_l, wgts_r) = self.weights.split_at_mut(boundary);
        let rows_l = &self.rows[..=mid];
        let rows_r = &self.rows[mid..];
        (
            CSRProducer { rows: rows_l, columns: cols_l, weights: wgts_l },
            CSRProducer { rows: rows_r, columns: cols_r, weights: wgts_r },
        )
    }
}

/// Sequential leaf iterator; peels one node off the front of its buffers per call.
struct CSRSeqIter<'a> {
    rows: &'a [NodeID],
    cols: &'a mut [NodeID],
    wgts: &'a mut [f32],
}

impl<'a> Iterator for CSRSeqIter<'a> {
    type Item = (&'a mut [NodeID], &'a mut [f32]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.rows.len() < 2 {
            return None;
        }
        // Move the full-length slice out of self so we can split it; the tail
        // goes back in.  mem::take works here since &mut [T] has a Default impl.
        let len = self.rows[1] - self.rows[0];
        let cols = std::mem::take(&mut self.cols);
        let wgts = std::mem::take(&mut self.wgts);
        let (ch, ct) = cols.split_at_mut(len);
        let (wh, wt) = wgts.split_at_mut(len);
        self.cols = ct;
        self.wgts = wt;
        self.rows = &self.rows[1..];
        Some((ch, wh))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.rows.len().saturating_sub(1);
        (n, Some(n))
    }
}

impl<'a> DoubleEndedIterator for CSRSeqIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.rows.len() < 2 {
            return None;
        }
        // Peel the tail node by splitting off the last `last_len` elements.
        let n = self.rows.len() - 1;
        let last_len = self.rows[n] - self.rows[n - 1];
        let keep = self.cols.len() - last_len;
        let cols = std::mem::take(&mut self.cols);
        let wgts = std::mem::take(&mut self.wgts);
        let (ch, ct) = cols.split_at_mut(keep);
        let (wh, wt) = wgts.split_at_mut(keep);
        self.cols = ch;
        self.wgts = wh;
        self.rows = &self.rows[..n];
        Some((ct, wt))
    }
}

impl<'a> ExactSizeIterator for CSRSeqIter<'a> {
    fn len(&self) -> usize {
        self.rows.len().saturating_sub(1)
    }
}

/// Normalizes sum of weights for a node to 1
pub struct NormalizedCSR(CSR);

impl NormalizedCSR {
    pub fn convert(mut csr: CSR) -> Self {
        for start_stop in csr.rows.windows(2) {
            let (start, stop) = (start_stop[0], start_stop[1]);
            let slice = &mut csr.weights[start..stop];
            let denom = slice.iter().sum::<f32>();
            if denom > 0f32 {
                slice.iter_mut().for_each(|w| *w /= denom);
            } else {
                let n = slice.len() as f32;
                slice.iter_mut().for_each(|w| *w = 1f32 / n );
            }
        }
        NormalizedCSR(csr)
    }

    /// Parallel mutable access to each node's edge and weight slices.
    pub fn par_iter_mut(&mut self) -> CSRParIterMut<'_> {
        self.0.par_iter_mut()
    }
}

impl Graph for NormalizedCSR {
    /// Get number of nodes in graph
    fn len(&self) -> usize {
        self.0.len()
    }
    
    /// Get number of nodes in graph
    fn edges(&self) -> usize {
        self.0.edges()
    }

    /// Get degree of node in graph
    fn degree(&self, idx: NodeID) -> usize {
        self.0.degree(idx)
    }

    /// Get edges and corresponding weights
    fn get_edges(&self, idx: NodeID) -> (&[NodeID], &[f32]) {
        self.0.get_edges(idx)
    }
    
    // get edge range
    fn get_edge_range(&self, idx: NodeID) -> (usize, usize) {
        self.0.get_edge_range(idx)
    }

     
}

impl ModifiableGraph for NormalizedCSR {
    /// Get edges and corresponding weights
    fn modify_edges(&mut self, idx: NodeID) -> (&mut [NodeID], &mut [f32]) {
        self.0.modify_edges(idx)
    }
}

impl ParallelModifiableGraph for NormalizedCSR {
    fn par_iter_mut(&mut self) -> CSRParIterMut<'_> {
        self.0.par_iter_mut()
    }
}

impl NormalizedGraph for NormalizedCSR {}

/// This is the main one used in Cloverleaf - converts CSR formatted graphs into CDF format to make sampling from
/// edges efficient (log(N)).  
#[derive(Clone)]
pub struct CumCSR(CSR);

impl CumCSR {
    /// Converts a Compressed Sparse Row Format into a CDF version of weights
    pub fn convert(mut csr: CSR) -> Self {
        for start_stop in csr.rows.windows(2) {
            let (start, stop) = (start_stop[0], start_stop[1]);
            if start < stop {
                convert_edges_to_cdf(&mut csr.weights[start..stop]);
            }
        }
        CumCSR(csr)
    }

    /// Swaps out the underlying edge weghts with new edge weights and confirm invariants
    pub fn clone_with_edges(&self, weights: Vec<f32>) -> Result<CumCSR,&'static str> {
        if weights.len() != self.0.weights.len() {
            Err("weights lengths not equal!")?
        }

        let graph = CSR {
            rows: self.0.rows.clone(),
            columns: self.0.columns.clone(),
            weights: weights
        };

        // Test that the weights are properly CDF
        for node_id in 0..graph.len() {
            let weights = self.get_edges(node_id).1;
            for pair in weights.windows(2) {
                let &[p, n] = pair else { panic!("Should never hit!") };
                if p > 1.0 {
                    Err("Edge weight exceeds 1.0, illegal in CDF")?
                } else if n < p {
                    Err("Edge weight for node in decreasing order")?
                }
            }

            if !weights.is_empty() && weights[weights.len() - 1] > 1.0 {
                Err("Edge weight exceeds 1.0, illegal in CDF")?
            }
        }

        Ok(CumCSR(graph))
    }

    /// Parallel mutable access to each node's edge and weight slices.  Note that arbitrary
    /// mutation can break the CDF invariant - callers are responsible for restoring it.
    pub fn par_iter_mut(&mut self) -> CSRParIterMut<'_> {
        self.0.par_iter_mut()
    }
}

impl Transpose for CumCSR {
    type Output = CumCSR;

    fn transpose(&self) -> CumCSR {

        // Count the inbound edges in the graph
        let mut inbound_counts = vec![0usize; self.len()];
        for node_id in 0..self.len() {
            let edges = self.get_edges(node_id).0;
            for inbound_node in edges {
                inbound_counts[*inbound_node] += 1;
            }
        }

        // Convert the counts into weight offsets
        let mut rows = vec![0usize; self.len() + 1];
        let mut state = 0;
        rows.iter_mut().zip(inbound_counts.iter()).for_each(|(row, count)| {
            *row = state;
            state += count;
        });

        let n = rows.len() - 1;
        rows[n] = state;

        // Reverse the edge weights into the weights and columns matrix
        let mut weights = self.0.weights.clone();
        let mut columns = self.0.columns.clone();
        for node_id in 0..self.len() {
            let (edges, edge_weights) = self.get_edges(node_id);
            for (edge, weight) in edges.iter().zip(CDFtoP::new(edge_weights)) {
                let offset = rows[*edge];
                let idx = offset + inbound_counts[*edge] - 1;
                weights[idx] = weight;
                columns[idx] = node_id;
                inbound_counts[*edge] -= 1;
            }
        }

        CumCSR::convert(CSR { rows, columns, weights })

    }
}

impl Graph for CumCSR {
    /// Get number of nodes in graph
    fn len(&self) -> usize {
        self.0.len()
    }

    /// Get number of nodes in graph
    fn edges(&self) -> usize {
        self.0.edges()
    }

    /// Get degree of node in graph
    fn degree(&self, idx: NodeID) -> usize {
        self.0.degree(idx)
    }

    /// Get edges and corresponding weights
    fn get_edges(&self, idx: NodeID) -> (&[NodeID], &[f32]) {
        self.0.get_edges(idx)
    }
    
    /// Get edge Range
    fn get_edge_range(&self, idx: NodeID) -> (usize, usize) {
        self.0.get_edge_range(idx)
    }

}

impl ModifiableGraph for CumCSR {

    /// Get edges and corresponding weights
    fn modify_edges(&mut self, idx: NodeID) -> (&mut [NodeID], &mut [f32]) {
        self.0.modify_edges(idx)
    }

}

impl ParallelModifiableGraph for CumCSR {
    fn par_iter_mut(&mut self) -> CSRParIterMut<'_> {
        self.0.par_iter_mut()
    }
}

impl CDFGraph for CumCSR {}

/// This is a graph which allows us to swap in a new set of edge weights without having to copy the
/// entire graph.  We use it in cases where policies update edge transition probabilities.
pub struct OptCDFGraph<'a, G> {
    graph: &'a G,
    weights: Vec<f32>
}

impl <'a, G:Graph> OptCDFGraph<'a,G> {
    pub fn new(graph: &'a G, weights: Vec<f32>) -> Self {
        let mut s = OptCDFGraph { graph, weights };
        s.convert_edges();
        s
    }

    pub fn into_weights(self) -> Vec<f32> {
        self.weights
    }

    pub fn convert_edges(&mut self) {
        for idx in 0..self.len() {
            let (start, stop) = self.get_edge_range(idx);
            if start < stop {
                convert_edges_to_cdf(&mut self.weights[start..stop]);
            }
        }
    }

}

impl <'a, G:CDFGraph> OptCDFGraph<'a, G> {
    pub fn clone_from_cdf(graph: &'a G) -> Self {
        let mut weights = vec![0f32; graph.edges()];
        for node_id in 0..graph.len() {
            let w = graph.get_edges(node_id).1;
            let (start, stop) = graph.get_edge_range(node_id);
            weights[start..stop].clone_from_slice(w);
        }

        OptCDFGraph { graph, weights }
    }

}

impl <'a, G:Graph> Graph for OptCDFGraph<'a,G> {
    /// Get number of nodes in graph
    fn len(&self) -> usize {
        self.graph.len()
    }

    /// Get number of nodes in graph
    fn edges(&self) -> usize {
        self.graph.edges()
    }

    /// Get degree of node in graph
    fn degree(&self, idx: NodeID) -> usize {
        self.graph.degree(idx)
    }

    /// Get edges and corresponding weights
    fn get_edges(&self, idx: NodeID) -> (&[NodeID], &[f32]) {
        let edges = self.graph.get_edges(idx).0;
        let (start, stop) = self.get_edge_range(idx);
        let weights = &self.weights[start..stop];
        (edges, weights)
    }
    
    /// Get edge Range
    fn get_edge_range(&self, idx: NodeID) -> (usize, usize) {
        self.graph.get_edge_range(idx)
    }

}

impl <'a,G:Graph> CDFGraph for OptCDFGraph<'a,G> {}

/// Struct which converts CDF format to transition probabilities.
#[derive(Clone,Copy)]
pub struct CDFtoP<'a> {
    cdf: &'a [f32],
    idx: usize
}

impl <'a> CDFtoP<'a> {
    pub fn new(weights: &'a [f32]) -> Self {
        CDFtoP { cdf: weights, idx: 0 }
    }

    pub fn prob(&self, idx: usize) -> f32 {
        if idx == 0 {
            self.cdf[idx]
        } else {
            self.cdf[idx] - self.cdf[idx - 1]
        }
    }
}

impl <'a> Iterator for CDFtoP<'a> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.cdf.len() {
            let p = self.prob(self.idx);
            self.idx += 1;
            Some(p)
        } else {
            None
        }
    }
}

/// Converts a set of weights to CDF
pub fn convert_edges_to_cdf(weights: &mut [f32]) {
    let mut denom = weights.iter().sum::<f32>();
    if denom == 0f32 {
        // If we have no weights, set all weights to uniform.
        weights.iter_mut().for_each(|w| {
            *w = 1.
        });
        denom = weights.len() as f32;
    }

    let mut acc = 0.;
    weights.iter_mut().for_each(|w| {
        acc += *w;
        *w = acc / denom;
    });

    // Accumulation error can result in it not equaling 1
    weights[weights.len() - 1] = 1.;
}

#[cfg(test)]
mod csr_tests {
    use super::*;

    fn build_edges() -> Vec<(usize, usize, f32)> {
        vec![
            (0, 1, 1.),
            (1, 1, 3.),
            (1, 2, 2.),
            (2, 0, 2.5),
            (1, 0, 10.),
        ]
    }

    #[test]
    fn construct_csr() {
        let edges = build_edges();

        let csr = CSR::construct_from_edges(edges, true);
        assert_eq!(csr.rows, vec![0, 1, 4, 5]);
        assert_eq!(csr.columns, vec![1, 1, 2, 0, 0]);
        assert_eq!(csr.weights, vec![1., 3., 2., 10., 2.5]);
    }

    #[test]
    fn test_graph() {
        let edges = build_edges();

        let mut csr = CSR::construct_from_edges(edges, true);
        assert_eq!(csr.len(), 3);
        assert_eq!(csr.degree(0), 1);
        assert_eq!(csr.degree(1), 3);
        assert_eq!(csr.degree(2), 1);
        assert_eq!(csr.get_edges(2), (vec![0].as_slice(), vec![2.5].as_slice()));
        assert_eq!(csr.get_edges(1), (vec![1,2,0].as_slice(), vec![3., 2., 10.].as_slice()));

        {
            let (_edges, weights) = csr.modify_edges(1);
            weights[2] = 20.;
        }

        assert_eq!(csr.get_edges(1), (vec![1,2,0].as_slice(), vec![3., 2., 20.].as_slice()));
    }

    #[test]
    fn construct_mk() {
        let edges = build_edges();

        let csr = CSR::construct_from_edges(edges, true);
        let mk = NormalizedCSR::convert(csr);

        assert_eq!(mk.get_edges(0), (vec![1].as_slice(), vec![1.].as_slice()));
        assert_eq!(mk.get_edges(1), (vec![1,2,0].as_slice(), vec![3./15., 2./15., 10./15.].as_slice()));
        assert_eq!(mk.get_edges(2), (vec![0].as_slice(), vec![1.].as_slice()));

    }

    #[test]
    fn construct_cdf() {
        let edges = build_edges();

        let csr = CSR::construct_from_edges(edges, true);
        let ccsr = CumCSR::convert(csr);

        assert_eq!(ccsr.0.rows, vec![0, 1, 4, 5]);
        assert_eq!(ccsr.0.columns, vec![1, 1, 2, 0, 0]);
        assert_eq!(ccsr.0.weights, vec![1., 3./15., 5./15., 15./15., 1.]);

    }

    #[test]
    fn construct_cdf_to_p() {
        let edges = build_edges();

        let csr = CSR::construct_from_edges(edges, true);
        let ccsr = CumCSR::convert(csr);

        let weights = ccsr.get_edges(1).1;
        let ps = CDFtoP::new(weights);
        let exp = vec![3./15., 2./15., 10./15.];
        ps.zip(exp.iter()).for_each(|(p, exp_p)| {
            assert!((p - exp_p).abs() < 1e-7);
        });
    }

    #[test]
    fn transpose_matrix() {
        let edges = build_edges();

        let csr = CSR::construct_from_edges(edges, true);
        let ccsr = CumCSR::convert(csr);

        let _t_ccsr = ccsr.transpose();

    }

    /// Sanity check that a plain `.for_each` reaches every weight.  After dedup the
    /// edges are sorted by (from, to), so node 1's weights are [10, 3, 2] -> doubled
    /// to [20, 6, 4].
    #[test]
    fn test_par_iter_mut_for_each_doubles_weights() {
        let edges = build_edges();
        let mut csr = CSR::construct_from_edges(edges, true);
        csr.par_iter_mut().for_each(|(_e, w)| {
            w.iter_mut().for_each(|x| *x *= 2.0);
        });
        assert_eq!(csr.get_edges(0), ([1usize].as_slice(), [2.0f32].as_slice()));
        assert_eq!(csr.get_edges(1).1, [20.0, 6.0, 4.0].as_slice());
        assert_eq!(csr.get_edges(2), ([0usize].as_slice(), [5.0f32].as_slice()));
    }

    /// Verify `.enumerate()` hands each worker the right node index.
    #[test]
    fn test_par_iter_mut_enumerate_writes_node_id() {
        let edges = build_edges();
        let mut csr = CSR::construct_from_edges(edges, true);
        csr.par_iter_mut().enumerate().for_each(|(i, (_e, w))| {
            w.iter_mut().for_each(|x| *x = i as f32);
        });
        for i in 0..csr.len() {
            for &x in csr.get_edges(i).1 {
                assert_eq!(x, i as f32);
            }
        }
    }

    /// A zero-degree node should yield an empty slice pair without tripping the split logic.
    #[test]
    fn test_par_iter_mut_zero_degree_node() {
        // node 1 has degree 0
        let edges = vec![(0usize, 2usize, 1.0f32), (2, 0, 2.0)];
        let mut csr = CSR::construct_from_edges(edges, true);
        assert_eq!(csr.degree(1), 0);

        csr.par_iter_mut().for_each(|(_e, w)| {
            if !w.is_empty() {
                w.iter_mut().for_each(|x| *x += 100.0);
            }
        });

        assert_eq!(csr.get_edges(0).1, &[101.0]);
        assert_eq!(csr.get_edges(2).1, &[102.0]);
        assert_eq!(csr.degree(1), 0);
    }

    /// Guard against double-visits or skipped nodes in the split tree.  Atomic counters
    /// let the workers tally hits in parallel without racing the assertion.
    #[test]
    fn test_par_iter_mut_visits_each_node_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let edges = build_edges();
        let mut csr = CSR::construct_from_edges(edges, true);
        let n = csr.len();
        let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();

        csr.par_iter_mut().enumerate().for_each(|(i, (_e, w))| {
            counts[i].fetch_add(1, Ordering::Relaxed);
            w.iter_mut().for_each(|x| *x = i as f32);
        });

        for (i, c) in counts.iter().enumerate() {
            assert_eq!(c.load(Ordering::Relaxed), 1, "node {i} hit count");
        }
        for i in 0..n {
            for &x in csr.get_edges(i).1 {
                assert_eq!(x, i as f32);
            }
        }
    }

    /// NormalizedCSR delegates to the inner CSR; confirm mutation is visible through the wrapper.
    #[test]
    fn test_normalized_csr_par_iter_mut_delegates() {
        let edges = build_edges();
        let mut norm = NormalizedCSR::convert(CSR::construct_from_edges(edges, true));

        norm.par_iter_mut().for_each(|(_e, w)| {
            w.iter_mut().for_each(|x| *x = 0.0);
        });

        for i in 0..norm.len() {
            for &x in norm.get_edges(i).1 {
                assert_eq!(x, 0.0);
            }
        }
    }

    /// CumCSR delegates the same way; we ignore the CDF invariant here since we're testing plumbing.
    #[test]
    fn test_cum_csr_par_iter_mut_delegates() {
        let edges = build_edges();
        let mut cum = CumCSR::convert(CSR::construct_from_edges(edges, true));

        cum.par_iter_mut().enumerate().for_each(|(i, (_e, w))| {
            w.iter_mut().for_each(|x| *x = i as f32);
        });

        for i in 0..cum.len() {
            for &x in cum.get_edges(i).1 {
                assert_eq!(x, i as f32);
            }
        }
    }

    /// Exercise the Producer::split_at path under real rayon work-stealing with enough nodes
    /// that the split tree has real depth.  3 edges/node keeps the CSR small but non-trivial.
    #[test]
    fn test_par_iter_mut_scale_1000_nodes() {
        let mut edges: Vec<(usize, usize, f32)> = Vec::with_capacity(3_000);
        for i in 0..1000usize {
            edges.push((i, (i + 1) % 1000, 1.0));
            edges.push((i, (i + 7) % 1000, 1.0));
            edges.push((i, (i + 31) % 1000, 1.0));
        }

        let mut csr = CSR::construct_from_edges(edges, true);
        csr.par_iter_mut().enumerate().for_each(|(i, (_e, w))| {
            w.iter_mut().for_each(|x| *x = i as f32);
        });

        for i in 0..csr.len() {
            for &x in csr.get_edges(i).1 {
                assert_eq!(x, i as f32);
            }
        }
    }
}
