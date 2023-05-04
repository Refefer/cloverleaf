
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

pub trait ModifiableGraph {
    /// Get edges and corresponding weights
    fn modify_edges(&mut self, idx: NodeID) -> (&mut [NodeID], &mut [f32]);
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
    pub fn construct_from_edges(edges: Vec<(NodeID, NodeID, f32)>) -> Self {

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

impl NormalizedGraph for NormalizedCSR {}

#[derive(Clone)]
pub struct CumCSR(CSR);

impl CumCSR {
    pub fn convert(mut csr: CSR) -> Self {
        for start_stop in csr.rows.windows(2) {
            let (start, stop) = (start_stop[0], start_stop[1]);
            if start < stop {
                convert_edges_to_cdf(&mut csr.weights[start..stop]);
            }
        }
        CumCSR(csr)
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

impl CDFGraph for CumCSR {}

pub struct OptCDFGraph<'a,G> {
    graph: &'a G,
    weights: Vec<f32>
}

impl <'a,G:Graph> OptCDFGraph<'a,G> {
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

impl <'a,G:CDFGraph> OptCDFGraph<'a,G> {
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

impl <'a,G:Graph> Graph for OptCDFGraph<'a,G> {
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

#[derive(Clone,Copy)]
pub struct CDFtoP<'a> {
    cdf: &'a [f32],
    idx: usize
}

impl <'a> CDFtoP<'a> {
    pub fn new(weights: &'a [f32]) -> Self {
        CDFtoP { cdf: weights, idx: 0 }
    }
}

impl <'a> Iterator for CDFtoP<'a> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.cdf.len() {
            let p = if self.idx == 0 {
                Some(self.cdf[self.idx])
            } else {
                Some(self.cdf[self.idx] - self.cdf[self.idx-1])
            };
            self.idx += 1;
            p
        } else {
            None
        }
    }
}

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

        let csr = CSR::construct_from_edges(edges);
        assert_eq!(csr.rows, vec![0, 1, 4, 5]);
        assert_eq!(csr.columns, vec![1, 1, 2, 0, 0]);
        assert_eq!(csr.weights, vec![1., 3., 2., 10., 2.5]);
    }

    #[test]
    fn test_graph() {
        let edges = build_edges();

        let mut csr = CSR::construct_from_edges(edges);
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

        let csr = CSR::construct_from_edges(edges);
        let mk = NormalizedCSR::convert(csr);

        assert_eq!(mk.get_edges(0), (vec![1].as_slice(), vec![1.].as_slice()));
        assert_eq!(mk.get_edges(1), (vec![1,2,0].as_slice(), vec![3./15., 2./15., 10./15.].as_slice()));
        assert_eq!(mk.get_edges(2), (vec![0].as_slice(), vec![1.].as_slice()));

    }

    #[test]
    fn construct_cdf() {
        let edges = build_edges();

        let csr = CSR::construct_from_edges(edges);
        let ccsr = CumCSR::convert(csr);

        assert_eq!(ccsr.0.rows, vec![0, 1, 4, 5]);
        assert_eq!(ccsr.0.columns, vec![1, 1, 2, 0, 0]);
        assert_eq!(ccsr.0.weights, vec![1., 3./15., 5./15., 15./15., 1.]);

    }

    #[test]
    fn construct_cdf_to_p() {
        let edges = build_edges();

        let csr = CSR::construct_from_edges(edges);
        let ccsr = CumCSR::convert(csr);

        let (edges, weights) = ccsr.get_edges(1);
        let ps = CDFtoP::new(weights);
        let exp = vec![3./15., 2./15., 10./15.];
        ps.zip(exp.iter()).for_each(|(p, exp_p)| {
            assert!((p - exp_p).abs() < 1e-7);
        });
    }

}
