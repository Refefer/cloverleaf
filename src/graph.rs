
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
        let start = self.rows[idx];
        let stop  = self.rows[idx+1];
        (&self.columns[start..stop], &self.weights[start..stop])
    }
    
    // Get edges and corresponding weights
    fn modify_edges(&mut self, idx: NodeID) -> (&mut [NodeID], &mut [f32]) {
        let start = self.rows[idx];
        let stop  = self.rows[idx+1];
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
            slice.iter_mut().for_each(|w| *w /= denom);
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
    
    /// Get edges and corresponding weights
    fn modify_edges(&mut self, idx: NodeID) -> (&mut [NodeID], &mut [f32]) {
        self.0.modify_edges(idx)
    }
 
}

impl NormalizedGraph for NormalizedCSR {}

pub struct CumCSR(CSR);

impl CumCSR {
    pub fn convert(mut csr: CSR) -> Self {
        for start_stop in csr.rows.windows(2) {
            let (start, stop) = (start_stop[0], start_stop[1]);
            let slice = &mut csr.weights[start..stop];
            let denom = slice.iter().sum::<f32>();
            let mut acc = 0.;
            slice.iter_mut().for_each(|w| {
                acc += *w;
                *w = acc / denom;
            });
            slice[slice.len() - 1] = 1.;
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
    
    /// Get edges and corresponding weights
    fn modify_edges(&mut self, idx: NodeID) -> (&mut [NodeID], &mut [f32]) {
        self.0.modify_edges(idx)
    }
 
}

impl CDFGraph for CumCSR {}

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


}
