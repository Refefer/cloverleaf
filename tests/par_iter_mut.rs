//! Integration tests for the parallel mutation API on CSR-backed graph types.

use cloverleaf::graph::{CSR, CumCSR, Graph, NormalizedCSR};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

fn build_edges() -> Vec<(usize, usize, f32)> {
    vec![
        (0, 1, 1.),
        (1, 1, 3.),
        (1, 2, 2.),
        (2, 0, 2.5),
        (1, 0, 10.),
    ]
}

/// Sanity check that a plain `.for_each` reaches every weight.  After dedup the
/// edges are sorted by (from, to), so node 1's weights are [10, 3, 2] -> doubled
/// to [20, 6, 4].
#[test]
fn par_iter_mut_for_each_doubles_weights() {
    let mut csr = CSR::construct_from_edges(build_edges(), true);
    csr.par_iter_mut().for_each(|(_e, w)| {
        w.iter_mut().for_each(|x| *x *= 2.0);
    });
    assert_eq!(csr.get_edges(0), ([1usize].as_slice(), [2.0f32].as_slice()));
    assert_eq!(csr.get_edges(1).1, [20.0, 6.0, 4.0].as_slice());
    assert_eq!(csr.get_edges(2), ([0usize].as_slice(), [5.0f32].as_slice()));
}

/// Verify `.enumerate()` hands each worker the right node index.
#[test]
fn par_iter_mut_enumerate_writes_node_id() {
    let mut csr = CSR::construct_from_edges(build_edges(), true);
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
fn par_iter_mut_zero_degree_node() {
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
fn par_iter_mut_visits_each_node_once() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let mut csr = CSR::construct_from_edges(build_edges(), true);
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
fn normalized_csr_par_iter_mut_delegates() {
    let mut norm = NormalizedCSR::convert(CSR::construct_from_edges(build_edges(), true));
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
fn cum_csr_par_iter_mut_delegates() {
    let mut cum = CumCSR::convert(CSR::construct_from_edges(build_edges(), true));
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
fn par_iter_mut_scale_1000_nodes() {
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
