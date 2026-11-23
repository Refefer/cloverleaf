//! Sums per-feature gradients from many batches, so they can be applied in one update.
use std::sync::atomic::{AtomicU32, Ordering};

/// A dense `rows x dims` table of gradient sums.  Batches running in parallel add to it without
/// locks: each cell is an f32 stored as bits and updated with compare-and-swap, so no
/// contribution is lost.  Takes as much memory as the embeddings it accumulates for.
pub struct GradientAccumulator {
    dims: usize,
    sums: Vec<AtomicU32>
}

impl GradientAccumulator {
    pub fn new(rows: usize, dims: usize) -> Self {
        let zero = 0f32.to_bits();
        GradientAccumulator { dims, sums: (0..rows * dims).map(|_| AtomicU32::new(zero)).collect() }
    }

    pub fn rows(&self) -> usize {
        self.sums.len() / self.dims.max(1)
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Adds `grad` to `row`'s sum.
    pub fn add(&self, row: usize, grad: &[f32]) {
        assert_eq!(grad.len(), self.dims, "gradient has the wrong number of dimensions");
        let start = row * self.dims;
        for (cell, g) in self.sums[start..start + self.dims].iter().zip(grad) {
            let mut cur = cell.load(Ordering::Relaxed);
            loop {
                let new = (f32::from_bits(cur) + *g).to_bits();
                match cell.compare_exchange_weak(cur, new, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => break,
                    Err(actual) => cur = actual
                }
            }
        }
    }

    /// Moves `row`'s sum into `out` and resets the row to zero.  Returns false, leaving `out`
    /// zeroed, if nothing was added to the row: every component is zero.
    pub fn take_row(&self, row: usize, out: &mut [f32]) -> bool {
        assert_eq!(out.len(), self.dims, "output has the wrong number of dimensions");
        let start = row * self.dims;
        let zero = 0f32.to_bits();
        let mut any = false;
        for (cell, o) in self.sums[start..start + self.dims].iter().zip(out.iter_mut()) {
            *o = f32::from_bits(cell.swap(zero, Ordering::Relaxed));
            any |= *o != 0f32;
        }
        any
    }
}

#[cfg(test)]
mod accumulator_tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn test_sums_and_resets_rows() {
        let acc = GradientAccumulator::new(3, 2);
        assert_eq!((acc.rows(), acc.dims()), (3, 2));
        acc.add(1, &[1., -2.]);
        acc.add(1, &[0.5, 0.5]);
        let mut out = vec![9f32; 2];
        assert!(acc.take_row(1, &mut out));
        assert_eq!(out, vec![1.5, -1.5]);
        assert!(!acc.take_row(1, &mut out), "taking a row resets it");
        assert_eq!(out, vec![0., 0.]);
        assert!(!acc.take_row(0, &mut out));
    }

    /// Concurrent adds to the same row are all kept.
    #[test]
    fn test_parallel_adds_are_not_lost() {
        let acc = GradientAccumulator::new(2, 4);
        (0..10_000).into_par_iter().for_each(|i| acc.add(i % 2, &[1., 2., 0.5, -1.]));
        let mut out = vec![0f32; 4];
        acc.take_row(0, &mut out);
        assert_eq!(out, vec![5000., 10000., 2500., -5000.]);
    }
}
