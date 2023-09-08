//! ProNE: Fast and Scalable Network Representation Learning
//!
//! Operates in two steps:
//!   1) Initializes network embeddings via sparse matrix factorization.
//!   2) Enhance the initial embeddings by propagating them in spectrally modulated space

#![allow(non_snake_case)]

use nalgebra_sparse::csr::CsrMatrix;
use nalgebra::{DMatrix, DVector, SVD};
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use special_fun::FloatSpecial;
use std::fmt::Write;

use crate::algos::rsvd::rsvd;
use crate::embeddings::Distance;
use crate::embeddings::EmbeddingStore;
use crate::progress::CLProgressBar;

/// ProNE
///
/// Compute embeddings for a given graph represented by `mat`.
///
/// * `A` - the adjacency graph in CSR format
/// * `dim` - dimension of embeddings to compute
/// * `n_samples` - how many samples to draw for the rSVD approximation
/// * `n_subspace_iters` - how many power iterations to perform when computing rSVD
/// * `order` - is how many terms to compute for the Chebyshev approximation of the Laplacian filter
/// * `mu` - scaling factor used in spectral propagation
/// * `theta` - term used in spectral propagation (recommendation is to choose a value close to 0)
/// * Returns an `EmbeddingStore`
pub fn get_prone_embeddings(
    A: &CsrMatrix<f32>,
    dim: usize,
    n_samples: Option<usize>,
    n_subspace_iters: Option<usize>,
    order: usize,
    mu: f32,
    theta: f32,
    seed: u64
) -> EmbeddingStore {
    
    let pb = CLProgressBar::new(80, true);
    pb.update_message(|msg| { msg.clear(); write!(msg, "Stage 1/3").expect("Error writing out indicator message!"); });
    
    let mat = pre_factorization(A);
    pb.inc(10);
    pb.update_message(|msg| { msg.clear(); write!(msg, "Stage 2/3").expect("Error writing out indicator message!");});
    
    let initial_embeddings = get_rsvd_embeddings(&mat, dim, n_samples, n_subspace_iters, seed);
    pb.inc(70);
    pb.update_message(|msg| { msg.clear(); write!(msg, "Stage 3/3").expect("Error writing out indicator message!"); });

    let spectral_embeddings = spectral_propagate(&mat, initial_embeddings, order, mu, theta);
    pb.inc(20);
    let mut es = EmbeddingStore::new(mat.nrows(), dim, Distance::Cosine);
    spectral_embeddings.row_iter()
        .into_iter()
        .enumerate()
        .for_each(|(i, row)| {
            es.set_embedding(i, row.clone_owned().data.as_vec());
        });

    return es;
}

fn pre_factorization(A: &CsrMatrix<f32>) -> CsrMatrix<f32> {
    let l1 = 0.75;
    let mut C1 = l1_normalize(A);
    let mut neg = csr_column_sum(&C1);
    neg.iter_mut().for_each(|x| *x = x.powf(l1));
    let d: f32 = neg.iter().sum();
    neg.iter_mut().for_each(|x| *x /= d);

    let N = diag(neg);
    let mut Neg = A * N;

    let c1_data = C1.csr_data_mut().2;
    let n_data = Neg.csr_data_mut().2;

    c1_data.iter_mut().for_each(|x| {
        *x = if *x > 0.0 { x.ln() } else { 0.0 };
    });
    n_data.iter_mut().for_each(|x| {
        *x = if *x > 0.0 { x.ln() } else { 0.0 };
    });

    let F = C1 - Neg;
    return F;
}

/// Get L2-normalized embeddings via rSVD
fn get_rsvd_embeddings(
    A: &CsrMatrix<f32>,
    dim: usize,
    n_samples: Option<usize>,
    n_subspace_iters: Option<usize>,
    seed: u64
) -> DMatrix<f32> {

    let mut rng = XorShiftRng::seed_from_u64(seed);

    let (mut U, mut S) = rsvd(A, dim, n_samples, n_subspace_iters, &mut rng);

    // sqrt(S)
    for e in S.iter_mut() {
        *e = e.clone().sqrt();
    }
    let S = S.transpose();
    // compute row-wise product with S.T
    // L-2 normalize each row
    U.row_iter_mut()
     .for_each(|mut row| {
         row.component_mul_assign(&S);
         row.normalize_mut();
      });
    return U;
}

/// Propagate embeddings through spectrally modulated space
/// Incorporates local and global graph features in the initial embeddings
/// and returns a new set of embeddings of the same dimensions
pub fn spectral_propagate(
    A: &CsrMatrix<f32>,
    embeddings: DMatrix<f32>,
    order: usize,
    mu: f32,
    s: f32,
) -> DMatrix<f32> {
    if order == 1 {
        return embeddings;
    }
    let node_number = A.nrows();
    let D = speye(node_number);
    let A = &D + A;
    let DA = l1_normalize(&A);
    let L = &D - DA;
    let M = L - mu * D;

    let mut Lx0 = embeddings.clone();
    let mut Lx1 = &M * &Lx0;
    Lx1 = 0.5 * (&M * Lx1) - &Lx0;

    let mut conv = i0(s) * &Lx0;
    conv -= 2.0 * i1(s) * &Lx1;

    for i in 2..order {
        let mut Lx2 = &M * &Lx1;
        Lx2 = ((&M * Lx2) - 2.0 * &Lx1) - &Lx0;
        if i % 2 == 0 {
            conv += 2.0 * iv(i as f32, s) * &Lx2;
        } else {
            conv -= 2.0 * iv(i as f32, s) * &Lx2;
        }
        Lx0 = Lx1;
        Lx1 = Lx2;
    }
    
    let dim = embeddings.ncols();
    let mm = A * (embeddings - conv);
    let emb = get_embedding_dense(mm, dim);
    return emb;
}

/// ====================== Utilities ========================//

/// Modified Bessel function of the first kind, real order 0 
fn i0(v: f32) -> f32 {
    v.besseli(0.0)
}

/// Modified Bessel function of the first kind, real order 1
fn i1(v: f32) -> f32 {
    v.besseli(1.0)
}

/// Modified Bessel function of the first kind, real order n
fn iv(n: f32, v: f32) -> f32 {
    v.besseli(n)
}

/// Compute L2-normalized embeddings via SVD
fn get_embedding_dense(A: DMatrix<f32>, dim: usize) -> DMatrix<f32> {
    let SVD {
        u, singular_values, ..
    } = A.svd(true, false);

    let mut U = u.unwrap();
    U = U.index((.., ..dim)).into();
    let mut S: DVector<f32> = singular_values.index((..dim, ..)).into();
    // sqrt(S)
    for e in S.iter_mut() {
        *e = e.clone().sqrt();
    }
    let S = S.transpose();
    // L2-normalize, compute row-wise product with S.T
    U.row_iter_mut()
    .for_each(|mut row| {
        row.component_mul_assign(&S);
        row.normalize_mut();
    });
    return U;
}

/// Create a diagonal matrix from the given data
/// Ignores zero values
fn diag(diag_data: Vec<f32>) -> CsrMatrix<f32> {

    let n = diag_data.len();
    let mut indptr = vec![0; n+1];
    let mut indices = Vec::with_capacity(n);
    let mut data = Vec::with_capacity(n);
    let mut nnz = 0;
    for i in 0..n {
        if diag_data[i] != 0.0 {
            indices.push(i);
            data.push(diag_data[i]);
            nnz += 1;
        }
        indptr[i+1] = nnz;
    }

    indices.shrink_to_fit();
    data.shrink_to_fit();

    CsrMatrix::try_from_csr_data(
        n, n,
        indptr,
        indices,
        data
    ).unwrap()
}

/// L1-normalize CSR data in-place
fn l1_normalize_mut(
    indptr: &[usize],
    data: &mut [f32],
    nrows: usize,
) {
    for i in 0..nrows {
        let mut sum = 0.0;
        for j in indptr[i]..indptr[i + 1] {
            sum += data[j].abs();
        }
        if sum == 0.0 { continue; }
        for j in indptr[i]..indptr[i + 1] {
            data[j] /= sum;
        };
    }
}

/// L1-normalize a sparse CSR Matrix
/// Returns a new Matrix
fn l1_normalize(A: &CsrMatrix<f32>) -> CsrMatrix<f32> {
    let mut B = A.clone();
    let nrows = B.nrows();
    let (indptr, _, data) = B.csr_data_mut();
    l1_normalize_mut(indptr, data, nrows);
    return B;
}

/// Returns nxn Identity matrix, in CSR format
fn speye(n: usize) -> CsrMatrix<f32> {
    let indptr = (0..=n).collect();
    let indices = (0..n).collect();
    let data = vec![1.0; n];
    CsrMatrix::try_from_csr_data(
        n,n,
        indptr,
        indices,
        data,
    ).unwrap()
}

/// Compute the column-wise sum of the matrix
/// Returns a Vec containing the sum of each column
fn csr_column_sum(A: &CsrMatrix<f32>) -> Vec<f32> {
    let mut sum = vec![0.0; A.ncols()];
    let n = A.nrows();
    let (indptr, indices, data) = A.csr_data();
    for i in 0..n {
        let (s,e) = (indptr[i], indptr[i+1]);
        for j in s..e {
            sum[indices[j]] += data[j];
        }
    }
    return sum;
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn get_csr() -> CsrMatrix<f32> {
        let indptr = vec![ 0 ,2, 3, 6, 7, 8, 9, 10];
        let indices = vec![1,2,0,0,3,4,2,2,6,5];
        let data = vec![0.5, 1.,1.,0.33333334, 0.66666669, 1., 1., 1.,1.,1.,];
        let A = CsrMatrix::try_from_csr_data(
            7,7,
            indptr,
            indices,
            data,
        ).unwrap();
        return A;
    }

    #[test]
    fn test_pre_factorization() {
        let A = get_csr();
        let F = pre_factorization(&A);
        let (indptr, indices, data) = F.csr_data();

        assert_eq!(&indptr, &[0, 2, 3, 6, 7, 8, 9, 10]);
        assert_eq!(&indices, &[1, 2, 0, 0, 3, 4, 2, 2, 6, 5]);
        assert_eq!(
            *data,
            [2.3178108, 0.7582297, 1.7837036, 1.0905564, 2.0301285, 1.7260299, 1.1636947, 1.1636947, 1.8993165, 1.8993165],
        );
    }

    #[test]
    fn test_spectral_propagate() {
        let A = get_csr();
        let E = DMatrix::from_row_slice(
            7, 3,
            &[-1.0, -1.5000466e-9, -8.9961766e-8,
              -6.387856e-8, -1.0, -4.4688345e-6,
              1.721193e-7, -1.0, 1.0590724e-6,
              -1.0, -1.217417e-7, 1.5562911e-8,
              -1.0, -1.0825446e-7, 1.12080976e-7,
              -6.3385265e-8, 2.2471597e-6,
              -1.0, 4.0365595e-9, 4.63808e-7,
              1.0
            ]);

        let spectre = spectral_propagate(&A, E, 10, 0.2, 0.5);

        let expected = DMatrix::from_column_slice(7,3,
            &[0.9087086, 0.9305056, 0.91668296,
              0.99904305, 0.9990429, -0.29472774,
              -0.29472774, -0.4174307, -0.36627328,
              0.39961514, 0.043729097, 0.043729093,
              0.95557725, 0.95557725, -0.00056542596,
              0.0018052289, 0.00036060097, -0.0008914651,
              -0.0009595304, 0.0027597758, 0.0027597758,
            ],
        );
        assert_relative_eq!(spectre, expected, epsilon=1e-5);
    }
}
