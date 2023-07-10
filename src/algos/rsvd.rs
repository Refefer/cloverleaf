#![allow(non_snake_case)]

extern crate nalgebra as na;

use na::{DMatrix, DMatrixViewMut, DVector};
use na::SVD;
use rand::prelude::*;
use rand::thread_rng;
use rand::Rng;
use rand_distr::StandardNormal;
use rand_xorshift::XorShiftRng;

use crate::graph::CSR;
use crate::embeddings::Distance;
use crate::embeddings::EmbeddingStore;


/// Compute embeddings using randomized-svd
/// normalizes all vectors to L-2
///
/// This method works well for very large matrices, and low dim (<10)
/// can be computed in seconds.
///
/// Peak memory usage is at least (2 x dim x |V|)
///
/// A is a square symmetric sparse matrix in CSR format
/// d is the size of the embeddings you want (smaller is faster to compute)
/// n_samples is how many samples to take for the initial gaussian matrix range estimate (default is 2*rank)
/// n_subspace_iters is how many times to loop when calculating the approximate range
/// Users should prefer taking more samples to performing more loop iterations
pub fn rand_embeddings(
    A: &CSR,
    dim: usize,
    n_samples: Option<usize>,
    n_subspace_iters: Option<usize>,
    seed: u64,
) -> EmbeddingStore
{
    let mut rng = XorShiftRng::seed_from_u64(seed);

    let (mut U, mut S) = rsvd(A, dim, n_samples, n_subspace_iters, &mut rng);

    // sqrt(S)
    for e in S.iter_mut() {
        *e = e.clone().sqrt();
    }
    let S = S.transpose();

    let (nrows, ncols) = U.shape();

    let mut es = EmbeddingStore::new(nrows, dim, Distance::Cosine);

    U.row_iter_mut()
     .into_iter()
     .enumerate()
     .for_each(|(i, mut row)| {
        // compute row-wise product with S.T
        // L-2 normalize each row
         row.component_mul_assign(&S);
         row.normalize_mut();
         // insert into embedding store
         es.set_embedding(i, row.clone_owned().data.as_vec());
      });

    return es;
}

/// Randomized Truncated SVD for large sparse matrices.
/// Based on [Halko et al. (2009)](https://arxiv.org/abs/0909.4061)
/// We make a strong assumption that the matrix is square and symmetric,
/// which lets us skip transposition of A.
///
/// Users should prefer increasing sample number to increasing iteration number.
///
pub fn rsvd<R>(
    A: &CSR,
    rank: usize,
    n_oversamples: Option<usize>,
    n_subspace_iters: Option<usize>,
    mut rng: &mut R,
) -> (DMatrix<f32>, DVector<f32>)
where
    R: Rng,
{
    // A must be square symmetric
    assert!(A.nrows == A.ncols);

    let n_samples = match n_oversamples {
        None => 2 * rank,
        Some(n) => rank + n,
    };

    let Q = find_range(A, n_samples, n_subspace_iters, &mut rng);

    // B = Q.T * A = (A.T * Q).T = (A * Q).T
    let Btmp = csr_dense_matmul(&A, &Q);
    let B = Btmp.transpose();
    drop(Btmp);

    // svd complexity is  O(m^2n + n^3)
    let SVD {
        u, singular_values, ..
    } = B.svd(true, true);

    let u_tilde = u.unwrap();
    let U_tmp = Q * u_tilde;

    // truncate
    let U = U_tmp.index((.., ..rank));
    let S = singular_values.index((..rank, ..));

    (U.into(), S.into())
}

/// Given a matrix A and a number of samples, computes an orthonormal matrix
/// that approximates the range of A.
///
/// We assume that A is square symmetric
///
/// n_samples controls how much to oversample the random gaussian matrix
/// n_subspace_iters controls how many loops
/// peak memory consumption is O(2 * N * n_samples) where N is column number of A
fn find_range<R>(
    A: &CSR,
    n_samples: usize,
    n_subspace_iters: Option<usize>,
    mut rng: &mut R,
) -> DMatrix<f32>
where
    R: Rng,
{
    let (_, N) = A.shape();
    let mut Y = DMatrix::zeros(N, n_samples);
    let O = DMatrix::from_fn(N, n_samples, |_, _| StandardNormal.sample(&mut rng));
    // Y = A*O
    csr_dense_matmul_into(&A, &O, &mut Y);
    drop(O);
    match n_subspace_iters {
        Some(iters) => subspace_iter(&A, &mut Y, iters),
        None => ortho_basis(Y),
    }
}

/// Numerically stable subspace iteration
/// Specialized to square symmetric matrices
fn subspace_iter(
    A: &CSR,               // a square symmetric matrix
    Y0: &mut DMatrix<f32>, // initial estimate of Q
    n_iters: usize,        // number of iterations to perform
) -> DMatrix<f32> {
    let mut Q = ortho_basis2(view_mut(Y0));
    for _ in 0..n_iters {
        /*
         * For each iteration of this loop we compute:
         *  Z = A.T*Q = A*Q  (A is square symmetric)
         *  Q = A*Z
         * reusing the Y0 buffer when possible
         */
        clear(Y0);
        // A.T*Q = A*Q
        csr_dense_matmul_into(A, &Q, Y0);
        // Z = ortho(A*Q)
        let Z = ortho_basis2(view_mut(Y0));
        clear(Y0);
        csr_dense_matmul_into(A, &Z, Y0);
        drop(Z);
        // Q = ortho(A*Z)
        Q = ortho_basis2(view_mut(Y0));
    }
    return Q;
}

/// computes orthonormal basis of matrix, and consumes it
fn ortho_basis(M: DMatrix<f32>) -> DMatrix<f32> {
    let qr = M.qr();
    qr.q()
}

/// computes orthonormal basis of matrix view, and consumes it
fn ortho_basis2(M: DMatrixViewMut<f32>) -> DMatrix<f32> {
    let qr = M.qr();
    qr.q()
}

/// C += A*B
/// C should be buffer size A.nrows * B.ncols
fn csr_dense_matmul_into(A: &CSR, B: &DMatrix<f32>, C: &mut DMatrix<f32>) {
    let (M, P) = A.shape();
    let (_, N) = B.shape();

    let Ad = &A.weights;
    let Ap = &A.rows;
    let Ai = &A.columns;
    let Bd = B.as_slice();
    let Cd = C.as_mut_slice();

    for j in 0..N {
        let b_off = j * P;
        let c_off = j * M;
        for l in 0..P {
            let bi = b_off + l;
            for k in Ap[l]..Ap[l + 1] {
                let i = c_off + Ai[k];
                Cd[i] = Ad[k].mul_add(Bd[bi], Cd[i]);
            }
        }
    }
}

/// C = A*B
/// C is buffer size A.nrows * B.ncols
/// Output will be written to C
fn csr_dense_matmul(A: &CSR, B: &DMatrix<f32>) -> DMatrix<f32> {
    let (M, P) = A.shape();
    let (_, N) = B.shape();
    let mut C = DMatrix::zeros(M, N);
    csr_dense_matmul_into(A, B, &mut C);
    return C;
}

/// returns mutable view of the whole matrix
fn view_mut(M: &mut DMatrix<f32>) -> DMatrixViewMut<f32> {
    M.view_mut((0, 0), M.shape())
}

/// sets all matrix entries to 0
fn clear(M: &mut DMatrix<f32>) {
    M.fill(0.0);
}
