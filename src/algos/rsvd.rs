//! Randomized Singular Value Decomposition

use nalgebra::{DMatrix, DVector, SVD};
use nalgebra_sparse::csr::CsrMatrix;
use rand::prelude::*;
use rand_distr::StandardNormal;

/// Randomized SVD
/// 
/// As seen in (p. 227 of Halko et al).
///
/// * `A` - (m x n) matrix.
/// * `rank` - Desired rank approximation.
/// * `n_oversamples` - Oversampling parameter for Gaussian random samples.
/// * `n_subspace_iters` - Number of power iterations.
/// * return  U, S as in truncated SVD.
pub(crate) fn rsvd(
    A: &CsrMatrix<f32>,
    rank: usize,
    n_oversamples: Option<usize>,
    n_subspace_iters: Option<usize>,
    mut rng: &mut impl Rng,
) -> (DMatrix<f32>, DVector<f32>) {

    let n_samples = match n_oversamples {
        None => 2 * rank,
        Some(n) => rank + n,
    };

    let Q = find_range(A, n_samples, n_subspace_iters, &mut rng);
    let B = mul_AT_by_B(&Q, A);

    let SVD {
        u, singular_values, ..
    } = B.svd(true, false);

    let u_tilde = u.unwrap();
    let U_tmp = Q * u_tilde;
    
    // truncate
    let U = U_tmp.index((.., ..rank));
    let S = singular_values.index((..rank, ..));

    (U.into(), S.into())
}

/// Given a matrix A and a number of samples,
/// computes an orthonormal matrix that approximates the range of A.
fn find_range(
    A: &CsrMatrix<f32>,
    n_samples: usize,
    n_subspace_iters: Option<usize>,
    mut rng: &mut impl Rng,
) -> DMatrix<f32> {

    let N = A.ncols();
    let O = DMatrix::from_fn(N, n_samples, |_, _| StandardNormal.sample(&mut rng));
    let Y = A * O;
    match n_subspace_iters {
        Some(iters) => subspace_iter(&A, Y, iters),
        None => ortho_basis(Y),
    }
}

/// Computes orthonormal basis of matrix M
fn ortho_basis(M: DMatrix<f32>) -> DMatrix<f32> {
    let qr = M.qr();
    qr.q()
}

/// Computes orthonormalized approximate range of A
/// after power iterations.
fn subspace_iter(A: &CsrMatrix<f32>, Y0: DMatrix<f32>, n_iters: usize) -> DMatrix<f32> {
    let mut Q = ortho_basis(Y0);
    for _ in 0..n_iters {
        let Z = ortho_basis(A.transpose() * &Q);
        Q = ortho_basis(A * Z);
    }
    return Q;
}

/// dense-sparse product
/// multiplies a dense row-major matrix by a compresed sparse row matrix
/// Writes output in Column-Major order
fn dnsrow_csr_matmul(
    a_nrows: usize,
    b_nrows: usize,
    a_data: &[f32],
    b_data: &[f32],
    b_indptr: &[usize],
    b_indices: &[usize],
    c: &mut [f32],
) {
    for i in 0..a_nrows {
        for j in 0..b_nrows {
            for k in b_indptr[j]..b_indptr[j + 1] {
                let l = b_indices[k];
                c[l * a_nrows + i] += a_data[i * b_nrows + j] * b_data[k];
            }
        }
    }
}

/// Compute A.T * B
/// Allocates a new matrix in column-major order 
fn mul_AT_by_B(A: &DMatrix<f32>, B: &CsrMatrix<f32>) -> DMatrix<f32> {
    // Since A is in column-major form
    // simply reinterpret the data to be
    // in row-major form by swapping axes
    let a_nrows = A.ncols();
    let b_nrows = B.nrows();
    let b_ncols = B.ncols();
    let (b_indptr, b_indices, b_data) = B.csr_data();
    let mut c_data = vec![0.0; a_nrows*b_ncols];
    dnsrow_csr_matmul(a_nrows, b_nrows, A.as_slice(), b_data, b_indptr, b_indices, &mut c_data);
    DMatrix::from_vec(a_nrows, b_ncols, c_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand_xorshift::XorShiftRng;

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
    fn test_rsvd() {
        let A = get_csr();
        let mut rng = XorShiftRng::seed_from_u64(1337);
        let (U, S) = rsvd(&A, 3, Some(3), Some(1), &mut rng);
        let u_expected = DMatrix::from_column_slice(
            7, 3,
            &[0.6097918, 7.7831737e-7, -4.945517e-8,
              0.56042546, 0.56042546, -3.7524546e-7,
              -4.8548575e-8, -4.5747225e-8, 0.42415532,
              0.9055893, -2.5126192e-7, -1.0671046e-7,
              -1.2517704e-7, 7.8956566e-7, 4.0070574e-8,
              4.4752966e-7, 8.698171e-8, -9.680764e-8,
              -1.6516456e-7, -0.66398156, -0.74774903]
        );
        let s_expected = DVector::from_row_slice(&[1.75729548, 1.30831209, 1.00000001]);

        assert_relative_eq!(U, u_expected, epsilon=1e-5);
        assert_relative_eq!(S, s_expected, epsilon=1e-5);
    }


    #[test]
    fn test_find_range() {
        let A = get_csr();
        let mut rng = XorShiftRng::seed_from_u64(1337);
        let Q = find_range(&A, 6, Some(1), &mut rng);
        let expected = DMatrix::from_column_slice(
            7, 6,
            &[  0.14272499, 0.44805098, 0.2673037, 0.13724963, 0.13724963, -0.6995637,
                0.42469338, 0.5310906, -0.15537736, 0.4018808, 0.470897, 0.470897,
                0.26774397, -0.13083324, 0.08768566, -0.6213763, -0.1978631, 0.033032026,
                0.033032026, 0.013470635, 0.7514579, 0.25793642, -0.3626468, -0.47192055,
                0.10225809, 0.10225797, -0.5945787, -0.45255947, 0.011312813, -0.45840725,
                0.69421583, -0.3286546, -0.3286544, -0.25462615, -0.16412558, -0.7894455,
                -0.2175996, 0.1524669, 0.37397456, 0.37397438, -0.14277324, -0.0779866,
            ],
        );
        assert_relative_eq!(Q, expected, epsilon=1e-6);
    }

}