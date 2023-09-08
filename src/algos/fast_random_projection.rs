//! Fast Random Projection
//!
//! This method works by:
//!   1) constructing a node similarity matrix
//!   2) very sparse random projection for dimension reduction

use nalgebra::DMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use rand::prelude::*;
use rand::Rng;
use rand_distr::Binomial;
use nalgebra_sparse::convert::serial::convert_csr_dense;
use rand_xorshift::XorShiftRng;

use crate::embeddings::EmbeddingStore;
use crate::embeddings::Distance;
use crate::progress::CLProgressBar;
use std::fmt::Write;

/// Fast Random Projection
///
/// Compute an embedding per-row of A.
/// * `A` - the adjacency matrix in CSR format
/// * `dims` - dimension of embeddings
/// * `weights` - factor used for weighting of transition matrix power k
/// * `norm_powers` - if true L2-normalizes matrix powers
/// * `seed` - a seed for the random number generator
pub fn get_fastrp_embeddings(
    A: &CsrMatrix<f32>,
    dims: usize,
    weights: &[f32],
    norm_powers: bool,
    seed: u64,
) -> EmbeddingStore {
    let mut rng = XorShiftRng::seed_from_u64(seed);
    let k = weights.len();

    let pb = CLProgressBar::new(100, true);
    pb.update_message(|msg| { msg.clear(); write!(msg, "Stage 1/2").expect("Error writing out indicator message!"); });

    let ulist = fast_random_projection(
        A, dims, k, &mut rng
    );
    pb.inc(20);
    
    pb.update_message(|msg| { msg.clear(); write!(msg, "Stage 2/2").expect("Error writing out indicator message!"); });
    let embeddings = merge_matrices(ulist, weights, norm_powers);
    pb.inc(80);
    let mut es = EmbeddingStore::new(A.nrows(), dims, Distance::Cosine);
    embeddings.row_iter()
    .into_iter()
    .enumerate()
    .for_each(|(i,row)| {
        es.set_embedding(i, row.clone_owned().data.as_vec());
    });
    return es;
}

// Computes a list of powers of the transition matrix of A
fn fast_random_projection(
    A: &CsrMatrix<f32>,
    dim: usize,
    k: usize,
    rng: &mut impl Rng
) -> Vec<DMatrix<f32>> {

    let mut a_sum = csr_row_sum(&A);
    a_sum.iter_mut().for_each(|x| *x = 1.0 / *x);
    let normalizer = diag(a_sum);

    // the graph transition matrix
    let M = normalizer * A;

    // a projection matrix for reducing dimensionality
    let projection_matrix = get_sparse_random_transform(
        A.nrows(),
        A.ncols(),
        Some(dim),
        None,
        None,
        rng
    );

    // powers of the transition matrix by chain matrix multiplication
    let initial = convert_csr_dense(&(&M * projection_matrix.transpose()));
    let mut ulist = vec![initial];
    for _ in 2..(k+1) {
        let prev = &ulist[ulist.len()-1];
        ulist.push(&M * prev);
    }
    return ulist;
}

// linearly combine a list of matrices using the given weights
// optionally L2-normalizes them before merging
fn merge_matrices(mut ulist: Vec<DMatrix<f32>>, weights: &[f32], norm_powers: bool) -> DMatrix<f32> {
    if norm_powers {
        ulist.iter_mut().for_each(|u| l2_normalize(u));
    }
    let (m,n) = (ulist[0].nrows(), ulist[0].ncols());
    let mut embeddings = DMatrix::zeros(m, n);
    for i in 0..ulist.len() {
        embeddings += &ulist[i] * weights[i];
    }
    return embeddings;
}

/// minimum dimension for embedding by Johnson–Lindenstrauss lemma
fn min_dim(n_samples: usize, eps: f32) -> usize {
    let denom = ((eps*eps) / 2.0) - ((eps*eps*eps) / 3.0);
    return (4.0 * (n_samples as f32).ln() / denom) as usize;
}

/// Computes a generalized Achlioptas random sparse matrix for random projection
/// of shape (n_components, n_features)
/// if n_components is None, automatically computes a value by Johnson-Lindenstrauss lemma
/// density must be between 0 and 1
fn get_sparse_random_transform(
    n_samples: usize,
    n_features: usize,
    n_components: Option<usize>,
    density: Option<f32>,  // mut be 0 < density < 1
    eps: Option<f32>,
    rng: &mut impl Rng,
) -> CsrMatrix<f32> {
    let eps = eps.unwrap_or(0.1);
    let n_components = match n_components {
        Some(n) => n,
        None => min_dim(n_samples, eps),
    };
    let density = match density {
        Some(d) => d,
        None => 1.0/(n_features as f32).sqrt()
    };
    let transform = sparse_random_matrix(n_components, n_features, density, rng);
    return transform;
}

/// construct a CSR random matrix
fn sparse_random_matrix(
    nrows: usize,
    ncols: usize,
    density: f32,  // must be 0 < density < 1
    mut rng: &mut impl Rng
) -> CsrMatrix<f32> {

    let mut indices = vec![];
    let mut indptr: Vec<usize> = vec![0; nrows+1];
    let mut offset: usize = 0;
    let mut index_pool: Vec<usize> = (0..ncols).collect();

    // distribution representing row non-zero counts
    let nonzero_dist = Binomial::new(ncols as u64, density as f64).unwrap();

    for row in 0..nrows {
        // generate the indices of the non-zero components for row i
        // by sampling from 0..n_features without replacement
        let n_nonzero_i = nonzero_dist.sample(rng) as usize;
        index_pool.shuffle(&mut rng);
        indices.extend_from_slice(&index_pool[0..n_nonzero_i]);
        // enforce sorting of column indices for canonical CSR 
        indices[offset..].sort_unstable();
        offset += n_nonzero_i;
        indptr[row+1] = offset;
    }

    let val = (1.0 / density).sqrt() / (nrows as f32).sqrt();
    let mut data: Vec<f32> = vec![0.0; indices.len()];
    for i in 0..data.len() {
        let p = rng.gen::<f32>();
        data[i] = if p > 0.5 { val } else { -val };
    }

    CsrMatrix::try_from_csr_data(
        nrows, ncols,
        indptr,
        indices,
        data,
    ).unwrap()
}

// ============= Matrix utilities ================ //

/// compute the sum of each row of the matrix
fn csr_row_sum(
    A: &CsrMatrix<f32>,
) -> Vec<f32> {
    let n = A.nrows();
    let (indptr, _, data) = A.csr_data();
    let mut out = vec![0.0; n];
    for i in 0..n {
        let (s,e) = (indptr[i], indptr[i+1]);
        out[i] = data[s..e].iter().sum();
    }
    return out;
}

/// create an NxN diagonal matrix from the given Vec
/// ignores 0 values
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

// row-wise l2-normalization of matrix
fn l2_normalize(
    data: &mut DMatrix<f32>,
) {
    for mut row in data.row_iter_mut() {
        let sum = row.sum();
        if sum == 0.0 {
            continue;
        }
        row.normalize_mut();
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    use super::*;

    fn get_csr() -> CsrMatrix<f32> {
        let data = vec![0.5, 1.,1.,0.33333334, 0.66666669, 1., 1., 1.,1.,1.,];
        let indices = vec![1,2,0,0,3,4,2,2,6,5];
        let indptr = vec![ 0 ,2, 3, 6, 7, 8, 9, 10];    
        let A = CsrMatrix::try_from_csr_data(
            7,7,
            indptr,
            indices,
            data,
        ).unwrap();
        return A;
    }

    #[test]
    fn test_get_sparse_random_transform() {
        let mut rng = XorShiftRng::seed_from_u64(1337);
        let mat = sparse_random_matrix(
            10,
            100,
            0.01,  // must be 0 < density < 1
            &mut rng
        );

        let expected = [
            3.1622777, -3.1622777, 3.1622777, -3.1622777,
            -3.1622777, -3.1622777, -3.1622777, -3.1622777,
            3.1622777, -3.1622777, 3.1622777, -3.1622777,
            -3.1622777, -3.1622777
        ];

        let (_, _, data) = mat.csr_data();
        assert_eq!(mat.nrows(), 10);
        assert_eq!(mat.ncols(), 100);
        assert_eq!(data, &expected);
    }

    #[test]
    fn test_fast_random_projection() {
        let mut rng = XorShiftRng::seed_from_u64(1337);
        let A = get_csr();
        let ulist = fast_random_projection(&A, 3, 3, &mut rng);
        assert_eq!(ulist.len(), 3);
    }

    #[test]
    fn test_merge_matrices() {
        let mut rng = XorShiftRng::seed_from_u64(1337);
        let A = get_csr();
        let weights = [0.333, 0.333, 0.333];
        let ulist = fast_random_projection(&A, 3, 3, &mut rng);
        let actual = merge_matrices(ulist, &weights, true);
        let expected = DMatrix::from_column_slice(7, 3, &[
            0.1006749, 0.043618284, -0.2715608,
            -0.1293677, -0.1293677, -0.6631907,
            -0.6199818, -0.31000823, 0.3510673,
            0.26653367, -0.36758113, -0.36758113,
            -0.6631907, -0.6199818, 0.0,
            0.0, 0.0, 0.0,
            0.0, -0.19225764, -0.3845153
        ]);
        assert_eq!(actual.nrows(), 7);
        assert_eq!(actual.ncols(), 3);
        assert_eq!(actual, expected);
    }
}