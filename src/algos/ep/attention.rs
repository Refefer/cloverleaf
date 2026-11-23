//! This module constructs a node embedding from a set of features through various different
//! attention methods.  Notably, we use averaged versus concat -> linear projection due to
//! performance cost.  This has representational downsides but performs significantly faster and
//! with no extra parmeters.
use crate::autograd::*;
use float_ord::FloatOrd;
use rand::prelude::*;

use crate::algos::utils::Sample;
use crate::error::Error;

/// Attention pooling's weights: multi-head self-attention over a node's feature embeddings with
/// query, key, value and output projections shared by every feature, as the GPU trainer's
/// `AttentionPool` (gpu/ep_gpu/model.py).  A node's embedding is
///
///   h = wmean(x) + out · wmean(ctx),   ctx_i = concat_h softmax_j(q_i·k_j / sqrt(hd)) v_j
///
/// where q, k and v are each token's projections through `qkv`, wmean is the token-weighted
/// mean, and the softmax runs over every token with a positive weight.  Token weights don't
/// enter the logits.  With `out` zero this is mean pooling.
///
/// The projections are in PyTorch's `Linear` layout, row-major: `qkv` is
/// (3 · heads · head_dims) × dims, its rows the query heads, then the key heads, then the value
/// heads; `out` is dims × (heads · head_dims).  Only `new` builds one, so the shapes always
/// agree.
#[derive(Clone, Debug, PartialEq)]
pub struct AttentionWeights {
    dims: usize,
    heads: usize,
    head_dims: usize,
    qkv: Vec<f32>,
    out: Vec<f32>
}

impl AttentionWeights {
    pub fn new(dims: usize, heads: usize, head_dims: usize, qkv: Vec<f32>, out: Vec<f32>) -> Result<Self, Error> {
        if dims == 0 || heads == 0 || head_dims == 0 {
            return Err(Error::InvalidArgument(format!(
                "attention needs dims, heads and head_dims > 0, got {}, {} and {}", dims, heads, head_dims)))
        }
        let inner = heads * head_dims;
        if qkv.len() != 3 * inner * dims {
            return Err(Error::InvalidArgument(format!(
                "qkv has {} values, expected {} ({} x {})", qkv.len(), 3 * inner * dims, 3 * inner, dims)))
        }
        if out.len() != dims * inner {
            return Err(Error::InvalidArgument(format!(
                "out has {} values, expected {} ({} x {})", out.len(), dims * inner, dims, inner)))
        }
        if !qkv.iter().chain(out.iter()).all(|v| v.is_finite()) {
            return Err(Error::InvalidArgument("attention weights must be finite".into()))
        }
        Ok(AttentionWeights { dims, heads, head_dims, qkv, out })
    }

    /// Each head's width for `dims`-wide embeddings: dims / heads rounded up to a multiple of 8,
    /// the GPU trainer's rule (its fused attention kernels need it), so weights trained by
    /// either have the same shapes.
    pub fn head_dims_for(dims: usize, heads: usize) -> usize {
        dims.div_ceil(heads.max(1)).div_ceil(8).max(1) * 8
    }

    /// Fresh weights: `qkv` uniform in ±1/sqrt(dims), PyTorch's `Linear` default, and `out` zero,
    /// so training starts from mean pooling.
    pub fn init(dims: usize, heads: usize, rng: &mut impl Rng) -> Result<Self, Error> {
        let head_dims = Self::head_dims_for(dims, heads);
        let inner = heads * head_dims;
        let bound = 1f32 / (dims.max(1) as f32).sqrt();
        let qkv = (0..3 * inner * dims).map(|_| bound * (2f32 * rng.gen::<f32>() - 1f32)).collect();
        AttentionWeights::new(dims, heads, head_dims, qkv, vec![0f32; dims * inner])
    }

    pub fn dims(&self) -> usize { self.dims }
    pub fn heads(&self) -> usize { self.heads }
    pub fn head_dims(&self) -> usize { self.head_dims }
    pub fn qkv(&self) -> &[f32] { &self.qkv }
    pub fn out(&self) -> &[f32] { &self.out }

    fn inner(&self) -> usize {
        self.heads * self.head_dims
    }

    /// Pools tokens `x` (each `dims` wide) with weights `w` into `out`.  Tokens with weight 0 are
    /// ignored; with none left the result is zeros.
    pub fn pool(&self, x: &[&[f32]], w: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), w.len(), "one weight per token");
        assert_eq!(out.len(), self.dims, "output width");
        out.fill(0f32);
        let tokens: Vec<(&[f32], f32)> = x.iter().zip(w.iter())
            .filter(|(_, wi)| **wi > 0f32)
            .map(|(xi, wi)| { assert_eq!(xi.len(), self.dims, "token width"); (*xi, *wi) })
            .collect();
        let total: f32 = tokens.iter().map(|(_, wi)| wi).sum();
        if tokens.is_empty() || total <= 0f32 {
            return
        }

        // Each token's query, key and value projections: (T, 3 · inner).
        let (dims, inner, hd) = (self.dims, self.inner(), self.head_dims);
        let proj: Vec<Vec<f32>> = tokens.iter().map(|(xi, _)| {
            self.qkv.chunks_exact(dims).map(|row| dot(row, xi)).collect()
        }).collect();

        // The token-weighted mean of every token's attention context.
        let scale = 1f32 / (hd as f32).sqrt();
        let mut ctx_mean = vec![0f32; inner];
        let mut logits = vec![0f32; tokens.len()];
        for h in 0..self.heads {
            let (q0, k0, v0) = (h * hd, inner + h * hd, 2 * inner + h * hd);
            for (i, (_, wi)) in tokens.iter().enumerate() {
                let q = &proj[i][q0..q0 + hd];
                logits.iter_mut().zip(proj.iter()).for_each(|(l, pj)| *l = dot(q, &pj[k0..k0 + hd]) * scale);
                let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                logits.iter_mut().for_each(|l| *l = (*l - max).exp());
                let z: f32 = logits.iter().sum();
                let share = wi / total;
                for (pj, l) in proj.iter().zip(logits.iter()) {
                    let p = share * l / z;
                    ctx_mean[h * hd..(h + 1) * hd].iter_mut().zip(&pj[v0..v0 + hd]).for_each(|(c, v)| *c += p * v);
                }
            }
        }

        for (xi, wi) in tokens.iter() {
            out.iter_mut().zip(xi.iter()).for_each(|(o, v)| *o += wi / total * v);
        }
        out.iter_mut().zip(self.out.chunks_exact(inner)).for_each(|(o, row)| *o += dot(row, &ctx_mean));
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// The number of heads to use, and parmeters, for attention.
#[derive(Copy,Clone)]
pub struct MultiHeadedAttention {
    /// Number of attention heads
    pub num_heads: usize,

    /// Number of dimentions to use for both query and key vectors
    pub d_k: usize,

    /// Attention type
    pub attention_type: AttentionType
}

#[derive(Copy,Clone)]
pub enum AttentionType {
    /// Computes the full attention matrix of the feature step.  Incredibly expensive if the
    /// feature size is large: O(N^2)
    Full,

    /// Sliding attention, which looks at [i-window_size:i+window_size] for each attention matrix.  Quite a bit faster than Full, but can't attend to features beyond the window size: O(N * window_size * 2)
    Sliding { window_size: usize },

    /// Randomly selects num_features and computes full attention on it.  Can learn longer distance
    /// relationships due to random selection at the expense of longer train times.
    Random { num_features: Sample }
}

// Multi-headed attention is encoded as
// Q1,K1,Q2,K2,V1,V2
impl MultiHeadedAttention {
    pub fn new(num_heads: usize, d_k: usize, attention_type: AttentionType) -> Self {
        MultiHeadedAttention { num_heads, d_k, attention_type }
    }

    pub fn preserve_feature_order(&self) -> bool {
        matches!(self.attention_type,  AttentionType::Sliding {window_size:_})
    }

    fn get_query_vec(&self, emb: &ANode, head_num: usize) -> ANode {
        let start = self.d_k * head_num;
        emb.slice(start, self.d_k)
    }

    fn get_key_vec(&self, emb: &ANode, head_num: usize) -> ANode {
        let start = (self.num_heads * self.d_k) + self.d_k * head_num;
        emb.slice(start, self.d_k)
    }

    fn get_value_vec(&self, emb: &ANode, head_num: usize) -> ANode {
        let query_key_size = self.num_heads * self.d_k * 2;
        let v = emb.value().len();
        let d_model = ((v - query_key_size) as f32 / self.num_heads as f32) as usize;
        let start = query_key_size + d_model * head_num;
        emb.slice(start, d_model)
    }

}

/// Struct for easy access of the different pieces of the vector.
#[derive(Clone)]
struct Attention {
    query: ANode,
    key: ANode,
    value: ANode
}

impl Attention {
    fn new(node: &ANode, mha: &MultiHeadedAttention, head: usize) -> Self {
        let query = mha.get_query_vec(&node, head);
        let key = mha.get_key_vec(&node, head);
        let value = mha.get_value_vec(&node, head);
        Attention {query, key, value}
    }

}

/// The big chalupa: given attention and a set of vectors, computes the attention according to the
/// attention type within the MHA parameter.  
pub fn attention_mean<'a>(
    it: impl Iterator<Item=&'a (ANode, f32)>,
    mha: &MultiHeadedAttention,
    rng: &mut impl Rng
) -> ANode {

    let features = it.collect::<Vec<_>>();

    // A lone feature attends only to itself, so each head returns its value slice.
    if features.len() == 1 {
        let node = &features[0].0;
        let values = (0..mha.num_heads)
            .map(|head| mha.get_value_vec(node, head))
            .collect::<Vec<_>>();
        return values.sum_all() / (mha.num_heads as f32)
    }

    let mut averages = Vec::with_capacity(mha.num_heads);
    for head in 0..mha.num_heads {
        let items: Vec<_> = features.iter().map(|(node, count)| {
            (Attention::new(node, mha, head), *count)
        }).collect();

        // Compute attention matrix
        let attention_matrix = compute_attention_matrix(&items, &mha.attention_type, rng);
        
        let sm_att_mat = compute_attention_softmax(attention_matrix, mha.d_k);

        let n = items.len() as f32;
        let mean = scale_vecs(items, &sm_att_mat)
            .collect::<Vec<_>>().sum_all() / n;

        //averages.push(mean.tanh());
        averages.push(mean);
    }

    averages.sum_all() / (mha.num_heads as f32)
}

/// Computes value level attention scaling.
fn scale_vecs<'a>(
    items: Vec<(Attention, f32)>, 
    sm_att_mat: &'a AttentionMatrix 
) -> impl Iterator<Item=ANode> + 'a {

    let mut rows = vec![Vec::new(); sm_att_mat.len()];
    sm_att_mat.iter().enumerate().for_each(|(ri, row)| {
        items.iter().zip(row.iter()).for_each(|((att, _), scaled_v)| {
            if let Some(v) = scaled_v {
                rows[ri].push(&att.value * v);
            }
        });
    });

    rows.into_iter().map(|sums| sums.sum_all())
}

type AttentionMatrix = Vec<Vec<Option<ANode>>>;

#[inline]
fn compute_attention_matrix(
    items: &[(Attention, f32)],
    at: &AttentionType,
    rng: &mut impl Rng
) -> AttentionMatrix {
    match at {
        AttentionType::Full => compute_full_attention_matrix(items),
        AttentionType::Sliding{ window_size } => {
            compute_sliding_attention_matrix(items, *window_size)
        },
        AttentionType::Random { num_features } => {
            compute_random_attention_matrix(items, *num_features, rng)
        }
    }
}

// Computes the full N^2 attention matrix.  
fn compute_full_attention_matrix(
    items: &[(Attention, f32)]
) -> AttentionMatrix {
    
     // Get the attention for each feature
    let mut scaled = vec![vec![None; items.len()]; items.len()];
    for i in 0..items.len() {
        let (at_i, ic) = &items[i];
        let row = &mut scaled[i];
        for j in 0..items.len() {
            let (at_j, jc) = &items[j];
            let mut dot_i_j = (&at_i.query).dot(&at_j.key);
            let num = ic * jc;
            if num >= 1f32 {
                dot_i_j = dot_i_j * (num as f32);
            }
            row[j] = Some(dot_i_j);
        }
    }
    scaled
}

// Computes the sliding attention matrix
fn compute_sliding_attention_matrix(
    items: &[(Attention, f32)],
    window: usize
) -> AttentionMatrix {
    
     // Get the attention for each feature
    let mut scaled = vec![vec![None; items.len()]; items.len()];
    for i in 0..items.len() {
        let (j_start, j_end) = {
            let start = if window > i { 0 } else {i - window};
            let stop = (i + window+ 1).min(items.len());
            (start, stop)
        };

        let at_i = &items[i].0;
        let row = &mut scaled[i];
        for j in j_start..j_end {
            let at_j = &items[j].0;
            let dot_i_j = (&at_i.query).dot(&at_j.key);
            row[j] = Some(dot_i_j);
        }
    }
    scaled
}

// Computes the random attention matrix.  For each features, we select k random features to attend
// toward.
fn compute_random_attention_matrix(
    items: &[(Attention, f32)],
    sample: Sample,
    rng: &mut impl Rng
) -> AttentionMatrix {
    
    // Sample K features.  The softmax normalizes over whichever keys were sampled, so the logits
    // need no correction for the sampling rate.
    let (k, _) = sample.sample(items.len(), true, rng);

    // Get the attention for each feature
    let mut scaled = vec![vec![None; items.len()]; items.len()];
    let mut buff = vec![0; k];
    for i in 0..items.len() {
        let at_i = &items[i].0;
        let row = &mut scaled[i];
        items.iter().enumerate().map(|(i,_)| i).choose_multiple_fill(rng, buff.as_mut_slice());
        for j in buff.iter() {
            let at_j = &items[*j].0;
            let dot_i_j = (&at_i.query).dot(&at_j.key);
            row[*j] = Some(dot_i_j);
        }
    }
    scaled
}


// Learn the softmax of the attention matrix.  Lots of effort in place to make this efficient,
// within the limitations of dynamic allocations.  Might make sense to make a special attention
// operator ala pytorch/tensorflow within the autograd backend.
fn compute_attention_softmax(
    mut attention_matrix: AttentionMatrix,
    d_k: usize
) -> AttentionMatrix {
    // Compute softmax
    let d_k = Constant::scalar((d_k as f32).sqrt());

    // Compute softmax for each non-masked feature
    attention_matrix.iter_mut().for_each(|row| {
        // Get non-zero rows
        let non_zero_row: Vec<_>  = row.iter()
            .filter(|x| x.is_some())
            .map(|x| x.clone().unwrap())
            .collect();

        let nz_row = non_zero_row.concat() / &d_k;
        let sm = softmax(nz_row, false);

        let mut idx = 0;
        row.iter_mut().for_each(|ri| {
            if ri.is_some() {
                *ri = Some(sm.slice(idx,1));
                idx += 1;
            }
        });
    });
    attention_matrix
}

// Simple softmax.
pub fn softmax(numers: ANode, exact: bool) -> ANode {
    // Doesn't need to be part of the graph
    let numer_values = numers.value();
    let max_value = numer_values.iter()
        .max_by_key(|v| FloatOrd(**v))
        .expect("Shouldn't be non-zero!");

    let mv = Constant::scalar(*max_value);
    let n = if exact {
        (numers - &mv).exp()
    } else {
        (numers - &mv).exp_approx()
    };
    &n / n.sum()
}

#[cfg(test)]
mod attention_tests {
    use super::*;
    use rand_xorshift::XorShiftRng;

    fn create_att_vecs() -> Vec<(Attention, f32)> {
        let mha = MultiHeadedAttention {
            d_k: 1,
            num_heads: 1,
            attention_type: AttentionType::Full
        };
        vec![
            (Attention::new(&Variable::new(vec![-1., -1., 1., 1.]), &mha, 0), 1f32),
            (Attention::new(&Variable::new(vec![0., 0., 2., 2.]), &mha, 0), 1f32),
            (Attention::new(&Variable::new(vec![1., 1., -1., -1.]), &mha, 0), 1f32)
        ]
    }

    #[test]
    fn test_attention_matrix_global() {
        let feats = create_att_vecs();

        let exp_att_matrix = vec![
            vec![vec![-1. * -1.], vec![-1. * 0.], vec![-1. * 1.]],
            vec![vec![0.  * -1.], vec![0.  * 0.], vec![0.  * 1.]],
            vec![vec![1.  * -1.], vec![1.  * 0.], vec![1.  * 1.]],
        ];

        let mut rng = XorShiftRng::seed_from_u64(0);
        let att_matrix = compute_attention_matrix(&feats, &mut AttentionType::Full, &mut rng);
        for (row, exp_row) in att_matrix.into_iter().zip(exp_att_matrix.into_iter()) {
            for (ri, eri) in row.into_iter().zip(exp_row.into_iter()) {
                assert_eq!(ri.unwrap().value(), eri);
            }
        }

    }

    #[test]
    fn test_attention_matrix_cw() {
        let feats = create_att_vecs();

        let exp_att_matrix = vec![
            vec![vec![-1. * -1.], vec![-1. * 0.], vec![0.]],
            vec![vec![0.  * -1.], vec![0.  * 0.], vec![0.  * 1.]],
            vec![vec![0.], vec![1.  * 0.], vec![1.  * 1.]],
        ];

        let mut rng = XorShiftRng::seed_from_u64(0);
        let att_matrix = compute_attention_matrix(&feats, &mut AttentionType::Sliding {window_size: 1}, &mut rng);
        for (row, exp_row) in att_matrix.into_iter().zip(exp_att_matrix.into_iter()) {
            assert_eq!(row.len(), exp_row.len());
            for (ri, eri) in row.into_iter().zip(exp_row.into_iter()) {
                if let Some(v) = ri {
                    assert_eq!(v.value(), eri);
                } else {
                    assert_eq!(eri, vec![0f32]);
                }
            }
        }

        let exp_att_matrix = vec![
            vec![vec![-1. * -1.], vec![-1. * 0.], vec![-1. * 1.]],
            vec![vec![0.  * -1.], vec![0.  * 0.], vec![0.  * 1.]],
            vec![vec![1.  * -1.], vec![1.  * 0.], vec![1.  * 1.]],
        ];

        // larger window than feat set
        let att_matrix = compute_attention_matrix(&feats, &mut AttentionType::Sliding { window_size: 10}, &mut rng);
        for (row, exp_row) in att_matrix.into_iter().zip(exp_att_matrix.into_iter()) {
            assert_eq!(row.len(), exp_row.len());
            for (ri, eri) in row.into_iter().zip(exp_row.into_iter()) {
                if let Some(v) = ri {
                    assert_eq!(v.value(), eri);
                } else {
                    assert_eq!(eri, vec![0f32]);
                }
            }
        }
    }

    #[test]
    fn test_att_softmax() {
        let feats = create_att_vecs();

        let exp_softmax = vec![
            vec![0.66524096,0.24472847,0.09003057],
            vec![1./3.,1./3.,1./3.],
            vec![0.09003057, 0.24472847, 0.66524096],
        ];

        let mut rng = XorShiftRng::seed_from_u64(0);
        let att_matrix = compute_attention_matrix(&feats, &mut AttentionType::Full, &mut rng);
        let softmax_matrix = compute_attention_softmax(att_matrix, 1);

        assert_eq!(softmax_matrix.len(), exp_softmax.len());
        for (row, exp_row) in softmax_matrix.into_iter().zip(exp_softmax.into_iter()) {
            let r: Vec<_> = row.into_iter().map(|x| x.unwrap()).collect();
            assert_eq!(r.concat().value(), exp_row);
        }

    }

    #[test]
    fn test_att_reweighted() {
        let feats = create_att_vecs();

        let mut rng = XorShiftRng::seed_from_u64(0);
        let att_matrix = compute_attention_matrix(&feats, &mut AttentionType::Full, &mut rng);
        let softmax_matrix = compute_attention_softmax(att_matrix, 1);
        let reweighted = scale_vecs(feats, &softmax_matrix).collect::<Vec<_>>();

        let exp_weights = vec![
            vec![ 1.0647,  1.0647],
            vec![ 0.6667,  0.6667],
            vec![-0.0858, -0.0858]
        ];

        for row in reweighted.iter() {
            println!("{:?}", row.value());
        }

        for (row, erow) in reweighted.iter().zip(exp_weights.into_iter()) {
            for (ri, eri) in row.value().iter().zip(erow.iter()) {
                assert!((ri - eri).abs() < 1e-4);
            }
        }
    }



    /// Layout per feature with d_k=1, 2 heads, d_model=1: [Q1, Q2, K1, K2, V1, V2].  One feature
    /// gives the mean over heads of V_h, the same as the general path.
    #[test]
    fn test_single_feature_uses_all_heads() {
        let mha = MultiHeadedAttention::new(2, 1, AttentionType::Full);
        let feats = vec![(Variable::new(vec![0., 0., 0., 0., 1., 3.]), 1f32)];
        let mut rng = XorShiftRng::seed_from_u64(0);
        let out = attention_mean(feats.iter(), &mha, &mut rng);
        assert_eq!(out.value(), &[2.]);
    }

    /// Sampling keys must not change the logits of the keys that were sampled: each should be
    /// the plain q_i . k_j, as in Full attention.  Softmax already normalizes over the sample.
    #[test]
    fn test_random_attention_does_not_rescale_logits() {
        let mha = MultiHeadedAttention::new(1, 1, AttentionType::Random {
            num_features: Sample::Fixed(2)
        });
        let items: Vec<_> = (0..4).map(|i| {
            let e = Variable::new(vec![1. + i as f32, 2. - i as f32, 1.]);
            (Attention::new(&e, &mha, 0), 1f32)
        }).collect();
        let mut rng = XorShiftRng::seed_from_u64(0);
        let m = compute_attention_matrix(&items, &mha.attention_type, &mut rng);
        let mut checked = 0;
        for (i, row) in m.iter().enumerate() {
            for (j, logit) in row.iter().enumerate() {
                if let Some(l) = logit {
                    let expected = (1. + i as f32) * (2. - j as f32);
                    assert_eq!(l.value(), &[expected], "logit ({}, {})", i, j);
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 8);
    }
}

#[cfg(test)]
mod attention_weights_tests {
    use super::*;

    /// dims 2, 2 heads of width 1: head h's query, key and value are all x[h], and `out` is the
    /// identity, so each head attends over one coordinate.
    fn per_coordinate() -> AttentionWeights {
        let qkv = vec![1., 0., 0., 1.,   1., 0., 0., 1.,   1., 0., 0., 1.];
        AttentionWeights::new(2, 2, 1, qkv, vec![1., 0., 0., 1.]).unwrap()
    }

    fn pooled(w: &AttentionWeights, x: &[&[f32]], weights: &[f32]) -> Vec<f32> {
        let mut out = vec![f32::NAN; w.dims()];
        w.pool(x, weights, &mut out);
        out
    }

    #[test]
    fn test_head_dims_match_the_gpu_rule() {
        for (dims, heads, expected) in [(151, 4, 40), (128, 4, 32), (8, 16, 8), (16, 2, 8), (17, 2, 16), (1, 1, 8)] {
            assert_eq!(AttentionWeights::head_dims_for(dims, heads), expected, "{} / {}", dims, heads);
        }
    }

    #[test]
    fn test_new_rejects_bad_shapes_and_values() {
        assert!(AttentionWeights::new(2, 2, 1, vec![0.; 11], vec![0.; 4]).is_err());
        assert!(AttentionWeights::new(2, 2, 1, vec![0.; 12], vec![0.; 5]).is_err());
        assert!(AttentionWeights::new(0, 2, 1, vec![], vec![]).is_err());
        assert!(AttentionWeights::new(2, 0, 1, vec![], vec![]).is_err());
        let mut qkv = vec![0.; 12];
        qkv[3] = f32::NAN;
        assert!(AttentionWeights::new(2, 2, 1, qkv, vec![0.; 4]).is_err());
        assert!(AttentionWeights::new(2, 2, 1, vec![0.; 12], vec![0., f32::INFINITY, 0., 0.]).is_err());
    }

    #[test]
    fn test_hand_computed_two_tokens() {
        let (x1, x2) = ([1f32, 0.], [0f32, 2.]);
        let e = 1f32.exp();
        let e4 = 4f32.exp();
        // Head 0: logits [1, 0] for token 1 and [0, 0] for token 2, values [1, 0].
        let ctx0 = 0.5 * e / (e + 1.) + 0.5 * 0.5;
        // Head 1: logits [0, 0] for token 1 and [0, 4] for token 2, values [0, 2].
        let ctx1 = 0.5 * 1. + 0.5 * 2. * e4 / (1. + e4);
        let out = pooled(&per_coordinate(), &[&x1, &x2], &[1., 1.]);
        assert!((out[0] - (0.5 + ctx0)).abs() < 1e-6, "{:?}", out);
        assert!((out[1] - (1.0 + ctx1)).abs() < 1e-6, "{:?}", out);
        // A zero-weight token is padding: it changes nothing.
        let padded = pooled(&per_coordinate(), &[&x1, &x2, &[9., 9.]], &[1., 1., 0.]);
        assert_eq!(out, padded);
    }

    /// Logits are scaled by 1/sqrt(head_dims): with head width 4 and q = k = x, two tokens'
    /// self logits are |x|^2 / 2.
    #[test]
    fn test_logits_are_scaled_by_head_width() {
        let mut qkv = vec![0f32; 3 * 4 * 4];
        for part in 0..3 {
            for i in 0..4 { qkv[(part * 4 + i) * 4 + i] = 1.; }
        }
        let mut out_w = vec![0f32; 16];
        for i in 0..4 { out_w[i * 4 + i] = 1.; }
        let w = AttentionWeights::new(4, 1, 4, qkv, out_w).unwrap();
        let (a, b) = ([1f32, 1., 0., 0.], [0f32, 0., 1., 1.]);
        // Token a: logits [2/2, 0] = [1, 0]; token b likewise.
        let p = 1f32.exp() / (1f32.exp() + 1.);
        let out = pooled(&w, &[&a, &b], &[1., 1.]);
        let expected = [0.5 + p * 0.5 + (1. - p) * 0.5; 4];
        assert!(out.iter().zip(expected.iter()).all(|(o, e)| (o - e).abs() < 1e-6), "{:?}", out);
    }

    #[test]
    fn test_zero_output_projection_is_the_weighted_mean() {
        let mut rng = rand_xorshift::XorShiftRng::seed_from_u64(1);
        let w = AttentionWeights::init(3, 2, &mut rng).unwrap();
        assert!(w.out().iter().all(|v| *v == 0.));
        let out = pooled(&w, &[&[1., 2., 3.], &[3., 2., 1.]], &[3., 1.]);
        assert_eq!(out, vec![1.5, 2., 2.5]);
    }

    #[test]
    fn test_single_token_is_itself_plus_its_projected_value() {
        let out = pooled(&per_coordinate(), &[&[2., 3.]], &[1.]);
        assert_eq!(out, vec![4., 6.]);
    }

    #[test]
    fn test_no_tokens_pool_to_zeros() {
        assert_eq!(pooled(&per_coordinate(), &[], &[]), vec![0., 0.]);
        assert_eq!(pooled(&per_coordinate(), &[&[1., 1.]], &[0.]), vec![0., 0.]);
    }

    #[test]
    fn test_init_shapes_and_bounds() {
        let mut rng = rand_xorshift::XorShiftRng::seed_from_u64(2);
        let w = AttentionWeights::init(151, 4, &mut rng).unwrap();
        assert_eq!((w.dims(), w.heads(), w.head_dims()), (151, 4, 40));
        assert_eq!(w.qkv().len(), 3 * 160 * 151);
        let bound = 1. / 151f32.sqrt();
        assert!(w.qkv().iter().all(|v| v.abs() <= bound));
    }
}
