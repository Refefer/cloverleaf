//! Attention pooling: multi-head self-attention over a node's feature embeddings, the same model
//! as the GPU trainer's `AttentionPool` (gpu/ep_gpu/model.py), so weights trained by either embed
//! nodes identically.  [`AttentionWeights`] holds the weights and pools in plain f32, for
//! embedding nodes after training; [`attention_pool`] is the same computation on the autograd
//! graph, for training.
use crate::autograd::*;
use rand::prelude::*;

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

    /// Every weight, `qkv` then `out`, the order optimisers and [`AttentionVars::gradients`] use.
    pub fn parameters(&self) -> Vec<f32> {
        self.qkv.iter().chain(self.out.iter()).cloned().collect()
    }

    /// The same shapes with new weights, in [`parameters`](Self::parameters) order.
    pub fn with_parameters(&self, parameters: Vec<f32>) -> Result<Self, Error> {
        let split = self.qkv.len();
        if parameters.len() != split + self.out.len() {
            return Err(Error::InvalidArgument(format!(
                "expected {} attention parameters, got {}", split + self.out.len(), parameters.len())))
        }
        let mut qkv = parameters;
        let out = qkv.split_off(split);
        AttentionWeights::new(self.dims, self.heads, self.head_dims, qkv, out)
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

/// The weights as autograd variables, made once per training forward pass from a snapshot of
/// the current weights, so every embedding in the pass uses (and its gradient flows to) the same
/// ones.
pub struct AttentionVars {
    qkv: ANode,
    out: ANode,
    heads: usize,
    head_dims: usize
}

impl AttentionVars {
    pub fn new(weights: &AttentionWeights) -> Self {
        let (dims, inner) = (weights.dims, weights.inner());
        AttentionVars {
            qkv: Variable::matrix(&weights.qkv, 3 * inner, dims),
            out: Variable::matrix(&weights.out, dims, inner),
            heads: weights.heads,
            head_dims: weights.head_dims
        }
    }

    /// The gradients of `qkv` then `out` after `graph`'s backward pass, in
    /// [`AttentionWeights::parameters`] order, or None when the loss didn't use the weights.
    pub fn gradients(&self, graph: &Graph) -> Option<Vec<f32>> {
        let (qkv, out) = (graph.get_grad(&self.qkv), graph.get_grad(&self.out));
        if qkv.is_none() && out.is_none() {
            return None
        }
        let qkv = qkv.unwrap_or_else(|| vec![0f32; self.qkv.value().len()]);
        let out = out.unwrap_or_else(|| vec![0f32; self.out.value().len()]);
        Some(qkv.into_iter().chain(out).collect())
    }
}

/// [`AttentionWeights::pool`] on the autograd graph: pools `tokens` (feature embeddings and
/// their weights) into one embedding, differentiable in the tokens and the weights.
pub fn attention_pool(tokens: &[(ANode, f32)], vars: &AttentionVars) -> ANode {
    let tokens: Vec<&(ANode, f32)> = tokens.iter().filter(|(_, w)| *w > 0f32).collect();
    let total: f32 = tokens.iter().map(|(_, w)| w).sum();
    let dims = vars.out.shape()[0];
    if tokens.is_empty() || total <= 0f32 {
        return Constant::new(vec![0f32; dims])
    }
    let (hd, inner) = (vars.head_dims, vars.heads * vars.head_dims);
    let x = ANode::stack(&tokens.iter().map(|(e, _)| e.clone()).collect::<Vec<_>>());
    // (1, T): each token's share of the weighted means.
    let shares = Constant::matrix(&tokens.iter().map(|(_, w)| w / total).collect::<Vec<_>>(), 1, tokens.len());
    let proj = x.matmul(&vars.qkv.t());
    let scale = 1f32 / (hd as f32).sqrt();
    let heads: Vec<ANode> = (0..vars.heads).map(|h| {
        let q = proj.narrow(1, h * hd, hd);
        let k = proj.narrow(1, inner + h * hd, hd);
        let v = proj.narrow(1, 2 * inner + h * hd, hd);
        let attention = (q.matmul(&k.t()) * scale).softmax_rows();
        // The weighted mean of the tokens' contexts, (1, hd).
        shares.matmul(&attention.matmul(&v))
    }).collect();
    let context = ANode::hcat(&heads);
    (shares.matmul(&x) + context.matmul(&vars.out.t())).flatten()
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

#[cfg(test)]
mod attention_pool_tests {
    use super::*;
    use rand_xorshift::XorShiftRng;

    fn random_weights(seed: u64, dims: usize, heads: usize) -> AttentionWeights {
        let mut rng = XorShiftRng::seed_from_u64(seed);
        let w = AttentionWeights::init(dims, heads, &mut rng).unwrap();
        let out = (0..w.out().len()).map(|_| rng.gen::<f32>() - 0.5).collect();
        AttentionWeights::new(dims, heads, w.head_dims(), w.qkv().to_vec(), out).unwrap()
    }

    /// The autograd forward pass is the plain one, for any token count and weights.
    #[test]
    fn test_autograd_pool_matches_plain_pool() {
        let mut rng = XorShiftRng::seed_from_u64(3);
        for (dims, heads, tokens) in [(5, 2, 1), (5, 2, 4), (13, 3, 7), (4, 1, 2)] {
            let w = random_weights(tokens as u64, dims, heads);
            let x: Vec<Vec<f32>> = (0..tokens).map(|_| (0..dims).map(|_| rng.gen::<f32>() * 2. - 1.).collect()).collect();
            let weights: Vec<f32> = (0..tokens).map(|i| 0.5 + i as f32).collect();
            let mut expected = vec![0f32; dims];
            w.pool(&x.iter().map(|v| v.as_slice()).collect::<Vec<_>>(), &weights, &mut expected);
            let toks: Vec<(ANode, f32)> = x.iter().zip(weights.iter()).map(|(v, wi)| (Constant::new(v.clone()), *wi)).collect();
            let got = attention_pool(&toks, &AttentionVars::new(&w)).value();
            assert_eq!(got.len(), dims);
            assert!(got.iter().zip(expected.iter()).all(|(a, b)| (a - b).abs() < 1e-5), "{:?} vs {:?}", got, expected);
        }
    }

    #[test]
    fn test_autograd_pool_with_no_tokens_is_zeros() {
        let w = random_weights(1, 4, 2);
        assert_eq!(attention_pool(&[], &AttentionVars::new(&w)).value(), vec![0.; 4]);
    }

    /// Gradients reach the tokens and both projections, in `parameters` order, and match
    /// finite differences of the plain forward pass.
    #[test]
    fn test_gradients_reach_tokens_and_weights() {
        let w = random_weights(5, 4, 2);
        let x = [vec![0.3f32, -0.2, 0.8, 0.1], vec![-0.5, 0.4, 0.2, 0.9]];
        let target = [0.2f32, -0.1, 0.4, 0.3];
        let loss_of = |w: &AttentionWeights| {
            let mut out = vec![0f32; 4];
            w.pool(&[&x[0], &x[1]], &[1., 1.], &mut out);
            out.iter().zip(target.iter()).map(|(o, t)| o * t).sum::<f32>()
        };
        let vars = AttentionVars::new(&w);
        let toks: Vec<(ANode, f32)> = x.iter().map(|v| (Variable::pooled(v), 1f32)).collect();
        let loss = (attention_pool(&toks, &vars) * Constant::new(target.to_vec())).sum();
        let mut graph = Graph::new();
        graph.backward(&loss);
        assert!(graph.get_grad(&toks[0].0).is_some());
        let grads = vars.gradients(&graph).unwrap();
        let params = w.parameters();
        assert_eq!(grads.len(), params.len());
        let h = 1e-2f32;
        for i in (0..params.len()).step_by(7) {
            let (mut up, mut down) = (params.clone(), params.clone());
            up[i] += h;
            down[i] -= h;
            let numeric = (loss_of(&w.with_parameters(up).unwrap()) - loss_of(&w.with_parameters(down).unwrap())) / (2. * h);
            assert!((grads[i] - numeric).abs() < 2e-3, "parameter {}: {} vs {}", i, grads[i], numeric);
        }
    }

    #[test]
    fn test_with_parameters_round_trips_and_checks_length() {
        let w = random_weights(2, 3, 2);
        assert_eq!(w.with_parameters(w.parameters()).unwrap(), w);
        assert!(w.with_parameters(vec![0.; 3]).is_err());
    }
}
