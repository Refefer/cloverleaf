//! Defines the actual gradient optimizers for SGD.
//! Nothing particularly special about these, just canonical versions of the ones used in practice.
use rayon::prelude::*;
use crate::embeddings::EmbeddingStore;
use crate::distance::Distance;
use std::collections::{HashMap as CHashMap};

/// Optimizer trait.  We provide it the feature set, the gradient maps, and a few other details
/// (such as alpha==learning rate), and it optimizes.
pub trait Optimizer: Sync {
    /// Applies one feature's gradient.  `step` is the 1-based count of updates applied so far,
    /// including this one.
    fn update_row(
        &self,
        feature_embeddings: &EmbeddingStore,
        feat_id: usize,
        grad: &[f32],
        alpha: f32,
        step: f32
    );

    /// Called once at the end of every pass, with the learning rate schedule's multiplier there
    /// (see [`LRScheduler::multiplier`](super::scheduler::LRScheduler::multiplier)).  For
    /// regularisation that should happen once per pass whatever the number of updates.  Does
    /// nothing by default.
    fn end_pass(&self, _feature_embeddings: &EmbeddingStore, _schedule_multiplier: f32) {}

    /// Applies a batch of per-feature gradients in parallel.
    fn update(
        &self, 
        feature_embeddings: &EmbeddingStore,
        grads: CHashMap<usize, Vec<f32>>,
        alpha: f32,
        step: f32
    ) {
        grads.into_par_iter().for_each(|(feat_id, grad)| {
            self.update_row(feature_embeddings, feat_id, &grad, alpha, step);
        });
    }

}

/// Decoupled weight decay rate for AdamW.  Always finite and non-negative.
#[derive(Clone,Copy,Debug,PartialEq)]
pub struct WeightDecay(f32);

impl WeightDecay {
    pub fn new(rate: f32) -> Option<Self> {
        if rate.is_finite() && rate >= 0f32 { Some(WeightDecay(rate)) } else { None }
    }

    pub fn rate(&self) -> f32 {
        self.0
    }
}

/// Which optimizer the learners use.
#[derive(Clone,Copy,Debug,PartialEq)]
pub enum OptimizerKind {
    /// Adam (Kingma & Ba, 2015).
    Adam,

    /// AdamW (Loshchilov & Hutter, 2019): Adam plus weight decay applied directly to the weights
    /// rather than folded into the gradient, so the decay isn't rescaled by Adam's per-parameter
    /// step sizes.  The rate is per pass: at the end of each pass every feature embedding is
    /// multiplied by `exp(-m * rate)`, where `m` is the learning rate schedule's multiplier.
    AdamW(WeightDecay)
}

/// Adam Optimizer, with optional decoupled weight decay (AdamW).  Should basically be always
/// preferred over momentum due to better performance in almost all cases.  Requires 3x the
/// learnable parameters (aka all the feature embeddings): the weights plus the first and second
/// moments.  Weight decay adds no memory.
///
/// Gradient updates are sparse: only features with a gradient in a batch are touched.  Weight
/// decay is not: it is applied to every feature once per pass (see [`Optimizer::end_pass`]), so
/// how much a feature decays doesn't depend on how often it gets a gradient, on whether updates
/// are per batch or per pass, or on the base learning rate.  Decaying a feature only when it was
/// updated would tie its decay to its frequency, and decaying it lazily for the updates it missed
/// would need a step counter per feature.
pub struct AdamOptimizer {
    beta_1: f32,
    beta_2: f32,
    eps: f32,
    weight_decay: f32,
    mom: EmbeddingStore,
    var: EmbeddingStore
}

impl AdamOptimizer {
    pub fn new(kind: OptimizerKind, dims: usize, length: usize) -> Self {
        let weight_decay = match kind {
            OptimizerKind::Adam => 0f32,
            OptimizerKind::AdamW(wd) => wd.rate()
        };
        let mom = EmbeddingStore::new(length, dims, Distance::Cosine);
        let var = EmbeddingStore::new(length, dims, Distance::Cosine);
        AdamOptimizer { beta_1: 0.9, beta_2: 0.999, eps: 1e-8, weight_decay, mom, var }
    }
}

impl Optimizer for AdamOptimizer {

    fn update_row(
        &self,
        feature_embeddings: &EmbeddingStore,
        feat_id: usize,
        grad: &[f32],
        alpha: f32,
        step: f32
    ) {
        let t = step;
        {
            // Can get some nans in weird cases, such as the distance between
            // a node and it's reconstruction when it shares all features.
            // We just skip over those weird ones.
            if grad.iter().all(|gi| !gi.is_nan()) {

                // Update first order mean
                let mom = self.mom.get_embedding_mut_hogwild(feat_id);
                mom.iter_mut().zip(grad.iter()).for_each(|(m_i, g_i)| {
                    *m_i = self.beta_1 * *m_i + (1. - self.beta_1) * g_i;
                });

                // Update secord order variance 
                let var = self.var.get_embedding_mut_hogwild(feat_id);
                var.iter_mut().zip(grad.iter()).for_each(|(v_i, g_i)| {
                    *v_i = self.beta_2 * *v_i + (1. - self.beta_2) * g_i.powf(2.);
                });

                // Create the new grad and update
                let emb = feature_embeddings.get_embedding_mut_hogwild(feat_id);
                let bias_1 = 1. - self.beta_1.powf(t);
                let bias_2 = 1. - self.beta_2.powf(t);
                emb.iter_mut().zip(mom.iter().zip(var.iter())).for_each(|(e_i, (m_i, v_i))| {
                    let m_i = m_i / bias_1;
                    let v_i = v_i / bias_2;
                    let g_i = m_i / (v_i.sqrt() + self.eps);
                    *e_i -= alpha * g_i;
                });

            }
        }
    }

    /// Weight decay: `exp` keeps the factor in (0, 1], so a large rate can't flip signs.
    fn end_pass(&self, feature_embeddings: &EmbeddingStore, schedule_multiplier: f32) {
        if self.weight_decay > 0. && schedule_multiplier > 0. {
            let factor = (-schedule_multiplier * self.weight_decay).exp();
            (0..feature_embeddings.len()).into_par_iter().for_each(|feat_id| {
                feature_embeddings.get_embedding_mut_hogwild(feat_id).iter_mut()
                    .for_each(|e_i| *e_i *= factor);
            });
        }
    }
}


#[cfg(test)]
mod optimizer_tests {
    use super::*;

    /// With bias correction at step t, a constant gradient gives m_hat = g and v_hat = g^2, so
    /// every Adam step moves the weight by exactly alpha (minus eps).
    #[test]
    fn test_adam_steps_are_alpha_sized_with_constant_grad() {
        let fe = EmbeddingStore::new(1, 1, Distance::Cosine);
        let adam = AdamOptimizer::new(OptimizerKind::Adam, 1, 1);
        let mut last = 0f32;
        for step in 1..6 {
            let mut grads = CHashMap::new();
            grads.insert(0usize, vec![2f32]);
            adam.update(&fe, grads, 0.1, step as f32);
            let cur = fe.get_embedding(0)[0];
            assert!(((last - cur) - 0.1).abs() < 1e-4, "step {} moved {}", step, last - cur);
            last = cur;
        }
    }

    fn store_with(values: &[f32]) -> EmbeddingStore {
        let fe = EmbeddingStore::new(values.len(), 1, Distance::Cosine);
        for (i, v) in values.iter().enumerate() {
            fe.get_embedding_mut_hogwild(i)[0] = *v;
        }
        fe
    }

    /// Updates don't decay: AdamW's gradient step is exactly Adam's.
    #[test]
    fn test_adamw_update_matches_adam() {
        let adam_fe = store_with(&[1.0]);
        let adamw_fe = store_with(&[1.0]);
        let adam = AdamOptimizer::new(OptimizerKind::Adam, 1, 1);
        let adamw = AdamOptimizer::new(OptimizerKind::AdamW(WeightDecay::new(0.5).unwrap()), 1, 1);
        for (opt, fe) in [(&adam, &adam_fe), (&adamw, &adamw_fe)] {
            let mut grads = CHashMap::new();
            grads.insert(0usize, vec![2f32]);
            opt.update(fe, grads, 0.1, 1.0);
        }
        assert!((adam_fe.get_embedding(0)[0] - 0.9).abs() < 1e-4);
        assert_eq!(adam_fe.get_embedding(0)[0], adamw_fe.get_embedding(0)[0]);
    }

    /// The end of a pass multiplies every feature by exp(-multiplier * rate), whether or not it
    /// had a gradient.
    #[test]
    fn test_adamw_end_pass_decays_every_feature() {
        let fe = store_with(&[1.0, -2.0]);
        let adamw = AdamOptimizer::new(OptimizerKind::AdamW(WeightDecay::new(0.5).unwrap()), 1, 2);
        adamw.end_pass(&fe, 0.4);
        let factor = (-0.2f32).exp();
        assert!((fe.get_embedding(0)[0] - factor).abs() < 1e-6, "{}", fe.get_embedding(0)[0]);
        assert!((fe.get_embedding(1)[0] + 2. * factor).abs() < 1e-6, "{}", fe.get_embedding(1)[0]);
    }

    /// A huge rate shrinks weights toward zero but never flips their sign.
    #[test]
    fn test_adamw_end_pass_never_flips_sign() {
        let fe = store_with(&[1.0]);
        let adamw = AdamOptimizer::new(OptimizerKind::AdamW(WeightDecay::new(50.).unwrap()), 1, 1);
        adamw.end_pass(&fe, 1.0);
        let w = fe.get_embedding(0)[0];
        assert!(w >= 0. && w < 1e-6, "{}", w);
    }

    /// Adam has no decay, and a zero schedule multiplier means no decay for AdamW.
    #[test]
    fn test_end_pass_without_decay_is_a_no_op() {
        let fe = store_with(&[1.0]);
        AdamOptimizer::new(OptimizerKind::Adam, 1, 1).end_pass(&fe, 1.0);
        AdamOptimizer::new(OptimizerKind::AdamW(WeightDecay::new(0.5).unwrap()), 1, 1).end_pass(&fe, 0.0);
        assert_eq!(fe.get_embedding(0)[0], 1.0);
    }

    #[test]
    fn test_weight_decay_rejects_negative_and_non_finite() {
        assert!(WeightDecay::new(0.0).is_some());
        assert!(WeightDecay::new(0.01).is_some());
        assert!(WeightDecay::new(-0.01).is_none());
        assert!(WeightDecay::new(f32::NAN).is_none());
        assert!(WeightDecay::new(f32::INFINITY).is_none());
    }
}
