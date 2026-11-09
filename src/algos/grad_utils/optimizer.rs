//! Defines the actual gradient optimizers for SGD.
//! Nothing particularly special about these, just canonical versions of the ones used in practice.
use rayon::prelude::*;
use crate::embeddings::EmbeddingStore;
use crate::distance::Distance;
use std::collections::{HashMap as CHashMap};

/// Optimizer trait.  We provide it the feature set, the gradient maps, and a few other details
/// (such as alpha==learning rate), and it optimizes.
pub trait Optimizer {
    fn update(
        &self, 
        feature_embeddings: &EmbeddingStore,
        grads: CHashMap<usize, Vec<f32>>,
        alpha: f32,
        // 1-based count of updates applied so far, including this one
        step: f32
    );

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
    /// step sizes.
    AdamW(WeightDecay)
}

/// Adam Optimizer, with optional decoupled weight decay (AdamW).  Should basically be always
/// preferred over momentum due to better performance in almost all cases.  Requires 3x the
/// learnable parameters (aka all the feature embeddings): the weights plus the first and second
/// moments.  Weight decay adds no memory.
///
/// Updates are sparse: only features with a gradient in a batch are touched, so weight decay is
/// also only applied to those features.  Rare features are decayed less often than common ones.
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

    fn update(
        &self, 
        feature_embeddings: &EmbeddingStore,
        grads: CHashMap<usize, Vec<f32>>,
        alpha: f32,
        step: f32
    ) {
        let t = step;
        grads.into_par_iter().for_each(|(feat_id, grad)| {

            // Can get some nans in weird cases, such as the distance between
            // a node and it's reconstruction when it shares all features.
            // We just skip over those weird ones.
            if grad.par_iter().all(|gi| !gi.is_nan()) {

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
                let decay = 1. - alpha * self.weight_decay;
                emb.iter_mut().zip(mom.iter().zip(var.iter())).for_each(|(e_i, (m_i, v_i))| {
                    let m_i = m_i / (1. - self.beta_1.powf(t));
                    let v_i = v_i / (1. - self.beta_2.powf(t));
                    let g_i = m_i / (v_i.sqrt() + self.eps);
                    *e_i = *e_i * decay - alpha * g_i;
                });

            }
        });
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

    /// AdamW decays the weight by alpha * weight_decay * w on top of the Adam step.  A zero
    /// gradient makes the Adam step zero, isolating the decay.
    #[test]
    fn test_adamw_decays_weights_decoupled_from_gradient() {
        let fe = store_with(&[1.0]);
        let wd = WeightDecay::new(0.5).unwrap();
        let adamw = AdamOptimizer::new(OptimizerKind::AdamW(wd), 1, 1);
        let mut grads = CHashMap::new();
        grads.insert(0usize, vec![0f32]);
        adamw.update(&fe, grads, 0.1, 1.0);
        assert!((fe.get_embedding(0)[0] - 0.95).abs() < 1e-6, "{}", fe.get_embedding(0)[0]);
    }

    /// Decay is decoupled: the step size from the gradient is the same as Adam's.
    #[test]
    fn test_adamw_gradient_step_matches_adam() {
        let adam_fe = store_with(&[1.0]);
        let adamw_fe = store_with(&[1.0]);
        let adam = AdamOptimizer::new(OptimizerKind::Adam, 1, 1);
        let adamw = AdamOptimizer::new(OptimizerKind::AdamW(WeightDecay::new(0.5).unwrap()), 1, 1);
        for (opt, fe) in [(&adam, &adam_fe), (&adamw, &adamw_fe)] {
            let mut grads = CHashMap::new();
            grads.insert(0usize, vec![2f32]);
            opt.update(fe, grads, 0.1, 1.0);
        }
        // Adam: 1 - 0.1.  AdamW: 1 - 0.1 * 0.5 * 1 - 0.1.
        assert!((adam_fe.get_embedding(0)[0] - 0.9).abs() < 1e-4);
        assert!((adamw_fe.get_embedding(0)[0] - 0.85).abs() < 1e-4);
    }

    /// Only features with a gradient this step are decayed; the rest are left untouched.
    #[test]
    fn test_adamw_only_decays_updated_features() {
        let fe = store_with(&[1.0, 1.0]);
        let adamw = AdamOptimizer::new(OptimizerKind::AdamW(WeightDecay::new(0.5).unwrap()), 1, 2);
        let mut grads = CHashMap::new();
        grads.insert(0usize, vec![0f32]);
        adamw.update(&fe, grads, 0.1, 1.0);
        assert_eq!(fe.get_embedding(1)[0], 1.0);
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
