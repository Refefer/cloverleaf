//! Learning rate scheduler.  We use a couple of different variants depending on the method needed.

/// Tracks the different schedulers which have different benefits
#[derive(Debug)]
pub enum LRScheduler {
    /// Generally useful: uses a warmup period to get good starting gradients in Adam, then slowly
    /// decays over the course of the rest of the passes.
    CosDecay {
        min_alpha: f32,
        alpha: f32,
        warmup_steps: usize,
        max_steps: usize
    },
    /// Simple exponential decay
    ExpDecay {
        min_alpha: f32,
        alpha: f32,
        decay: f32
    },
    /// Returns zero
    Noop
}

impl LRScheduler {
    pub fn cos_decay(min_alpha: f32, alpha: f32, warmup_steps: usize, max_steps: usize) -> Self {
        LRScheduler::CosDecay { min_alpha, alpha, warmup_steps, max_steps }
    }

    pub fn exp_decay(min_alpha: f32, alpha: f32, decay: f32) -> Self {
        LRScheduler::ExpDecay { min_alpha, alpha, decay }
    }

    pub fn noop() -> Self {
        LRScheduler::Noop 
    }

    pub fn compute(&self, cur_step: usize) -> f32 {
        match self {
            LRScheduler::CosDecay {min_alpha, alpha, warmup_steps, max_steps} => {
                if *max_steps == 0 {
                    // No planned steps, so there is no schedule to follow.
                    *alpha
                } else if cur_step > *warmup_steps {
                    // Decay runs over the steps after warmup, so it starts at alpha
                    let decay_steps = max_steps.saturating_sub(*warmup_steps);
                    let ratio = if decay_steps == 0 {
                        1f32
                    } else {
                        ((cur_step - warmup_steps) as f32 / decay_steps as f32).min(1f32)
                    };
                    min_alpha + 0.5 * (alpha - min_alpha) * (1f32 + (std::f32::consts::PI * ratio).cos())
                } else {
                    let ratio = cur_step as f32 / *warmup_steps as f32;
                    min_alpha + ratio * (alpha - min_alpha)
                }
            },
            LRScheduler::ExpDecay { min_alpha, alpha, decay } => {
                (alpha * decay.powf(cur_step as f32)).max(*min_alpha)
            },
            LRScheduler::Noop => 0.0
        }

    }

}



#[cfg(test)]
mod scheduler_tests {
    use super::*;

    /// Fewer examples than one batch means zero planned steps; the LR must stay finite.
    #[test]
    fn test_cos_decay_zero_steps_is_finite() {
        let s = LRScheduler::cos_decay(0.01, 1.0, 0, 0);
        for step in 0..3 {
            assert!(s.compute(step).is_finite(), "step {} -> {}", step, s.compute(step));
        }
    }

    /// Steps past max_steps must not climb back up the cosine.
    #[test]
    fn test_cos_decay_does_not_rise_past_max_steps() {
        let s = LRScheduler::cos_decay(0.01, 1.0, 2, 10);
        let at_end = s.compute(10);
        for step in 11..15 {
            assert!(s.compute(step) <= at_end + 1e-6,
                "step {} -> {} > {}", step, s.compute(step), at_end);
        }
    }

    /// The decay phase must start where warmup ends.
    #[test]
    fn test_cos_decay_is_continuous_after_warmup() {
        let s = LRScheduler::cos_decay(0.01, 1.0, 100, 1000);
        let (a, b) = (s.compute(100), s.compute(101));
        assert!((a - b).abs() < 1e-3, "warmup end {} -> {}", a, b);
    }
}
