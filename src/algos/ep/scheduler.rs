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
                if cur_step > *warmup_steps {
                    let ratio = cur_step as f32 / *max_steps as f32;
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


