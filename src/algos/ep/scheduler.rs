
pub enum LRScheduler {
    Attention {
        min_alpha: f32,
        alpha: f32,
        warmup_steps: usize 
    },
    ExpDecay {
        min_alpha: f32,
        alpha: f32,
        decay: f32
    },
}

impl LRScheduler {
    pub fn attention(min_alpha: f32, alpha: f32, warmup_steps: usize) -> Self {
        LRScheduler::Attention { min_alpha, alpha, warmup_steps }
    }

    pub fn exp_decay(min_alpha: f32, alpha: f32, decay: f32) -> Self {
        LRScheduler::ExpDecay { min_alpha, alpha, decay }
    }

    pub fn compute(&self, cur_step: usize) -> f32 {
        match self {
            LRScheduler::Attention {min_alpha, alpha, warmup_steps} => {
                if cur_step > *warmup_steps {
                    alpha / ((cur_step - warmup_steps) as f32).sqrt()
                } else {
                    let ratio = cur_step as f32 / *warmup_steps as f32;
                    min_alpha + ratio * (alpha - min_alpha)
                }
            },
            LRScheduler::ExpDecay { min_alpha, alpha, decay } => {
                (alpha * decay.powf(cur_step as f32)).max(*min_alpha)
            }
        }

    }

}


