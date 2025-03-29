/// Defines different distance metrics such that a distance of zero is perfect.
use std::arch::x86_64::*;
use float_ord::FloatOrd;

#[derive(Copy,Clone,Debug)]
pub enum Distance {
    /// A* using Landmark Triangulation
    ALT,

    /// Cosine distance
    Cosine,

    /// Simple Dot distance.  We modify it by taking the negative, so lower is closer.  Not a true
    /// distance but oh well
    Dot,

    /// Simple L2 Norm Euclidean Distance
    Euclidean,

    /// Simple binary hamming distance
    Hamming,

    /// Jaccard distance, treating each float as an identifier
    Jaccard
}

impl Distance {
    
    fn fast_cosine(e1: &[f32], e2: &[f32]) -> f32 {
        let mut d1 = 0.;
        let mut d2 = 0.;
        let dot = e1.iter().zip(e2.iter()).map(|(ei, ej)| {
            d1 += ei.powf(2.);
            d2 += ej.powf(2.);
            ei * ej
        }).sum::<f32>();
        let cosine_score = dot / (d1.sqrt() * d2.sqrt());
        if cosine_score.is_nan() {
            std::f32::INFINITY
        } else {
            -cosine_score + 1.
        }
    }

    /// Horizontal sum of all 8 floats in an __m256, returning a single f32.
    /// Uses pairwise `_mm_hadd_ps` in SSE after extracting the high 128 bits.
    ///
    /// If you prefer a different approach (like repeated `_mm256_hadd_ps`),
    /// you can adapt this function to suit.
    #[target_feature(enable = "avx")]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe fn hsum_avx_ps(v: __m256) -> f32 {
        // Extract the high 128 bits
        let high = _mm256_extractf128_ps(v, 1);
        // Add the low 128 bits (cast) to the high 128 bits
        let sum_128 = _mm_add_ps(_mm256_castps256_ps128(v), high);
        // Now we have a 128-bit register with 4 floats. Use SSE horizontal add twice.
        let sum_128 = _mm_hadd_ps(sum_128, sum_128);
        let sum_128 = _mm_hadd_ps(sum_128, sum_128);
        // Move the lowest float out
        _mm_cvtss_f32(sum_128)
    }

    #[target_feature(enable = "avx")]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub unsafe fn fast_cosine_similarity_avx(a: &[f32], b: &[f32]) -> f32 {
        // Ensure slices have the same length
        assert_eq!(a.len(), b.len(), "Input slices must have the same length.");

        let length = a.len();
        let mut i = 0;

        // 256-bit accumulators
        let mut dot_sum = _mm256_setzero_ps();
        let mut a_sum   = _mm256_setzero_ps();
        let mut b_sum   = _mm256_setzero_ps();

        // Process in chunks of 8 floats
        while i + 8 <= length {
            // Load 8 f32s from each slice (unaligned load)
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));

            // dot = a[i] * b[i]
            dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
            //let product = _mm256_mul_ps(va, vb);
            //dot_sum = _mm256_add_ps(dot_sum, product);

            // a^2, b^2 accumulators
            a_sum = _mm256_fmadd_ps(va, va, a_sum);
            //let va2 = _mm256_mul_ps(va, va);
            //a_sum = _mm256_add_ps(a_sum, va2);

            b_sum = _mm256_fmadd_ps(vb, vb, b_sum);
            //let vb2 = _mm256_mul_ps(vb, vb);
            //b_sum = _mm256_add_ps(b_sum, vb2);

            i += 8;
        }

        // Reduce each accumulator to a single f32
        let dot_val = Distance::hsum_avx_ps(dot_sum);
        let a_val   = Distance::hsum_avx_ps(a_sum);
        let b_val   = Distance::hsum_avx_ps(b_sum);

        // Handle any leftover elements
        let mut tail_dot = 0.0;
        let mut tail_a   = 0.0;
        let mut tail_b   = 0.0;

        while i < length {
            tail_dot += a[i] * b[i];
            tail_a   += a[i] * a[i];
            tail_b   += b[i] * b[i];
            i += 1;
        }

        let dot   = dot_val + tail_dot;
        let norma = a_val + tail_a;
        let normb = b_val + tail_b;

        // cosine similarity = dot / (||a|| * ||b||)
        let dist = -(dot / (norma.sqrt() * normb.sqrt())) + 1f32;
        if dist.is_nan() {
            std::f32::INFINITY
        } else {
            dist
        }
    }

    pub fn compute(&self, e1: &[f32], e2: &[f32]) -> f32 {
        match &self {
            Distance::ALT => e1.iter().zip(e2.iter())
                .map(|(ei, ej)| (*ei - *ej).abs())
                .max_by_key(|v| FloatOrd(*v)).unwrap_or(0.),

            Distance::Cosine => {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    // Note that this `unsafe` block is safe because we're testing
                    // that the `avx2` feature is indeed available on our CPU.
                    if is_x86_feature_detected!("avx") {
                        return unsafe { Distance::fast_cosine_similarity_avx(e1, e2) };
                    }
                }
                Distance::fast_cosine(e1, e2)
            },

            Distance::Euclidean => {
                e1.iter().zip(e2.iter()).map(|(ei, ej)| {
                    (*ei - *ej).powf(2.)
                }).sum::<f32>().sqrt()
            },

            Distance::Dot => {
                -e1.iter().zip(e2.iter()).map(|(ei, ej)| {
                    *ei * *ej
                }).sum::<f32>()
            },

            Distance::Hamming => {
                let not_matches = e1.iter().zip(e2.iter()).map(|(ei, ej)| {
                    if *ei != *ej { 1f32 } else {0f32}
                }).sum::<f32>();
                not_matches / e1.len() as f32
            },

            Distance::Jaccard => {
                let mut idx1 = 0;
                let mut idx2 = 0;
                let mut matches = 0;
                while idx1 < e1.len() && idx2 < e2.len() && e1[idx1] >= 0. && e2[idx2] >= 0. {
                    let v1 = e1[idx1];
                    let v2 = e2[idx2];
                    if v1 == v2 {
                        matches += 1;
                        idx1 += 1;
                        idx2 += 1;
                    } else if v1 < v2 {
                        idx1 += 1;
                    } else {
                        idx2 += 1;
                    }
                }
                while idx1 < e1.len() && e1[idx1] >= 0. {
                    idx1 +=1;
                }
                while idx2 < e2.len() && e2[idx2] >= 0. {
                    idx2 +=1;
                }

                let total_sets = matches + (idx1 - matches) + (idx2 - matches);
                1. - matches as f32 / total_sets as f32
                //( - matches) as f32 / e1.len() as f32
            }
        }
    }
}

