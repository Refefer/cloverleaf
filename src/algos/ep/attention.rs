//! This module constructs a node embedding from a set of features through various different
//! attention methods.  Notably, we use averaged versus concat -> linear projection due to
//! performance cost.  This has representational downsides but performs significantly faster and
//! with no extra parmeters.
use candle_core::{Device, Tensor};
use rand::prelude::*;

use crate::algos::utils::Sample;

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
    Full,
    Sliding { window_size: usize },
    Random { num_features: Sample }
}

impl MultiHeadedAttention {
    pub fn new(num_heads: usize, d_k: usize, attention_type: AttentionType) -> Self {
        MultiHeadedAttention { num_heads, d_k, attention_type }
    }

    pub fn preserve_feature_order(&self) -> bool {
        matches!(self.attention_type,  AttentionType::Sliding {window_size:_})
    }

    fn get_query_vec(&self, emb: &Tensor, head_num: usize) -> Tensor {
        let start = self.d_k * head_num;
        emb.narrow(0, start, self.d_k).unwrap()
    }

    fn get_key_vec(&self, emb: &Tensor, head_num: usize) -> Tensor {
        let start = (self.num_heads * self.d_k) + self.d_k * head_num;
        emb.narrow(0, start, self.d_k).unwrap()
    }

    fn get_value_vec(&self, emb: &Tensor, head_num: usize) -> Tensor {
        let query_key_size = self.num_heads * self.d_k * 2;
        let v = emb.dims1().unwrap();
        let d_model = ((v - query_key_size) as f32 / self.num_heads as f32) as usize;
        let start = query_key_size + d_model * head_num;
        emb.narrow(0, start, d_model).unwrap()
    }
}

#[derive(Clone)]
struct Attention {
    query: Tensor,
    key: Tensor,
    value: Tensor
}

impl Attention {
    fn new(node: &Tensor, mha: &MultiHeadedAttention, head: usize) -> Self {
        let query = mha.get_query_vec(&node, head);
        let key = mha.get_key_vec(&node, head);
        let value = mha.get_value_vec(&node, head);
        Attention {query, key, value}
    }

    fn scale(&self, scalar: f32, device: &Device) -> Attention {
        if scalar != 1f32 {
            let scalar_t = Tensor::from_slice(&[scalar], 1usize, device).unwrap();
            Attention {
                query: scalar_t.mul(&self.query).unwrap(),
                key: scalar_t.mul(&self.key).unwrap(),
                value: scalar_t.mul(&self.value).unwrap()
            }
        } else {
            self.clone()
        }
    }
}

pub fn attention_mean<'a>(
    it: impl Iterator<Item=&'a (Tensor, f32)>,
    mha: &MultiHeadedAttention,
    rng: &mut impl Rng
) -> Tensor {
    let device = Device::Cpu;
    let features = it.collect::<Vec<_>>();
    let mut averages = Vec::with_capacity(mha.num_heads);
    for head in 0..mha.num_heads {
        let items: Vec<_> = features.iter().map(|(node, count)| {
            (Attention::new(node, mha, head), *count)
        }).collect();

        if items.len() == 1 {
            return items[0].0.value.clone();
        }
        
        let attention_matrix = compute_attention_matrix(&items, &mha.attention_type, rng, &device);
        let sm_att_mat = compute_attention_softmax(attention_matrix, mha.d_k, &device);

        let n = items.len() as f32;
        let mean = scale_vecs(items, &sm_att_mat, &device)
            .collect::<Vec<_>>().iter()
            .cloned()
            .reduce(|a, b| a.add(&b).unwrap())
            .unwrap();
        let n_t = Tensor::from_slice(&[n], 1usize, &device).unwrap();
        let mean = mean.div(&n_t).unwrap();

        averages.push(mean);
    }

    let num_heads_t = Tensor::from_slice(&[mha.num_heads as f32], 1usize, &device).unwrap();
    averages.iter()
        .cloned()
        .reduce(|a, b| a.add(&b).unwrap())
        .unwrap()
        .div(&num_heads_t)
        .unwrap()
}

fn scale_vecs<'a>(
    items: Vec<(Attention, f32)>, 
    sm_att_mat: &'a AttentionMatrix,
    device: &Device
) -> impl Iterator<Item=Tensor> + 'a {
    let mut rows = vec![Vec::new(); sm_att_mat.len()];
    sm_att_mat.iter().enumerate().for_each(|(ri, row)| {
        items.iter().zip(row.iter()).for_each(|((att, _), scaled_v)| {
            if let Some(v) = scaled_v {
                rows[ri].push(att.value.mul(v).unwrap());
            }
        });
    });

    rows.into_iter().map(|sums| {
        sums.iter().cloned().reduce(|a, b| a.add(&b).unwrap()).unwrap()
    })
}

type AttentionMatrix = Vec<Vec<Option<Tensor>>>;

fn compute_attention_matrix(
    items: &[(Attention, f32)],
    at: &AttentionType,
    rng: &mut impl Rng,
    device: &Device
) -> AttentionMatrix {
    match at {
        AttentionType::Full => compute_full_attention_matrix(items, device),
        AttentionType::Sliding{ window_size } => {
            compute_sliding_attention_matrix(items, *window_size, device)
        },
        AttentionType::Random { num_features } => {
            compute_random_attention_matrix(items, *num_features, rng, device)
        }
    }
}

fn compute_full_attention_matrix(items: &[(Attention, f32)], device: &Device) -> AttentionMatrix {
    let mut scaled = vec![vec![None; items.len()]; items.len()];
    for i in 0..items.len() {
        let (at_i, ic) = &items[i];
        let row = &mut scaled[i];
        for j in 0..items.len() {
            let (at_j, jc) = &items[j];
            let dot_i_j = at_i.query.dot(&at_j.key).unwrap();
            let num = ic * jc;
            if num >= 1f32 {
                let num_t = Tensor::from_slice(&[num], 1usize, device).unwrap();
                row[j] = Some(dot_i_j.mul(&num_t).unwrap());
            } else {
                row[j] = Some(dot_i_j.clone());
            }
        }
    }
    scaled
}

fn compute_sliding_attention_matrix(items: &[(Attention, f32)], window: usize, device: &Device) -> AttentionMatrix {
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
            row[j] = Some(at_i.query.dot(&at_j.key).unwrap());
        }
    }
    scaled
}

fn compute_random_attention_matrix(items: &[(Attention, f32)], sample: Sample, rng: &mut impl Rng, device: &Device) -> AttentionMatrix {
    let (k, scale) = sample.sample(items.len(), true, rng);
    let items = items.iter()
        .map(|(at, w)| (at.scale(scale, device), w))
        .collect::<Vec<_>>();

    let mut scaled = vec![vec![None; items.len()]; items.len()];
    let mut buff = vec![0; k];
    for i in 0..items.len() {
        let at_i = &items[i].0;
        let row = &mut scaled[i];
        items.iter().enumerate().map(|(i,_)| i).choose_multiple_fill(rng, buff.as_mut_slice());
        for j in buff.iter() {
            let at_j = &items[*j].0;
            row[*j] = Some(at_i.query.dot(&at_j.key).unwrap());
        }
    }
    scaled
}

fn compute_attention_softmax(mut attention_matrix: AttentionMatrix, d_k: usize, device: &Device) -> AttentionMatrix {
    let d_k_sqrt = (d_k as f32).sqrt();
    let d_k_t = Tensor::from_slice(&[d_k_sqrt], 1usize, device).unwrap();

    attention_matrix.iter_mut().for_each(|row| {
        let non_zero_row: Vec<_> = row.iter()
            .filter(|x| x.is_some())
            .map(|x| x.clone().unwrap())
            .collect();

        let nz_row = Tensor::cat(&non_zero_row, 0).unwrap().div(&d_k_t).unwrap();
        let exp_vals = nz_row.exp().unwrap();
        let sum_exp = exp_vals.sum_all().unwrap();
        let sm = exp_vals.div(&sum_exp).unwrap();

        let mut idx = 0;
        row.iter_mut().for_each(|ri| {
            if ri.is_some() {
                *ri = Some(sm.narrow(0, idx, 1).unwrap());
                idx += 1;
            }
        });
    });
    attention_matrix
}

pub fn softmax(numers: &Tensor, _exact: bool) -> Tensor {
    let exp_vals = numers.exp().unwrap();
    let sum_exp = exp_vals.sum_all().unwrap();
    exp_vals.div(&sum_exp).unwrap()
}

#[cfg(test)]
mod attention_tests {
    use super::*;
    use rand_xorshift::XorShiftRng;

    fn create_att_vecs(device: &Device) -> Vec<(Attention, f32)> {
        let mha = MultiHeadedAttention {
            d_k: 1,
            num_heads: 1,
            attention_type: AttentionType::Full
        };
        vec![
            (Attention::new(&Tensor::from_slice(&[-1., -1., 1., 1.], 4usize, device).unwrap(), &mha, 0), 1f32),
            (Attention::new(&Tensor::from_slice(&[0., 0., 2., 2.], 4usize, device).unwrap(), &mha, 0), 1f32),
            (Attention::new(&Tensor::from_slice(&[1., 1., -1., -1.], 4usize, device).unwrap(), &mha, 0), 1f32)
        ]
    }

    #[test]
    fn test_attention_matrix_global() {
        let device = Device::Cpu;
        let feats = create_att_vecs(&device);
        let exp_att_matrix = vec![
            vec![vec![-1. * -1.], vec![-1. * 0.], vec![-1. * 1.]],
            vec![vec![0.  * -1.], vec![0.  * 0.], vec![0.  * 1.]],
            vec![vec![1.  * -1.], vec![1.  * 0.], vec![1.  * 1.]],
        ];

        let mut rng = XorShiftRng::seed_from_u64(0);
        let att_matrix = compute_attention_matrix(&feats, &AttentionType::Full, &mut rng, &device);
        for (row, exp_row) in att_matrix.into_iter().zip(exp_att_matrix.into_iter()) {
            for (ri, eri) in row.into_iter().zip(exp_row.into_iter()) {
                if let Some(v) = ri {
                    let vals: Vec<f32> = v.to_vec1().unwrap();
                    assert!((vals[0] - eri[0]).abs() < 1e-5);
                }
            }
        }
    }
}
