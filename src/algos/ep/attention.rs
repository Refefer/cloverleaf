use simple_grad::*;
use float_ord::FloatOrd;
use rand::prelude::*;

#[derive(Copy,Clone)]
pub struct MultiHeadedAttention {
    num_heads: usize,
    d_k: usize,
    attention_type: AttentionType
}

#[derive(Copy,Clone)]
pub enum AttentionType {
    Full,
    Sliding { window_size: usize },
    Random { num_features: usize }
}

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

    fn get_value_vec(&self, emb: &ANode) -> ANode {
        let offset = self.num_heads * self.d_k * 2;
        let v = emb.value().len();
        emb.slice(offset, v - offset)
    }

}

fn get_query_vec(emb: &ANode, dims: usize) -> ANode {
    emb.slice(0, dims)
}

fn get_key_vec(emb: &ANode, dims: usize) -> ANode {
    emb.slice(dims, dims)
}

fn get_value_vec(emb: &ANode, dims: usize) -> ANode {
    let v = emb.value().len();
    emb.slice(2*dims, v - 2*dims)
}

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
        let value = mha.get_value_vec(&node);
        Attention {query, key, value}
    }
}

pub fn attention_mean<'a>(
    it: impl Iterator<Item=&'a (ANode, usize)>,
    mha: &MultiHeadedAttention,
    rng: &mut impl Rng
) -> ANode {

    let features = it.collect::<Vec<_>>();
    let mut averages = Vec::with_capacity(mha.num_heads);
    for head in 0..mha.num_heads {
        let items: Vec<_> = features.iter().map(|(node, count)| {
            (Attention::new(node, mha, head), *count)
        }).collect();

        if items.len() == 1 {
            return items[0].0.value.clone()
        }
        
        // Compute attention matrix
        let attention_matrix = compute_attention_matrix(&items, &mha.attention_type, rng);
        
        let sm_att_mat = compute_attention_softmax(attention_matrix, mha.d_k);

        let n = items.len() as f32;
        let mean = scale_vecs(items, &sm_att_mat)
            .collect::<Vec<_>>().sum_all() / n;
        averages.push(mean);
    }

    averages.sum_all() / (mha.num_heads as f32)
}

/// Computes value level attention scaling.
fn scale_vecs<'a>(
    items: Vec<(Attention, usize)>, 
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
    items: &[(Attention, usize)],
    at: &AttentionType,
    rng: &mut impl Rng
) -> AttentionMatrix {
    match at {
        AttentionType::Full => compute_full_attention_matrix(items),
        AttentionType::Sliding{ window_size } => compute_sliding_attention_matrix(items, *window_size),
        AttentionType::Random { num_features } => compute_random_attention_matrix(items, *num_features, rng)
    }
}

fn compute_full_attention_matrix(
    items: &[(Attention, usize)]
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
            if num >= 1 {
                dot_i_j = dot_i_j * (num as f32);
            }
            row[j] = Some(dot_i_j);
        }
    }
    scaled
}

fn compute_sliding_attention_matrix(
    items: &[(Attention, usize)],
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

fn compute_random_attention_matrix(
    items: &[(Attention, usize)],
    k: usize,
    rng: &mut impl Rng
) -> AttentionMatrix {
    
     // Get the attention for each feature
    let mut scaled = vec![vec![None; items.len()]; items.len()];
    let mut buff = vec![0; k.min(items.len())];
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
        let sm = softmax(nz_row);

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

fn softmax(numers: ANode) -> ANode {
    // Doesn't need to be part of the graph
    let max_value = numers.value().iter()
        .max_by_key(|v| FloatOrd(**v))
        .expect("Shouldn't be non-zero!");

    let mv = Constant::scalar(*max_value);
    let n = (numers - &mv).exp();
    &n / n.sum()
}

#[cfg(test)]
mod attention_tests {
    use super::*;
    use rand_xorshift::XorShiftRng;

    fn create_att_vecs() -> Vec<(Attention, usize)> {
        let mha = MultiHeadedAttention {
            d_k: 1,
            num_heads: 1,
            attention_type: AttentionType::Full
        };
        vec![
            (Attention::new(&Variable::new(vec![-1., -1., 1., 1.]), &mha, 0), 1),
            (Attention::new(&Variable::new(vec![0., 0., 2., 2.]), &mha, 0), 1),
            (Attention::new(&Variable::new(vec![1., 1., -1., -1.]), &mha, 0), 1)
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


}
