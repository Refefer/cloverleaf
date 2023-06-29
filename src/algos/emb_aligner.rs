use simple_grad::*;

use crate::embeddings::Distance;

pub fn align_embedding(
    embedding: &[f32],
    t_embeddings: &[(&[f32], f32)],
    alpha: f32,
    eps: f32
) -> Vec<f32> {
    let embs: Vec<_> = t_embeddings.iter().map(|(e, d)| {
        Constant::new(e.to_vec())
    }).collect();

    let mut new_emb = embedding.to_vec();
    let mut last_err = std::f32::INFINITY;
    loop {
        let ne = Variable::pooled(new_emb.as_slice());

        let buff: Vec<_> = embs.iter().zip(t_embeddings.iter()).enumerate().map(|(pos, (e, (_, d)))| {
            let euc_dist = (e.clone() - &ne).pow(2.).sum().pow(0.5);
            (euc_dist - *d).pow(2.) / ((pos + 1) as f32).sqrt()
        }).collect();

        let loss = buff.sum_all();

        let mut graph = Graph::new();
        graph.backward(&loss);

        let grad = graph.get_grad(&ne).unwrap();
        new_emb.iter_mut().zip(grad.iter()).for_each(|(ei, gi)| {
            *ei -= alpha * *gi;          
        });

        let cur_err = loss.value()[0];
        if (last_err - cur_err).abs() / last_err < eps {
            break
        }
        last_err = cur_err;
    }
    new_emb
}
