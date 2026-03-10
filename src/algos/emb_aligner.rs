use candle_core::{Device, Tensor, Var};

pub fn align_embedding(
    embedding: &[f32],
    t_embeddings: &[(&[f32], f32)],
    alpha: f32,
    eps: f32,
    max_epochs: usize,
) -> Vec<f32> {
    let device = Device::Cpu;

    let embs: Vec<_> = t_embeddings
        .iter()
        .map(|(e, _d)| Tensor::from_slice(e, e.len(), &device).unwrap())
        .collect();

    let mut new_emb = embedding.to_vec();
    let mut last_err = std::f32::INFINITY;
    let mut i = 0;
    loop {
        // Put the embedding in a variable for autograd
        let ne = Var::from_slice(new_emb.as_slice(), new_emb.len(), &device).unwrap();

        // Compute the distances between the current embedding and the neighbor embeddings
        // we weight it based on position of the anchors
        let buff: Vec<_> = embs
            .iter()
            .zip(t_embeddings.iter())
            .enumerate()
            .map(|(pos, (e, (_, d)))| {
                let diff = ne.as_tensor().sub(e).unwrap();
                let euc_dist = diff.powf(2.0).unwrap().sum_all().unwrap().sqrt().unwrap();
                let d_tensor = Tensor::from_slice(&[*d], 1usize, &device).unwrap();
                let diff_d = euc_dist.sub(&d_tensor).unwrap();
                let divisor =
                    Tensor::from_slice(&[((pos + 1) as f64).sqrt()], 1usize, &device).unwrap();
                diff_d.powf(2.0).unwrap().div(&divisor).unwrap()
            })
            .collect();

        let loss = buff
            .iter()
            .map(|t| t.clone())
            .reduce(|a, b| a.add(&b).unwrap())
            .unwrap();

        let grad_store = loss.backward().unwrap();

        // Simple Backpropagation
        let grad = grad_store.get(&ne).unwrap();
        let grad_vec = grad.to_vec1::<f32>().unwrap();
        new_emb
            .iter_mut()
            .zip(grad_vec.iter())
            .for_each(|(ei, gi)| {
                *ei -= alpha * *gi;
            });

        let cur_err = loss.to_vec0::<f32>().unwrap();
        if (last_err - cur_err).abs() / last_err < eps {
            break;
        }
        last_err = cur_err;
        i += 1;
        if i >= max_epochs {
            break;
        }
    }
    new_emb
}
