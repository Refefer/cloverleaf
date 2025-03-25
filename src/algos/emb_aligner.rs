use simple_grad::*;

pub fn align_embedding(
    embedding: &[f32],
    t_embeddings: &[(&[f32], f32)],
    alpha: f32,
    eps: f32,
    max_epochs: usize
) -> Vec<f32> {
    use_shared_pool(false);

    let embs: Vec<_> = t_embeddings.iter().map(|(e, _d)| {
        Constant::new(e.to_vec())
    }).collect();

    let mut new_emb = embedding.to_vec();
    let mut last_err = std::f32::INFINITY;
    let mut i = 0;
    loop {
        // Put the embedding in a variable for graph
        let ne = Variable::pooled(new_emb.as_slice());

        // Compute the distances between the current embedding and the neighbor embeddings
        // we weight it based on position of the anchors
        let buff: Vec<_> = embs.iter().zip(t_embeddings.iter()).enumerate().map(|(pos, (e, (_, d)))| {
            let euc_dist = (e.clone() - &ne).pow(2f32).sum().sqrt();
            (euc_dist - *d).pow(2.) / ((pos + 1) as f32).sqrt()
        }).collect();

        let loss = buff.sum_all();

        let mut graph = Graph::new();
        graph.backward(&loss);

        // Simple Backpropagation
        let grad = graph.get_grad(&ne).unwrap();
        new_emb.iter_mut().zip(grad.iter()).for_each(|(ei, gi)| {
            *ei -= alpha * *gi;          
        });

        let cur_err = loss.value()[0];
        if (last_err - cur_err).abs() / last_err < eps {
            break
        }
        last_err = cur_err;
        i += 1;
        if i >= max_epochs { break }
    }
    new_emb
}
