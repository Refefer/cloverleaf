use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rand_distr::{Distribution,Uniform};

use criterion::{criterion_group, criterion_main, Criterion};
use cloverleaf::{Graph,CSR,CumCSR,FeatureStore};
use cloverleaf::algos::lpa::lpa;
use cloverleaf::algos::ep::EmbeddingPropagation;
use cloverleaf::algos::ep::loss::Loss;
use cloverleaf::algos::ep::model::AveragedFeatureModel;
use cloverleaf::algos::utils::Sample;

const SEED: u64 = 2022341;

/// A complete graph on 1,000 nodes.
fn build_clique_edges() -> Vec<(usize, usize, f32)> {
    let mut edges = Vec::new();
    let max = 1000;
    for ni in 0..max {
        for no in (ni+1)..max {
            edges.push((ni, no, 1f32));
            edges.push((no, ni, 1f32));
        }
    }
    edges
}

fn label_propagation(c: &mut Criterion) {
    let graph = CSR::construct_from_edges(build_clique_edges(), false);
    c.bench_function("lpa", |b| b.iter(|| lpa(&graph, 30, 1234567)));
}

fn generate_random_features(size: usize, num_feats: usize, feat_space: usize) -> FeatureStore {
    let mut feature_store = FeatureStore::new(size);
    let mut rng = XorShiftRng::seed_from_u64(SEED);
    let dist = Uniform::new(0, num_feats);
    let feat_dist = Uniform::new(0, feat_space);
    for node_id in 0..size {
        let nf = dist.sample(&mut rng);
        let feats: Vec<_> = (0..nf)
            .map(|_| ("feat", feat_dist.sample(&mut rng).to_string()))
            .collect();
        feature_store.set_features(node_id, feats.into_iter());
    }
    feature_store
}

fn embedding_propagation(c: &mut Criterion) {
    let ccsr = CumCSR::convert(CSR::construct_from_edges(build_clique_edges(), false));

    for num_feats in [10usize, 25].iter() {
        let mut feature_store = generate_random_features(ccsr.len(), *num_feats, 1000);
        feature_store.fill_missing_nodes();

        let ep = EmbeddingPropagation {
            alpha: 1e-2,
            batch_size: 128,
            d_model: 5,
            passes: 50,
            seed: 202220222,
            indicator: false,
            loss: Loss::MarginLoss(10f32, 1),
            ..EmbeddingPropagation::default()
        };
        let model = AveragedFeatureModel::new(Sample::All, Sample::Fixed(10), false, false);

        let label = format!("ep:{}", num_feats);
        c.bench_function(&label, |b| b.iter(|| ep.learn(&ccsr, &feature_store, None, &model)));
    }
}

criterion_group!(benches, label_propagation, embedding_propagation);
criterion_main!(benches);
