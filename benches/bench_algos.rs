use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rand_distr::{Distribution,Uniform};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cloverleaf::graph::{Graph,CSR,CumCSR};
use cloverleaf::algos::lpa::lpa;
use cloverleaf::algos::ep::{EmbeddingPropagation,Loss};
use cloverleaf::algos::utils::FeatureStore;

const SEED: u64 = 2022341;

fn criterion_benchmark(c: &mut Criterion) {
    // Create edges
    let mut edges = Vec::new();
    let max_nodes = 5000;
    for from_node in 0..max_nodes {
        for to_node in (from_node+1)..max_nodes {
            edges.push((from_node, to_node, 1.));
            edges.push((to_node, from_node, 1.));
        }
    }
    let graph = CSR::construct_from_edges(edges);
    c.bench_function("lpa", |b| b.iter(|| lpa(&graph, 30, 1234567)));
}

fn build_star_edges() -> Vec<(usize, usize, f32)> {
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

fn generate_random_features(size: usize, num_feats: usize, feat_space: usize) -> FeatureStore {
    let mut feature_store = FeatureStore::new(size, "feat".into());
    let mut rng = XorShiftRng::seed_from_u64(SEED);
    let dist = Uniform::new(0, num_feats);
    let feat_dist = Uniform::new(0, feat_space);
    for node_id in 0..size {
        let nf = dist.sample(&mut rng);
        let mut feats = Vec::with_capacity(nf);
        for _ in 0..nf {
            feats.push(format!("{}", feat_dist.sample(&mut rng)));
        }
        feature_store.set_features(node_id, feats);
    }
    feature_store
}

fn embedding_propagation(c: &mut Criterion) {
    let edges = build_star_edges();
    let csr = CSR::construct_from_edges(edges);
    let ccsr = CumCSR::convert(csr);

    for num_feats in [10usize, 25].iter() {
    //for num_feats in [10usize, 25, 75, 100].iter() {
        let mut feature_store = generate_random_features(ccsr.len(), *num_feats, 1000);

        feature_store.fill_missing_nodes();

        let ep = EmbeddingPropagation {
            alpha: 1e-2,
            gamma: 1f32,
            batch_size: 128,
            dims: 5,
            passes: 50,
            seed: 202220222,
            indicator: false,
            max_nodes: Some(10),
            max_features: None,
            wd: 0f32,
            loss: Loss::MarginLoss(10f32, 1)
        };

        let label = format!("ep:{}", num_feats);
        c.bench_function(&label, |b| b.iter(|| ep.learn(&ccsr, &feature_store)));
    }
}

//criterion_group!(benches, criterion_benchmark, embedding_propagation);
criterion_group!(benches, embedding_propagation);
criterion_main!(benches);
