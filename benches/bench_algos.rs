use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cloverleaf::graph::{Graph,CSR,CumCSR};
use cloverleaf::algos::lpa::lpa;
use cloverleaf::algos::ep::{EmbeddingPropagation,FeatureStore};

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

fn embedding_propagation(c: &mut Criterion) {
    let edges = build_star_edges();
    let csr = CSR::construct_from_edges(edges);
    let ccsr = CumCSR::convert(csr);

    let mut feature_store = FeatureStore::new(ccsr.len());
    feature_store.fill_missing_nodes();

    let ep = EmbeddingPropagation {
        alpha: 1e-2,
        gamma: 1f32,
        batch_size: 128,
        dims: 5,
        passes: 50,
        seed: 202220222
    };

    c.bench_function("embedding_propagation", |b| b.iter(|| ep.learn(&ccsr, &feature_store)));
}

//criterion_group!(benches, criterion_benchmark, embedding_propagation);
criterion_group!(benches, embedding_propagation);
criterion_main!(benches);
