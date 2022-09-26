use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cloverleaf::graph::{Graph,CSR};
use cloverleaf::algos::lpa::lpa;

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

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
