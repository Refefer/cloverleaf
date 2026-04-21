//! Benchmarks for PolicyEvaluation's softmax weight update.
//!
//! Two bench groups:
//!
//!   * `pe/micro/{degree}` — drive `update_one_node` directly across a range
//!     of per-node degrees. Shows per-node cost as degree grows, which is
//!     useful for spotting regressions in the inner loop itself.
//!
//!   * `pe/macro/size/{nodes}` — run the full
//!     `update_policy_weights_in_place` on Barabási-Albert graphs of varying
//!     size. This is the production path — `par_iter_mut` over nodes with a
//!     sequential per-node body.
//!
//! Typical usage:
//!   cargo bench --bench bench_policy_evaluation -- pe/micro
//!   cargo bench --bench bench_policy_evaluation -- pe/macro/size
//!
//! Memory note: the 5M-node size tier transiently allocates ~800MB during
//! graph generation.

use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rand_distr::{Distribution, Uniform};

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion,
};

use cloverleaf::algos::policy_evaluation::{update_one_node, PolicyEvaluation};
use cloverleaf::graph::{convert_edges_to_cdf, CumCSR, CSR};

const SEED: u64 = 0xC0FFEE_u64;

// ---------------------------------------------------------------------------
// Synthetic graph generation: Barabási-Albert preferential attachment.
// ---------------------------------------------------------------------------

/// Generates an undirected Barabási-Albert graph with `n` nodes.
///
/// Each new node attaches to `m` existing nodes, sampled with probability
/// proportional to current degree via the standard endpoint-multiset trick.
/// Returns edges in `(from, to, weight=1.0)` form with both directions
/// present. The returned vector has length close to `2 * m * n` (minus a
/// few dedup'd self-collisions per step).
fn ba_graph(n: usize, m: usize, seed: u64) -> Vec<(usize, usize, f32)> {
    assert!(m >= 1, "m must be >= 1");
    assert!(n > m, "n must be > m");

    let mut rng = XorShiftRng::seed_from_u64(seed);

    let mut edges: Vec<(usize, usize, f32)> = Vec::with_capacity(2 * m * n);
    let mut endpoints: Vec<usize> = Vec::with_capacity(2 * m * n);

    // Seed clique: m+1 nodes fully connected.
    for i in 0..=m {
        for j in (i + 1)..=m {
            edges.push((i, j, 1.0));
            edges.push((j, i, 1.0));
            endpoints.push(i);
            endpoints.push(j);
        }
    }

    let mut picked: Vec<usize> = Vec::with_capacity(m);

    for new_node in (m + 1)..n {
        picked.clear();
        let tries = m * 2;
        let dist = Uniform::new(0, endpoints.len());
        for _ in 0..tries {
            if picked.len() == m {
                break;
            }
            let idx = dist.sample(&mut rng);
            let target = endpoints[idx];
            if target == new_node || picked.contains(&target) {
                continue;
            }
            picked.push(target);
        }

        for &target in &picked {
            edges.push((new_node, target, 1.0));
            edges.push((target, new_node, 1.0));
            endpoints.push(new_node);
            endpoints.push(target);
        }
    }

    drop(endpoints);
    edges
}

fn random_rewards(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = XorShiftRng::seed_from_u64(seed);
    let dist = Uniform::new(0f32, 1f32);
    (0..n).map(|_| dist.sample(&mut rng)).collect()
}

fn build_power_law_graph(n: usize, m: usize) -> CumCSR {
    let edges = ba_graph(n, m, SEED);
    let csr = CSR::construct_from_edges(edges, true);
    CumCSR::convert(csr)
}

// ---------------------------------------------------------------------------
// Micro-bench: per-node update cost across degrees.
// ---------------------------------------------------------------------------

/// Builds inputs for a single call to `update_one_node` at the given degree.
fn make_single_node_inputs(degree: usize, seed: u64) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
    let edges: Vec<usize> = (0..degree).collect();

    let mut weights: Vec<f32> = (1..=degree).map(|i| i as f32 / degree as f32).collect();
    if let Some(last) = weights.last_mut() {
        *last = 1.0;
    }

    let mut rng = XorShiftRng::seed_from_u64(seed);
    let dist = Uniform::new(-1f32, 1f32);
    let values: Vec<f32> = (0..degree).map(|_| dist.sample(&mut rng)).collect();

    (edges, weights, values)
}

const MICRO_DEGREES: &[usize] = &[
    100, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000, 500_000, 10_000_000,
];

/// Pick a batch size that keeps per-sample memory bounded at huge degrees.
fn micro_batch_size(degree: usize) -> BatchSize {
    if degree >= 1_000_000 {
        BatchSize::LargeInput
    } else {
        BatchSize::SmallInput
    }
}

fn bench_micro(c: &mut Criterion) {
    let mut group = c.benchmark_group("pe/micro");
    group.sample_size(20);

    for &degree in MICRO_DEGREES {
        let (edges, weights, values) = make_single_node_inputs(degree, SEED);

        group.bench_with_input(
            BenchmarkId::from_parameter(degree),
            &degree,
            |b, _| {
                b.iter_batched(
                    || (edges.clone(), weights.clone()),
                    |(mut e, mut w)| {
                        update_one_node(
                            black_box(&mut e),
                            black_box(&mut w),
                            black_box(&values),
                            black_box(1.0),
                        );
                    },
                    micro_batch_size(degree),
                );
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Macro-bench: size sweep on power-law graphs.
// ---------------------------------------------------------------------------

const SIZE_SWEEP: &[usize] = &[500_000, 2_000_000, 5_000_000];
const SIZE_SWEEP_M: usize = 10;

fn bench_macro_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("pe/macro/size");
    group.sample_size(10);

    for &n in SIZE_SWEEP {
        let graph = build_power_law_graph(n, SIZE_SWEEP_M);
        let rewards = random_rewards(n, SEED);
        let pe = PolicyEvaluation::new(0.99, 1, 1e-6, 1.0, false);
        let values = pe.compute(&graph, &rewards);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter_batched(
                || graph.clone(),
                |mut g| {
                    pe.update_policy_weights_in_place(&mut g, black_box(&values));
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

#[allow(dead_code)]
fn _keep_import_live() {
    let mut w = [0.25f32, 0.5, 0.75, 1.0];
    convert_edges_to_cdf(&mut w, Some(1.0));
}

criterion_group!(benches, bench_micro, bench_macro_size);
criterion_main!(benches);
