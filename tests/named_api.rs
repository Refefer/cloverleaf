//! The Rust API end to end: the same workflow the Python bindings offer, without Python.

use std::collections::HashSet;
use std::sync::Arc;

use cloverleaf::algos::aggregator::{FeatureAggregator, NodeEmbedder};
use cloverleaf::algos::alignment::NeighborhoodAligner;
use cloverleaf::algos::ep::EmbeddingPropagation;
use cloverleaf::algos::ep::model::AveragedFeatureModel;
use cloverleaf::algos::lsr::{LSR, TournamentBuilder};
use cloverleaf::algos::pagerank::PageRank;
use cloverleaf::algos::utils::{Sample, top_k_scores};
use cloverleaf::io::read_node_values;
use cloverleaf::{Distance, EdgeType, Error, FeatureNamespace, FeatureSet, GraphBuilder,
                 GraphLoadOptions, NamedGraph, NodeEmbeddings, NodeEmbeddingsBuilder};

fn temp_path(name: &str) -> String {
    std::env::temp_dir().join(format!("cloverleaf-named-{}-{}", std::process::id(), name))
        .to_str().unwrap().to_string()
}

/// Two triangles joined by one edge.
fn two_triangles() -> NamedGraph {
    let mut gb = GraphBuilder::new();
    for (a, b) in [("a", "b"), ("b", "c"), ("c", "a"), ("x", "y"), ("y", "z"), ("z", "x")] {
        gb.add_edge(("n", a), ("n", b), 1., EdgeType::Undirected);
    }
    gb.add_edge(("n", "c"), ("m", "x"), 1., EdgeType::Undirected);
    gb.build(true).unwrap()
}

#[test]
fn graph_save_load_round_trip() {
    let g = two_triangles();
    let path = temp_path("graph.edges");
    g.save(&path, None).unwrap();
    let loaded = NamedGraph::load(&path, EdgeType::Directed, &GraphLoadOptions::default()).unwrap();
    std::fs::remove_file(&path).unwrap();

    assert_eq!(loaded.nodes(), g.nodes());
    assert_eq!(loaded.edges(), g.edges());
    let (edges, weights) = loaded.get_edges("n", "c", true).unwrap();
    let names: HashSet<_> = edges.iter().map(|id| loaded.vocab().get_name_owned(*id).unwrap()).collect();
    assert_eq!(names, [("n", "a"), ("n", "b"), ("m", "x")].iter()
        .map(|(t, n)| (t.to_string(), n.to_string())).collect());
    assert!(weights.iter().all(|w| (w - 1. / 3.).abs() < 1e-6));
    assert!(matches!(loaded.get_edges("n", "missing", true), Err(Error::UnknownNode { .. })));
}

#[test]
fn feature_learning_workflow() {
    let g = two_triangles();
    let path = temp_path("features.txt");
    std::fs::write(&path, "n\ta\tcolor:red\nn\tb\tcolor:red\nn\tx\tcolor:blue\n").unwrap();
    let mut features = FeatureSet::from_graph(&g);
    features.load_into(&path, &FeatureNamespace::prefix(":").unwrap()).unwrap();
    std::fs::remove_file(&path).unwrap();

    let ep = EmbeddingPropagation { d_model: 8, passes: 3, indicator: false, valid_pct: 0.0,
                                    ..EmbeddingPropagation::default() };
    let model = AveragedFeatureModel::new(Sample::All, Sample::All, false, false);
    let feature_embeddings = ep.learn_features(&g, &mut features, None, &model).unwrap();
    // Featureless nodes got a feature of their own.
    assert_eq!(feature_embeddings.len(), features.num_features());
    assert_eq!(feature_embeddings.dims(), 8);

    // Warm starting from the result continues training.
    let again = ep.learn_features(&g, &mut features, Some(feature_embeddings.clone()), &model).unwrap();
    assert_eq!(again.len(), feature_embeddings.len());

    let embedder = NodeEmbedder::new(FeatureAggregator::averaged());
    let nodes = embedder.embed_feature_set(&features, &feature_embeddings).unwrap();
    assert!(Arc::ptr_eq(nodes.vocab(), g.vocab()));
    assert_eq!(nodes.dims(), 8);

    let aligned = NeighborhoodAligner::new(Some(0.5), None).align_all(&nodes, &g).unwrap();
    assert_eq!(aligned.len(), g.nodes());

    let emb_path = temp_path("nodes.emb");
    aligned.save(&emb_path, None).unwrap();
    let loaded = NodeEmbeddings::load(&emb_path, Distance::Cosine, None, None, None).unwrap();
    std::fs::remove_file(&emb_path).unwrap();
    let query = aligned.get_embedding("n", "a").unwrap().to_vec();
    let nearest = loaded.nearest_neighbor(&query, 1, None);
    assert_eq!(loaded.vocab().get_name_owned(*nearest[0].to_tup().0).unwrap(),
               ("n".to_string(), "a".to_string()));
}

/// Features built over a different vocab are rejected rather than silently misindexed.
#[test]
fn mismatched_vocabs_are_errors() {
    let g = two_triangles();
    let mut gb = GraphBuilder::new();
    gb.add_edge(("q", "1"), ("q", "2"), 1., EdgeType::Directed);
    let mut other = FeatureSet::from_graph(&gb.build(false).unwrap());
    let ep = EmbeddingPropagation { indicator: false, ..EmbeddingPropagation::default() };
    let model = AveragedFeatureModel::new(Sample::All, Sample::All, false, false);
    assert!(matches!(ep.learn_features(&g, &mut other, None, &model), Err(Error::VocabMismatch(_))));

    let mut b = NodeEmbeddingsBuilder::new(Distance::Cosine);
    b.add_embedding("q", "1", &[1., 0.]).unwrap();
    let wrong = b.build().unwrap();
    assert!(matches!(NeighborhoodAligner::new(None, None).align_all(&wrong, &g),
                     Err(Error::VocabMismatch(_))));
}

#[test]
fn ranking() {
    let g = two_triangles();
    let ranks = PageRank::with_iterations(20).learn(&g, false);
    let scores = (0..ranks.len()).map(|id| (id, ranks.store().get_embedding(id)[0]));
    let only_n = HashSet::from(["n".to_string()]);
    let top = top_k_scores(scores, Some(2), |id| g.vocab().has_node_type(id, &only_n));
    assert_eq!(top.len(), 2);
    assert!(top.iter().all(|(id, _)| g.vocab().get_node_type(*id).unwrap().as_str() == "n"));
    assert!(top[0].1 >= top[1].1);

    let mut tb = TournamentBuilder::new();
    tb.add_ranked_outcomes(&[("p", "best"), ("p", "mid"), ("p", "worst")], 1.0);
    let scores = LSR { passes: 30 }.learn(&tb.build().unwrap(), false);
    let s = |n: &str| scores.get_embedding("p", n).unwrap()[0];
    assert!(s("best") > s("mid") && s("mid") > s("worst"));
}

#[test]
fn node_values_file() {
    let g = two_triangles();
    let path = temp_path("rewards.txt");
    std::fs::write(&path, "n\ta\t1.5\nm\tx\t-2\n").unwrap();
    let values = read_node_values(&path, g.vocab()).unwrap();
    std::fs::write(&path, "n\tmissing\t1\n").unwrap();
    let missing = read_node_values(&path, g.vocab());
    std::fs::remove_file(&path).unwrap();
    assert_eq!(values, vec![(g.node_id("n", "a").unwrap(), 1.5), (g.node_id("m", "x").unwrap(), -2.)]);
    assert!(matches!(missing, Err(Error::UnknownNode { .. })));
}

/// In-place optimisation copies a shared graph instead of changing it under the other owner.
#[test]
fn policy_evaluation_copies_shared_graphs() {
    use cloverleaf::algos::policy_evaluation::PolicyEvaluation;
    let mut g = two_triangles();
    let shared = g.clone();
    assert!(g.graph_mut().is_none());
    let mut rewards = vec![0f32; g.nodes()];
    rewards[g.node_id("n", "a").unwrap()] = 1.;
    let pe = PolicyEvaluation { indicator: false, ..PolicyEvaluation::default() };
    pe.optimize(&mut g, &rewards).unwrap();
    assert!(!Arc::ptr_eq(g.shared_graph(), shared.shared_graph()));
    assert!(g.graph_mut().is_some());
    assert!(pe.optimize(&mut g, &rewards[1..]).is_err());
}
