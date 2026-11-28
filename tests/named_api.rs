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

/// Walks, ranked and named, the way the Python walkers return them.
#[test]
fn walks_by_name() {
    use cloverleaf::algos::rwr::{RWR, SparsePpr};
    let g = two_triangles();
    let start = g.node_id("n", "a").unwrap();
    let rwr = RWR::from_restarts(0.3, 1000).unwrap();
    let only_n = HashSet::from(["n".to_string()]);
    for weighted in [true, false] {
        let top = g.vocab().top_k_named(rwr.walk_scores(g.graph(), start, weighted), Some(3), Some(&only_n));
        assert_eq!(top.len(), 3);
        assert!(top.iter().all(|((node_type, _), _)| node_type == "n"));
        assert!(top.windows(2).all(|w| w[0].1 >= w[1].1));
    }
    assert!(matches!(RWR::from_restarts(0., 10), Err(Error::InvalidArgument(_))));

    let ppr = SparsePpr::new(0.2, SparsePpr::DEFAULT_EPS).unwrap();
    let top = g.vocab().top_k_named(ppr.compute(g.graph(), start), Some(1), None);
    assert_eq!(top[0].0, ("n".to_string(), "a".to_string()));
    assert!(matches!(SparsePpr::new(1., 1e-5), Err(Error::InvalidArgument(_))));
}

/// Graph embedders name their results by the graph's nodes.
#[test]
fn graph_embedders() {
    use cloverleaf::algos::connected::connected_components;
    use cloverleaf::algos::instantembedding::{Estimator, InstantEmbeddings};
    let g = two_triangles();
    let components = connected_components(&g).unwrap();
    assert!(Arc::ptr_eq(components.vocab(), g.vocab()));
    assert_eq!(components.get_embedding("n", "a").unwrap(), components.get_embedding("m", "x").unwrap());

    let estimator = Estimator::random_walk(20, 2., Estimator::DEFAULT_BETA, 1).unwrap();
    let ie = InstantEmbeddings { estimator, dims: 4, hashes: 2 }.learn(&g).unwrap();
    assert_eq!((ie.len(), ie.dims()), (g.nodes(), 4));
}

/// Embeddings and features must index the graph's nodes: these list them in another order.
#[test]
fn graph_ann_and_ppr_embed_reject_misaligned_inputs() {
    use cloverleaf::Query;
    use cloverleaf::algos::graph_ann::{GraphAnn, GRAPH_ANN_SEED};
    use cloverleaf::algos::pprembed::PPREmbed;
    let g = two_triangles();
    let names: Vec<_> = g.vocab().iter().map(|(t, n)| (t.to_string(), n.to_string())).collect();
    let embeddings = |order: &mut dyn Iterator<Item=&(String, String)>| {
        let mut b = NodeEmbeddingsBuilder::new(Distance::Euclidean);
        for (i, (t, n)) in order.enumerate() {
            b.add_embedding(t, n, &[i as f32]).unwrap();
        }
        b.build().unwrap()
    };

    let ann = GraphAnn::new(&g, GraphAnn::DEFAULT_MAX_STEPS);
    let query = Query::Node("n".into(), "a".into());
    let aligned = embeddings(&mut names.iter());
    let found = ann.find(&query, &aligned, 2, GRAPH_ANN_SEED).unwrap();
    assert_eq!(found.len(), 2);
    let reordered = embeddings(&mut names.iter().rev());
    assert!(matches!(ann.find(&query, &reordered, 2, GRAPH_ANN_SEED), Err(Error::VocabMismatch(_))));

    let ppr = PPREmbed::new(8, 10, 0.5, PPREmbed::DEFAULT_BETA, PPREmbed::DEFAULT_EPS, 1).unwrap();
    let mut features = FeatureSet::from_graph(&g);
    assert_eq!(ppr.learn_features(&g, &mut features).unwrap().len(), g.nodes());
    let mut gb = GraphBuilder::new();
    gb.add_edge(("q", "1"), ("q", "2"), 1., EdgeType::Directed);
    let mut other = FeatureSet::from_graph(&gb.build(false).unwrap());
    assert!(matches!(ppr.learn_features(&g, &mut other), Err(Error::VocabMismatch(_))));
}

/// Named rewards, and the checks the Python `optimize` makes: rewards built from this very graph,
/// and no silent copy of a shared graph unless allowed.
#[test]
fn policy_evaluation_with_named_rewards() {
    use cloverleaf::algos::policy_evaluation::{NodeRewards, PolicyEvaluation};
    let mut g = two_triangles();
    let mut rewards = NodeRewards::new(&g);
    rewards.set("n", "a", 1.).unwrap();
    let pe = PolicyEvaluation { indicator: false, ..PolicyEvaluation::default() };

    let shared = g.clone();
    assert!(matches!(pe.optimize_rewards(&mut g, &rewards, false), Err(Error::WouldCopyGraph)));
    pe.optimize_rewards(&mut g, &rewards, true).unwrap();
    assert!(!Arc::ptr_eq(g.shared_graph(), shared.shared_graph()));
    pe.optimize_rewards(&mut g, &rewards, false).unwrap();

    // An identical graph built separately has its own vocab.
    let separate = NodeRewards::new(&two_triangles());
    assert!(matches!(pe.optimize_rewards(&mut g, &separate, true), Err(Error::VocabMismatch(_))));
}

/// FastRP from structure alone and with features; the inputs are checked.
#[test]
fn fastrp_embeddings() {
    use cloverleaf::algos::fastrp::FastRP;
    let g = two_triangles();
    let frp = FastRP { indicator: false, ..FastRP::new(16) };
    let embeddings = frp.learn(&g).unwrap();
    assert!(Arc::ptr_eq(embeddings.vocab(), g.vocab()));
    assert_eq!((embeddings.len(), embeddings.dims()), (g.nodes(), 16));
    assert!(matches!(embeddings.distance(), Distance::Cosine));

    let mut features = FeatureSet::from_graph(&g);
    features.set_features("n", "a", [("color", "red")].into_iter()).unwrap();
    let with_features = frp.learn_with_features(&g, &features, 8).unwrap();
    assert_eq!(with_features.dims(), 16);

    let mut other = GraphBuilder::new();
    other.add_edge(("n", "q"), ("n", "r"), 1., EdgeType::Undirected);
    let misaligned = FeatureSet::from_graph(&other.build(true).unwrap());
    assert!(matches!(frp.learn_with_features(&g, &misaligned, 8), Err(Error::VocabMismatch(_))));
    assert!(matches!(FastRP { weights: vec![], ..frp }.learn(&g), Err(Error::InvalidArgument(_))));
}

/// VPCG over users and items, from features to embeddings.
#[test]
fn vpcg_feature_mappings() {
    use cloverleaf::algos::vpcg::VPCG;
    let mut gb = GraphBuilder::new();
    for (u, i) in [("u1", "i1"), ("u1", "i2"), ("u2", "i2")] {
        gb.add_edge(("user", u), ("item", i), 1., EdgeType::Undirected);
    }
    let g = gb.build(true).unwrap();
    let mut features = FeatureSet::from_graph(&g);
    features.set_features("item", "i1", [("w", "red")].into_iter()).unwrap();
    features.set_features("item", "i2", [("w", "blue")].into_iter()).unwrap();

    let users = HashSet::from(["user".to_string()]);
    let mappings = VPCG::new(10, 2).learn_named(&g, &mut features, &users).unwrap();
    let mut u1: Vec<_> = mappings.get("user", "u1").unwrap().iter()
        .map(|(f, _)| features.feature_name(*f).unwrap().1).collect();
    u1.sort();
    assert_eq!(u1, vec!["blue", "red"]);
    let embeddings = mappings.into_embeddings(8).unwrap();
    assert!(Arc::ptr_eq(embeddings.vocab(), g.vocab()));
    assert_eq!((embeddings.len(), embeddings.dims()), (g.nodes(), 8));

    let mut misaligned = FeatureSet::from_graph(&two_triangles());
    assert!(matches!(VPCG::new(10, 2).learn_named(&g, &mut misaligned, &users), Err(Error::VocabMismatch(_))));
}

/// SMCI with named pair rewards returns a reweighted copy of the graph.
#[test]
fn smci_pair_rewards() {
    use cloverleaf::algos::smci::{PairRewards, SupervisedMCIteration};
    let g = two_triangles();
    let mut rewards = PairRewards::new(&g);
    rewards.add(("n", "a"), ("m", "x"), 1.).unwrap();
    assert!(matches!(rewards.add(("n", "a"), ("m", "zzz"), 1.), Err(Error::UnknownNode { .. })));
    let smci = SupervisedMCIteration::new(3, 10, 0.5, 0.9, 0., 0.1, 0.2);
    let learned = smci.learn_rewards(&g, &rewards, None).unwrap();
    assert!(Arc::ptr_eq(learned.vocab(), g.vocab()));
    assert_eq!(learned.edges(), g.edges());
}

/// Feature propagation fills featureless nodes from their neighbours.
#[test]
fn feature_propagation() {
    use cloverleaf::algos::feat_propagation::{DEFAULT_MAX_ITERS, DEFAULT_THRESHOLD};
    let g = two_triangles();
    let mut features = FeatureSet::from_graph(&g);
    features.set_features("n", "a", [("color", "red")].into_iter()).unwrap();
    features.propagate(&g, 2, DEFAULT_THRESHOLD, DEFAULT_MAX_ITERS).unwrap();
    for n in ["b", "c"] {
        assert_eq!(features.get_features("n", n).unwrap(), vec![("color".to_string(), "red".to_string())]);
    }
    assert_eq!(features.feature_name(0), Some(("color".to_string(), "red".to_string())));
    assert_eq!(features.feature_name(1), None);

    let mut gb = GraphBuilder::new();
    gb.add_edge(("q", "1"), ("q", "2"), 1., EdgeType::Directed);
    let mut other = FeatureSet::from_graph(&gb.build(false).unwrap());
    assert!(matches!(other.propagate(&g, 2, 0., 1), Err(Error::VocabMismatch(_))));
}

#[test]
fn node_embeddings_from_list() {
    let ne = NodeEmbeddings::from_list(vec![(("n", "a"), vec![1., 2.]), (("n", "b"), vec![3., 4.])],
                                       Distance::Euclidean).unwrap();
    assert_eq!(ne.get(1), Some((("n".to_string(), "b".to_string()), &[3f32, 4.][..])));
    assert_eq!(ne.get(2), None);
    let empty = NodeEmbeddings::from_list(Vec::<((&str, &str), Vec<f32>)>::new(), Distance::Cosine).unwrap();
    assert_eq!((empty.len(), empty.dims()), (0, 0));
    let ragged = NodeEmbeddings::from_list([(("n", "a"), [1f32, 2.].as_slice()), (("n", "b"), &[1.][..])],
                                           Distance::Cosine);
    assert!(matches!(ragged, Err(Error::DimensionMismatch(_))));
}
