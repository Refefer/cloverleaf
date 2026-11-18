//! Learns node embeddings with Embedding Propagation and writes them to a file.
//!
//! ```text
//! cargo run --release --example embed -- GRAPH_FILE OUTPUT_FILE [FEATURE_FILE]
//! ```
//!
//! The graph file has one edge per line, `from_type<TAB>from_name<TAB>to_type<TAB>to_name<TAB>weight`,
//! and may be gzipped.  The optional feature file has one node per line,
//! `node_type<TAB>name<TAB>f1 f2 ...`; nodes without features get one of their own, so without a
//! feature file EP learns one embedding per node.  The output has one node per line,
//! `type<TAB>name<TAB>[e1,e2,...]`, which `NodeEmbeddings::load` reads back.

use std::error::Error;

use cloverleaf::algos::aggregator::{FeatureAggregator, NodeEmbedder};
use cloverleaf::algos::ep::EmbeddingPropagation;
use cloverleaf::algos::ep::loss::Loss;
use cloverleaf::algos::ep::model::AveragedFeatureModel;
use cloverleaf::algos::utils::Sample;
use cloverleaf::{EdgeType, FeatureNamespace, FeatureSet, GraphLoadOptions, NamedGraph};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (graph_path, output_path, feature_path) = match &args[..] {
        [_, g, o] => (g, o, None),
        [_, g, o, f] => (g, o, Some(f)),
        _ => return Err("usage: embed GRAPH_FILE OUTPUT_FILE [FEATURE_FILE]".into())
    };

    let graph = NamedGraph::load(graph_path, EdgeType::Undirected, &GraphLoadOptions::default())?;
    eprintln!("loaded {} nodes and {} edges", graph.nodes(), graph.edges());

    let mut features = FeatureSet::from_graph(&graph);
    if let Some(path) = feature_path {
        features.load_into(path, &FeatureNamespace::default())?;
    }

    let ep = EmbeddingPropagation {
        d_model: 64,
        passes: 50,
        batch_size: 1024,
        loss: Loss::MarginLoss(10.0, 10),
        valid_pct: 0.0,
        ..EmbeddingPropagation::default()
    };
    // Reconstruct each node from the average of up to 10 sampled neighbours.
    let model = AveragedFeatureModel::new(Sample::All, Sample::Fixed(10), false, false);
    let feature_embeddings = ep.learn_features(&graph, &mut features, None, &model)?;

    // A node's embedding is the average of its features' embeddings.
    let embedder = NodeEmbedder::new(FeatureAggregator::averaged());
    let node_embeddings = embedder.embed_feature_set(&features, &feature_embeddings)?;
    node_embeddings.save(output_path, None)?;
    Ok(())
}
