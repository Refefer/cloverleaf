"""Checks against cloverleaf itself: the same nodes in the same order, the same features, and, with
mean pooling, node embeddings that cloverleaf's averaged embedder rebuilds from the exported
feature table.  Skipped when cloverleaf isn't installed."""
import numpy as np
import pytest

cl = pytest.importorskip("cloverleaf")

from ep_gpu import data
from ep_gpu.train import Config, export, train


@pytest.fixture
def files(tmp_path):
    edges = tmp_path / "g.edges"
    edges.write_text("n\ta\tn\tb\t1\n"
                     "n\tb\tn\tc\t1\n"
                     "n\tc\tn\ta\t3\n"
                     "n\tc\tm\td\t1\n"
                     "m\td\tn\te\t2\n")
    feats = tmp_path / "g.features"
    feats.write_text("n\ta\tx y\n"
                     "n\tb\ty z w\n"
                     "n\tc\tx\n"
                     "n\te\tz q\n")
    return str(edges), str(feats)


def cloverleaf_features(edges, feats):
    """Loaded as cloverleaf's EP pipeline loads them.  Cloverleaf gives featureless nodes their
    own ("node", id) feature when training starts, so a 1-pass training run fills them in, as
    in a real pipeline, which embeds after learn_features."""
    graph = cl.Graph.load(edges, cl.EdgeType.Undirected)
    features = cl.FeatureSet.new_from_graph(graph)
    features.load_into(feats, cl.FeatureNamespace.node_type())
    cl.EmbeddingPropagator(dims=4, passes=1, indicator=False).learn_features(graph, features)
    return graph, features


def test_same_nodes_in_the_same_order(files):
    ours = data.load(*files)
    graph, _ = cloverleaf_features(*files)
    assert list(graph.vocab()) == list(zip(ours.node_types, ours.node_names))


def test_mean_pooled_embeddings_match_cloverleafs_averaged_embedder(files, tmp_path):
    ours = data.load(*files)
    cfg = Config(dims=6, passes=3, batch_size=2, device="cpu", seed=5)
    model = train(ours, cfg, log=lambda m: None)
    prefix = str(tmp_path / "ep")
    export(model, ours, cfg, prefix)

    graph, features = cloverleaf_features(*files)
    feature_embs = cl.NodeEmbeddings.load(prefix + ".feature-embeddings", cl.Distance.Cosine)
    rebuilt = cl.NodeEmbedder(cl.FeatureAggregator.Averaged()).embed_feature_set(features, feature_embs)
    exported = cl.NodeEmbeddings.load(prefix + ".node-embeddings", cl.Distance.Cosine)
    for node in graph.vocab():
        assert np.allclose(rebuilt.get_embedding(node), exported.get_embedding(node), atol=1e-5), node


@pytest.mark.parametrize("min_count", [2, 3])
def test_pruning_matches_cloverleafs_prune_min_count(files, min_count):
    ours = data.load(*files, prune_min_count=min_count)
    graph = cl.Graph.load(files[0], cl.EdgeType.Undirected)
    features = cl.FeatureSet.new_from_graph(graph)
    features.load_into(files[1], cl.FeatureNamespace.node_type())
    pruned = features.prune_min_count(min_count)
    cl.EmbeddingPropagator(dims=4, passes=1, indicator=False).learn_features(graph, pruned)
    for v, node in enumerate(graph.vocab()):
        start, end = int(ours.feat_ptr[v]), int(ours.feat_ptr[v + 1])
        mine = {ours.feature_keys[f] for f in ours.feat_idx[start:end].tolist()}
        assert mine == {tuple(f) for f in pruned.get_features(node)}, node
