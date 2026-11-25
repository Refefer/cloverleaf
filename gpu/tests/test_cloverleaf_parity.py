"""Checks against cloverleaf itself: the same nodes in the same order, the same features, and, with
mean pooling, node embeddings that cloverleaf's averaged embedder rebuilds from the exported
feature table.  ep_gpu loads through cloverleaf, so these check the pieces around loading:
export, the aggregator, and pruning against cloverleaf's own."""
import numpy as np
import pytest
import torch

import cloverleaf as cl

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
    """Loaded as cloverleaf's EP pipeline loads them, featureless nodes filled in as learning
    does."""
    graph = cl.Graph.load(edges, cl.EdgeType.Undirected)
    features = cl.FeatureSet.new_from_graph(graph)
    features.load_into(feats, cl.FeatureNamespace.node_type())
    features.fill_missing_nodes()
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
    pruned.fill_missing_nodes()
    for v, node in enumerate(graph.vocab()):
        start, end = int(ours.feat_ptr[v]), int(ours.feat_ptr[v + 1])
        mine = {ours.feature_keys[f] for f in ours.feat_idx[start:end].tolist()}
        assert mine == {tuple(f) for f in pruned.get_features(node)}, node


def test_mean_pooling_exports_an_averaged_aggregator(files, tmp_path):
    ours = data.load(*files)
    cfg = Config(dims=6, passes=1, batch_size=2, device="cpu", seed=5)
    prefix = str(tmp_path / "ep")
    export(train(ours, cfg, log=lambda m: None), ours, cfg, prefix)
    assert open(prefix + ".aggregator").read() == "Averaged\n"
    assert repr(cl.FeatureAggregator.load(prefix + ".aggregator")) == "FeatureAggregator<Averaged>"


def test_attention_pooled_embeddings_match_cloverleafs_attention_embedder(files, tmp_path):
    """Cloverleaf, given the exported feature table and aggregator, embeds every node, and a set
    of features no node has, exactly as the GPU model does.  dims 13 with 2 heads pads each head
    to width 8, so the padding is exercised too."""
    from ep_gpu.train import embed_feature_sets
    ours = data.load(*files)
    cfg = Config(dims=13, passes=20, batch_size=2, lr=0.1, dense_lr=0.05, pooling="attention", heads=2,
                 precision="fp32", device="cpu", seed=5)
    model = train(ours, cfg, log=lambda m: None)
    assert model.pool.out.weight.abs().max() > 1e-3, "the output projection should have trained"
    prefix = str(tmp_path / "ep")
    export(model, ours, cfg, prefix)

    aggregator = cl.FeatureAggregator.load(prefix + ".aggregator")
    assert repr(aggregator) == "FeatureAggregator<Attention<dims=13,heads=2,head_dims=8>>"
    graph, features = cloverleaf_features(*files)
    feature_embs = cl.NodeEmbeddings.load(prefix + ".feature-embeddings", cl.Distance.Cosine)
    embedder = cl.NodeEmbedder(aggregator)
    rebuilt = embedder.embed_feature_set(features, feature_embs)
    exported = cl.NodeEmbeddings.load(prefix + ".node-embeddings", cl.Distance.Cosine)
    mean_pooled = cl.NodeEmbedder(cl.FeatureAggregator.Averaged()).embed_feature_set(features, feature_embs)
    for node in graph.vocab():
        assert np.allclose(rebuilt.get_embedding(node), exported.get_embedding(node), atol=1e-5), node
    assert not np.allclose(rebuilt.get_embedding(("n", "b")), mean_pooled.get_embedding(("n", "b")), atol=1e-3)

    new_node = [("n", "x"), ("n", "z"), ("n", "q")]
    ours_new = embed_feature_sets(model, ours.feature_index(), [new_node])[0].numpy()
    assert np.allclose(embedder.embed_adhoc(new_node, feature_embs), ours_new, atol=1e-5)


@pytest.mark.parametrize("pooling, clip", [("mean", None), ("mean", 1.5), ("attention", None), ("attention", 1.5)])
def test_poincare_models_rebuild_exactly_in_cloverleaf(files, tmp_path, pooling, clip):
    """A Poincaré model's exported aggregator records the geometry; cloverleaf then embeds every
    node, and a new feature set, as the GPU model does, and ranks them by the same distances."""
    from ep_gpu.train import embed_feature_sets
    ours = data.load(*files)
    extra = dict(pooling="attention", heads=2, dense_lr=0.05) if pooling == "attention" else {}
    cfg = Config(dims=13, passes=20, batch_size=2, lr=0.1, loss="starspace", gamma=1.0, geometry="poincare",
                 curvature=0.3, max_tangent_norm=clip, precision="fp32", device="cpu", seed=5, **extra)
    model = train(ours, cfg, log=lambda m: None)
    prefix = str(tmp_path / "ep")
    export(model, ours, cfg, prefix)

    aggregator = cl.FeatureAggregator.load(prefix + ".aggregator")
    assert aggregator.geometry == "poincare" and np.isclose(aggregator.curvature, 0.3)
    graph, features = cloverleaf_features(*files)
    feature_embs = cl.NodeEmbeddings.load(prefix + ".feature-embeddings", cl.Distance.Cosine)
    embedder = cl.NodeEmbedder(aggregator)
    rebuilt = embedder.embed_feature_set(features, feature_embs)
    assert isinstance(rebuilt.get_distance(), cl.PoincareDistance)
    exported = cl.NodeEmbeddings.load(prefix + ".node-embeddings", cl.PoincareDistance(0.3))
    for node in graph.vocab():
        assert np.allclose(rebuilt.get_embedding(node), exported.get_embedding(node), atol=1e-5), node
    nodes = list(graph.vocab())
    X = torch.tensor([exported.get_embedding(n) for n in nodes])      # already points of the ball
    gpu_d = model.space.distance(X[:, None], X[None, :]).numpy()   # element-wise: exact where x = y
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            d = rebuilt.compute_distance(cl.Query.node(*a), cl.Query.node(*b))
            assert np.isclose(d, gpu_d[i, j], rtol=1e-3, atol=1e-3), (a, b)

    new_node = [("n", "x"), ("n", "z"), ("n", "q")]
    ours_new = embed_feature_sets(model, ours.feature_index(), [new_node])[0].detach().numpy()
    assert np.allclose(embedder.embed_adhoc(new_node, feature_embs), ours_new, atol=1e-5)


@pytest.mark.parametrize("kwargs", [dict(feature_importance=True), dict(poincare_pooling="midpoint"),
                                    dict(pooling="attention", heads=2, attention_keeps_norm=True)])
def test_models_cloverleaf_cant_reproduce_export_no_aggregator(files, tmp_path, kwargs):
    import os
    ours = data.load(*files)
    cfg = Config(dims=8, passes=1, batch_size=2, geometry="poincare", device="cpu", **kwargs)
    prefix = str(tmp_path / "ep")
    export(train(ours, cfg, log=lambda m: None), ours, cfg, prefix)
    assert os.path.exists(prefix + ".pt") and not os.path.exists(prefix + ".aggregator")
