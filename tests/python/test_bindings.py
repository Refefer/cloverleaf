"""Tests of the Python bindings: argument handling, enums, indexing and the
input checks that turn bad inputs into Python exceptions instead of panics.

Build the module first (`maturin develop --release`), then run `pytest`.
"""
import gzip
import math

import pytest

import cloverleaf as cl


def n(i):
    return ("n", str(i))


@pytest.fixture
def graph():
    gb = cl.GraphBuilder()
    for a, b in [(0, 1), (1, 2), (2, 0), (2, 3)]:
        gb.add_edge(n(a), n(b), 1.0, cl.EdgeType.Undirected)
    return gb.build_graph()


def embeddings(order, distance=cl.Distance.Cosine):
    return cl.NodeEmbeddings.new_from_list(
        [(n(i), [float(i), 1.0]) for i in order], distance
    )


# Optional arguments and argument conversion


def test_optional_args_default_to_none(graph):
    graph.get_edges(n(0))
    cl.RandomWalker(0.5, 50).walk(graph, n(0))


@pytest.mark.parametrize("filter_type", ["n", ["n"]])
def test_filter_type_accepts_str_and_list(graph, filter_type):
    cl.RandomWalker(0.5, 50).walk(graph, n(0), seed=1, k=3, filter_type=filter_type)


def test_simple_enums_compare():
    assert cl.Distance.Cosine == cl.Distance.Cosine
    assert cl.EdgeType.Directed != cl.EdgeType.Undirected


# NodeEmbeddings


def test_new_from_list_rejects_ragged_embeddings():
    with pytest.raises(ValueError):
        cl.NodeEmbeddings.new_from_list(
            [(n(0), [1.0, 2.0]), (n(1), [1.0])], cl.Distance.Cosine
        )


def test_node_embeddings_indexing():
    ne = embeddings(range(4), cl.Distance.Euclidean)
    assert ne[-1] == (n(3), [3.0, 1.0])
    with pytest.raises(IndexError):
        ne[-5]
    with pytest.raises(IndexError):
        ne[4]


def test_set_embedding_checks_dimensions():
    ne = embeddings(range(4))
    with pytest.raises(ValueError):
        ne.set_embedding(n(0), [1.0, 2.0, 3.0])
    ne.set_embedding(n(0), [9.0, 9.0])
    assert ne.get_embedding(n(0)) == [9.0, 9.0]


def test_vocab_iteration_and_nearest_neighbor():
    ne = embeddings(range(4), cl.Distance.Euclidean)
    assert list(ne.vocab()) == [n(i) for i in range(4)]
    assert len(ne.nearest_neighbor([3.0, 1.0], 2)) == 2


def test_builder_rejects_wrong_dimensions_without_touching_vocab():
    b = cl.NodeEmbeddingsBuilder(cl.Distance.Cosine)
    b.add_embedding(n(0), [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        b.add_embedding(n(1), [1.0, 2.0, 3.0, 4.0, 5.0])
    b.add_embedding(n(1), [4.0, 5.0, 6.0])
    ne = b.build()
    assert ne.dims() == 3
    assert ne[1] == (n(1), [4.0, 5.0, 6.0])
    assert len(list(ne.vocab())) == 2


# FeatureSet


def test_empty_prefix_delimiter_is_rejected():
    with pytest.raises(ValueError):
        cl.FeatureNamespace.prefix("")


@pytest.fixture
def gz_features(tmp_path):
    path = tmp_path / "feats.gz"
    with gzip.open(path, "wt") as f:
        for i in range(4):
            f.write(f"n\t{i}\tcolor::red size→big\n")
    return str(path)


@pytest.mark.parametrize(
    "delim, expected", [("::", ("color", "red")), ("→", ("size", "big"))]
)
def test_gz_features_with_prefix_delimiters(gz_features, delim, expected):
    fs = cl.FeatureSet.new_from_file(gz_features, cl.FeatureNamespace.prefix(delim))
    assert expected in fs.get_features(n(0))


def test_feature_set_index_out_of_range(gz_features):
    fs = cl.FeatureSet.new_from_file(gz_features)
    with pytest.raises(IndexError):
        fs[-1000]


# Vocab alignment between graphs, features and embeddings


def test_propagate_rejects_misaligned_features(graph, tmp_path):
    # The feature file lists the nodes in a different order than the graph.
    path = tmp_path / "feats_reordered"
    path.write_text("".join(f"n\t{i}\tf{i}\n" for i in [3, 2, 1, 0]))
    fs = cl.FeatureSet.new_from_file(str(path))
    with pytest.raises(ValueError):
        cl.FeaturePropagator(2).propagate(graph, fs)


def test_propagate_accepts_aligned_features(graph):
    cl.FeaturePropagator(2).propagate(graph, cl.FeatureSet.new_from_graph(graph))


def test_align_rejects_misaligned_embeddings(graph):
    with pytest.raises(ValueError):
        cl.NeighborhoodAligner().align(embeddings([3, 2, 1, 0]), graph)


def test_align_accepts_aligned_embeddings(graph):
    out = cl.NeighborhoodAligner(alpha=0.5).align(embeddings(range(4)), graph)
    assert out.dims() == 2


def test_embedding_aligner_node_query_uses_original_embedding():
    # A node query is a point in the original space, where its anchors are searched for: it must
    # map exactly like the raw embedding the node has there, not its translated one.
    orig = cl.NodeEmbeddings.new_from_list(
        [(n(i), e) for i, e in enumerate([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])],
        cl.Distance.Euclidean)
    translated = cl.NodeEmbeddings.new_from_list(
        [(n(i), e) for i, e in enumerate([[0., 0.], [3., 0.], [0., 3.], [3., 3.]])],
        cl.Distance.Euclidean)
    ann = cl.EmbAnn(orig, 2, 2, seed=1)
    aligner = cl.EmbeddingAligner(3, 0, 0.1, 0.0, 20)
    by_node = aligner.align(orig, ann, translated, cl.Query.node(*n(1)), seed=1)
    by_embedding = aligner.align(orig, ann, translated, cl.Query.embedding([1., 0.]), seed=1)
    assert by_node == by_embedding


# Graph IO


def test_empty_edge_file(tmp_path):
    path = tmp_path / "empty.edges"
    path.touch()
    g = cl.Graph.load(str(path), cl.EdgeType.Directed)
    assert g.nodes() == 0
    g.save(str(tmp_path / "empty.out"))


# Optimizers


@pytest.mark.parametrize("weight_decay", [-0.1, float("nan"), float("inf")])
def test_adamw_rejects_invalid_weight_decay(weight_decay):
    with pytest.raises(ValueError):
        cl.Optimizer.adamw(weight_decay)


def test_optimizer_repr():
    assert repr(cl.Optimizer.adam()) == "Optimizer.adam()"
    assert repr(cl.Optimizer.adamw()) == "Optimizer.adamw(weight_decay=0.001)"


@pytest.mark.parametrize("optimizer", [None, cl.Optimizer.adam(), cl.Optimizer.adamw(0.1)])
def test_embedding_propagator_accepts_optimizer(graph, optimizer):
    ep = cl.EmbeddingPropagator(dims=4, passes=2, valid_pct=0.0, indicator=False,
                                optimizer=optimizer)
    feats = ep.learn_features(graph, cl.FeatureSet.new_from_graph(graph))
    assert feats.dims() == 4


@pytest.mark.parametrize("optimizer", [None, cl.Optimizer.adamw(0.1)])
def test_ppr_rank_accepts_optimizer(graph, optimizer):
    learner = cl.PprRankLearner(0.1, 2, 4, 2, 3.0, 2, 2, 1, valid_pct=0.0,
                                optimizer=optimizer)
    feats = learner.learn_features(graph, cl.FeatureSet.new_from_graph(graph), indicator=False)
    assert feats.dims() == 4


# Embedding Propagation training options


@pytest.mark.parametrize("kwargs", [dict(lr_schedule="linear"), dict(init="xavier")])
def test_embedding_propagator_rejects_unknown_options(kwargs):
    with pytest.raises(ValueError):
        cl.EmbeddingPropagator(**kwargs)


@pytest.mark.parametrize("lr_schedule", ["cosine", "constant"])
@pytest.mark.parametrize("init", ["unit", "glorot"])
def test_embedding_propagator_schedules_and_inits(graph, lr_schedule, init):
    ep = cl.EmbeddingPropagator(dims=4, passes=2, valid_pct=0.0, indicator=False,
                                lr_schedule=lr_schedule, init=init)
    feats = ep.learn_features(graph, cl.FeatureSet.new_from_graph(graph))
    assert feats.dims() == 4


@pytest.mark.parametrize("update_frequency", ["batch", "epoch"])
def test_embedding_propagator_update_frequency(graph, update_frequency):
    ep = cl.EmbeddingPropagator(dims=4, passes=2, indicator=False, valid_pct=0.0,
                                update_frequency=update_frequency)
    fe = ep.learn_features(graph, cl.FeatureSet.new_from_graph(graph))
    assert fe.dims() == 4


def test_embedding_propagator_rejects_unknown_update_frequency():
    with pytest.raises(ValueError, match="update_frequency"):
        cl.EmbeddingPropagator(update_frequency="step")


# Fast embedding export


def test_write_embeddings_round_trips(tmp_path):
    np = pytest.importorskip("numpy")
    keys = [("q", "a"), ("p", "b b"), ("q", "c")]
    values = np.array([[1.5, -2.25], [1e-8, 3.0e7], [0.1, 2.0 / 3.0]], dtype=np.float32)
    for name in ("out.emb", "out.emb.gz"):
        path = str(tmp_path / name)
        cl.write_embeddings(path, keys, values)
        loaded = cl.NodeEmbeddings.load(path, cl.Distance.Cosine)
        assert len(loaded) == 3
        for key, row in zip(keys, values):
            assert loaded.get_embedding(key) == [float(x) for x in row]


def _lists(graph, k, **kwargs):
    import struct
    nodes, weights = cl.ppr_lists(graph, k, 500, 0.5, indicator=False, **kwargs)
    count = graph.nodes() * k
    assert len(nodes) == 8 * count and len(weights) == 4 * count
    return (struct.unpack(f"<{count}q", nodes), struct.unpack(f"<{count}f", weights))


def test_ppr_lists_are_normalised_and_exclude_the_start(graph):
    k = 3
    nodes, weights = _lists(graph, k)
    vocab = list(graph.vocab())
    for v in range(graph.nodes()):
        listed = [(nodes[v * k + i], weights[v * k + i]) for i in range(k) if weights[v * k + i] > 0]
        assert listed and abs(sum(w for _, w in listed) - 1) < 1e-5
        assert all(vocab[u] != vocab[v] for u, _ in listed)
        assert [w for _, w in listed] == sorted((w for _, w in listed), reverse=True)


def test_ppr_lists_other_type_only():
    gb = cl.GraphBuilder()
    for q in range(4):
        for p in range(q, q + 2):
            gb.add_edge(("q", str(q)), ("p", str(p)), 1.0, cl.EdgeType.Undirected)
    g = gb.build_graph()
    k = 3
    nodes, weights = _lists(g, k, other_type_only=True)
    vocab = list(g.vocab())
    for v in range(g.nodes()):
        for i in range(k):
            if weights[v * k + i] > 0:
                assert vocab[nodes[v * k + i]][0] != vocab[v][0]


@pytest.mark.parametrize("args", [(3, 500, 0.0), (3, 0, 0.5), (0, 500, 0.5)])
def test_ppr_lists_rejects_bad_arguments(graph, args):
    with pytest.raises(ValueError):
        cl.ppr_lists(graph, *args, indicator=False)


def test_write_embeddings_rejects_bad_shapes(tmp_path):
    np = pytest.importorskip("numpy")
    path = str(tmp_path / "bad.emb")
    with pytest.raises(ValueError):
        cl.write_embeddings(path, [("q", "a")], np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError):
        cl.write_embeddings(path, [("q", "a")], np.zeros((2, 3), dtype=np.float32))


def test_attention_aggregator_from_weights(tmp_path):
    # dims 2, 2 heads of width 1; with a zero output projection it pools to the mean.
    agg = cl.FeatureAggregator.Attention(2, 2, 1, [0.5] * 12, [0.0] * 4)
    assert repr(agg) == "FeatureAggregator<Attention<dims=2,heads=2,head_dims=1>>"
    fe = cl.NodeEmbeddings.new_from_list([(("f", "a"), [1.0, 0.0]), (("f", "b"), [0.0, 3.0])], cl.Distance.Cosine)
    assert cl.NodeEmbedder(agg).embed_adhoc([("f", "a"), ("f", "b")], fe) == [0.5, 1.5]
    path = str(tmp_path / "agg")
    agg.save(path)
    assert repr(cl.FeatureAggregator.load(path)) == repr(agg)


@pytest.mark.parametrize("args", [(2, 2, 1, [0.0] * 11, [0.0] * 4), (2, 2, 1, [0.0] * 12, [0.0] * 3),
                                  (2, 0, 1, [], []), (2, 2, 1, [float("nan")] * 12, [0.0] * 4)])
def test_attention_aggregator_rejects_bad_weights(args):
    with pytest.raises(ValueError):
        cl.FeatureAggregator.Attention(*args)


def test_attention_aggregator_needs_matching_feature_width():
    agg = cl.FeatureAggregator.Attention(2, 2, 1, [0.5] * 12, [0.0] * 4)
    fe = cl.NodeEmbeddings.new_from_list([(("f", "a"), [1.0, 0.0, 0.0])], cl.Distance.Cosine)
    with pytest.raises(ValueError):
        cl.NodeEmbedder(agg).embed_adhoc([("f", "a")], fe)


# Attention EP


def attention_features(graph):
    fs = cl.FeatureSet.new_from_graph(graph)
    for i, feats in enumerate([["a", "b"], ["b", "c"], ["c", "a", "d"], ["d"]]):
        fs.set_features(n(i), [("f", f) for f in feats])
    return fs


@pytest.mark.parametrize("update_frequency", ["batch", "epoch"])
def test_attention_ep_learns_an_aggregator(graph, tmp_path, update_frequency):
    ep = cl.EmbeddingPropagator(dims=6, passes=3, batch_size=2, valid_pct=0.0, indicator=False,
                                attention_heads=2, dense_alpha=0.01, update_frequency=update_frequency)
    with pytest.raises(ValueError, match="no weights"):
        ep.feature_aggregator()
    fs = attention_features(graph)
    fe = ep.learn_features(graph, fs)
    assert fe.dims() == 6
    agg = ep.feature_aggregator()
    assert repr(agg) == "FeatureAggregator<Attention<dims=6,heads=2,head_dims=8>>"
    path = str(tmp_path / "agg")
    agg.save(path)
    nodes = cl.NodeEmbedder(cl.FeatureAggregator.load(path)).embed_feature_set(fs, fe)
    assert nodes.dims() == 6 and all(map(math.isfinite, nodes.get_embedding(n(2))))


def test_attention_ep_warm_starts_from_an_aggregator(graph, tmp_path):
    fs = attention_features(graph)
    first = cl.EmbeddingPropagator(dims=6, passes=2, valid_pct=0.0, indicator=False, attention_heads=2)
    first.learn_features(graph, fs)
    path = str(tmp_path / "agg")
    first.feature_aggregator().save(path)
    saved = open(path).read()

    second = cl.EmbeddingPropagator(dims=6, passes=0, valid_pct=0.0, indicator=False, attention_heads=2)
    second.learn_features(graph, attention_features(graph), aggregator=cl.FeatureAggregator.load(path))
    second.feature_aggregator().save(path)
    assert open(path).read() == saved      # no passes: the weights are the warm start's

    wrong_dims = cl.EmbeddingPropagator(dims=4, passes=0, valid_pct=0.0, indicator=False, attention_heads=2)
    with pytest.raises(ValueError):
        wrong_dims.learn_features(graph, attention_features(graph), aggregator=cl.FeatureAggregator.load(path))
    averaged = cl.EmbeddingPropagator(dims=6, passes=0, valid_pct=0.0, indicator=False)
    with pytest.raises(ValueError):
        averaged.learn_features(graph, attention_features(graph), aggregator=cl.FeatureAggregator.load(path))
    assert repr(averaged.feature_aggregator()) == "FeatureAggregator<Averaged>"


@pytest.mark.parametrize("kwargs", [dict(attention_heads=0), dict(dense_alpha=-1.0)])
def test_attention_ep_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        cl.EmbeddingPropagator(**kwargs)


def test_attention_ep_has_no_packed_attention_options():
    with pytest.raises(TypeError):
        cl.EmbeddingPropagator(attention=4)
    with pytest.raises(TypeError):
        cl.EmbeddingPropagator(context_window=2)
