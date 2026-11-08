"""Tests of the Python bindings: argument handling, enums, indexing and the
input checks that turn bad inputs into Python exceptions instead of panics.

Build the module first (`maturin develop --release`), then run `pytest`.
"""
import gzip

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


# Graph IO


def test_empty_edge_file(tmp_path):
    path = tmp_path / "empty.edges"
    path.touch()
    g = cl.Graph.load(str(path), cl.EdgeType.Directed)
    assert g.nodes() == 0
    g.save(str(tmp_path / "empty.out"))
