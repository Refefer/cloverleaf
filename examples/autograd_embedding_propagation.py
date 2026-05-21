import math

import cloverleaf


def build_graph():
    gb = cloverleaf.GraphBuilder()
    for src, dst in [
        (("doc", "a"), ("doc", "b")),
        (("doc", "b"), ("doc", "c")),
        (("doc", "c"), ("doc", "d")),
        (("doc", "d"), ("doc", "a")),
        (("doc", "a"), ("doc", "c")),
    ]:
        gb.add_edge(src, dst, 1.0, cloverleaf.EdgeType.Undirected)
    return gb.build_graph()


def build_features(graph):
    features = cloverleaf.FeatureSet.new_from_graph(graph)
    features.set_features(("doc", "a"), [("feat", "red"), ("feat", "round")])
    features.set_features(("doc", "b"), [("feat", "red"), ("feat", "square")])
    features.set_features(("doc", "c"), [("feat", "blue"), ("feat", "round")])
    features.set_features(("doc", "d"), [("feat", "blue"), ("feat", "square")])
    return features


def run_propagation(graph, features, *, attention=False):
    propagator = cloverleaf.EmbeddingPropagator(
        alpha=0.01,
        loss=cloverleaf.EPLoss.margin(1.0, 1),
        batch_size=2,
        dims=4,
        passes=2,
        seed=13,
        max_nodes=None,
        weighted_neighbor_sampling=False,
        weighted_neighbor_averaging=False,
        max_features=None,
        loss_weighting=None,
        valid_pct=0.0,
        hard_negatives=0,
        indicator=False,
        attention=1 if attention else None,
        attention_heads=1 if attention else None,
        context_window=None,
        noise=0.0,
    )
    return propagator.learn_features(graph, features, None)


def assert_finite_embedding(embeddings, node):
    vector = embeddings.get_embedding(node)
    assert vector, f"missing embedding for {node}"
    assert all(math.isfinite(value) for value in vector), vector
    return vector


if __name__ == "__main__":
    graph = build_graph()

    features = build_features(graph)
    embeddings = run_propagation(graph, features, attention=False)
    print("averaged dims:", embeddings.dims())
    print("averaged feat:red:", assert_finite_embedding(embeddings, ("feat", "red")))

    attention_features = build_features(graph)
    attention_embeddings = run_propagation(graph, attention_features, attention=True)
    print("attention dims:", attention_embeddings.dims())
    print("attention feat:red:", assert_finite_embedding(attention_embeddings, ("feat", "red")))
