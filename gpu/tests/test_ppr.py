"""PPR lists from cloverleaf, and the listwise losses trained on them."""
import pytest
import torch

from ep_gpu import data
from ep_gpu.ppr import ppr_lists
from ep_gpu.train import Config, build_lists, node_embeddings, train


@pytest.fixture
def cliques(tmp_path):
    """Two 10-node cliques joined by one edge, as an edge file and the graph loaded from it."""
    lines = [f"n\t{c * 10 + a}\tn\t{c * 10 + b}\t1"
             for c in range(2) for a in range(10) for b in range(a + 1, 10)]
    lines.append("n\t0\tn\t10\t1")
    path = tmp_path / "cliques.edges"
    path.write_text("\n".join(lines) + "\n")
    return str(path), data.load(str(path))


def separation(emb):
    e = torch.nn.functional.normalize(emb, dim=1)
    d = 1 - e @ e.T
    off = ~torch.eye(10, dtype=torch.bool)
    return float(d[:10, 10:].mean() - torch.cat([d[:10, :10][off], d[10:, 10:][off]]).mean())


def test_lists_are_normalised_and_stay_in_the_clique(cliques):
    path, graph = cliques
    lists = ppr_lists(path, graph, k=5, walks=500, steps=0.5)
    assert lists.nodes.shape == (20, 5) and lists.nodes.dtype == graph.adj_idx.dtype
    assert torch.allclose(lists.weights.sum(1), torch.ones(20), atol=1e-5)
    assert (lists.weights[:, :-1] >= lists.weights[:, 1:]).all()
    own = torch.arange(20)[:, None]
    assert (lists.nodes != own).all()
    same_clique = (lists.nodes // 10) == (own // 10)
    assert same_clique.float().mean() > 0.9


def test_other_type_only_keeps_the_other_type(tmp_path):
    path = tmp_path / "bipartite.edges"
    path.write_text("".join(f"q\t{i}\tp\t{j}\t1\n" for i in range(6) for j in range(i, i + 3)))
    graph = data.load(str(path))
    lists = ppr_lists(str(path), graph, k=3, walks=500, steps=0.5, other_type_only=True)
    types = [graph.node_types[i] for i in range(graph.num_nodes)]
    for v in range(graph.num_nodes):
        listed = lists.nodes[v][lists.weights[v] > 0].tolist()
        assert listed and all(types[u] != types[v] for u in listed)


@pytest.mark.parametrize("loss", ["listnet", "listmle", "approxndcg"])
@pytest.mark.parametrize("pool_negatives", [False, True])
def test_listwise_losses_learn_the_cliques(cliques, loss, pool_negatives):
    path, graph = cliques
    cfg = Config(dims=16, passes=40, batch_size=8, lr=0.05, negatives=3, loss=loss, gamma=0.1,
                 negative_pool=16, pool_negatives=pool_negatives, ppr_k=5, ppr_walks=500,
                 device="cpu", seed=3)
    lists = build_lists(path, graph, cfg, log=lambda m: None)
    model = train(graph, cfg, log=lambda m: None, lists=lists)
    assert separation(node_embeddings(model, graph)) > 0.3


def test_listwise_attention_trains(cliques):
    path, graph = cliques
    cfg = Config(dims=16, passes=5, batch_size=8, lr=0.05, loss="listnet", gamma=0.1,
                 negative_pool=16, pool_negatives=True, pooling="attention", heads=2, ppr_k=5,
                 ppr_walks=200, device="cpu", seed=3, valid_pct=0.2)
    train(graph, cfg, log=lambda m: None, lists=build_lists(path, graph, cfg, log=lambda m: None))


def test_listwise_loss_needs_lists(cliques):
    with pytest.raises(ValueError):
        train(cliques[1], Config(dims=4, passes=1, loss="listnet", device="cpu"), log=lambda m: None)


@pytest.mark.parametrize("cfg", [Config(loss="listnet", negative_pool=None),
                                 Config(loss="listmle", ppr_passive=True),
                                 Config(loss="starspace", ppr_passive=True)])
def test_invalid_listwise_settings_are_rejected(cliques, cfg):
    path, graph = cliques
    cfg.dims, cfg.passes, cfg.device = 4, 1, "cpu"
    with pytest.raises(ValueError):
        train(graph, cfg, log=lambda m: None, lists=build_lists(path, graph, Config(ppr_k=3, ppr_walks=10)))


def test_subsample_draws_a_renormalised_sorted_subset():
    from ep_gpu.ppr import subsample
    nodes = torch.arange(24).view(2, 12)
    weights = torch.tensor([[0.3, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0],
                            [0.6, 0.4] + [0.0] * 10])
    g = torch.Generator().manual_seed(0)
    for by in ("weight", "uniform"):
        drawn, w = subsample(nodes, weights, 4, by, g)
        assert drawn.shape == (2, 4)
        assert set(drawn[0].tolist()) <= set(range(11))           # never the zero-weight padding
        assert torch.allclose(w.sum(1), torch.ones(2))
        assert (w[:, :-1] >= w[:, 1:]).all()
        assert drawn[1, :2].tolist() == [12, 13] and (w[1, 2:] == 0).all()   # short rows keep all
        original = dict(zip(nodes[0].tolist(), weights[0].tolist()))
        ratio = torch.tensor([original[n] for n in drawn[0].tolist()]) / w[0]
        assert torch.allclose(ratio, ratio[0].expand(4))           # same proportions as the list


def test_weighted_subsampling_favours_heavy_entries_and_uniform_does_not():
    from ep_gpu.ppr import subsample
    weights = torch.tensor([[0.9] + [0.1 / 9] * 9]).expand(2000, 10)
    nodes = torch.arange(10).expand(2000, 10)
    g = torch.Generator().manual_seed(1)
    heavy = lambda by: float((subsample(nodes, weights, 1, by, g)[0] == 0).float().mean())
    assert heavy("weight") > 0.85
    assert abs(heavy("uniform") - 0.1) < 0.03


def test_fixed_odd_walks_on_a_bipartite_graph_end_on_the_other_side(tmp_path):
    path = tmp_path / "bipartite.edges"
    path.write_text("".join(f"q\t{i}\tp\t{j}\t1\n" for i in range(6) for j in range(i, i + 3)))
    graph = data.load(str(path))
    lists = ppr_lists(str(path), graph, k=6, walks=500, steps=3)
    for v in range(graph.num_nodes):
        listed = lists.nodes[v][lists.weights[v] > 0].tolist()
        assert listed and all(graph.node_types[u] != graph.node_types[v] for u in listed)


def test_subsampled_lists_learn_the_cliques(cliques):
    path, graph = cliques
    cfg = Config(dims=16, passes=40, batch_size=8, lr=0.05, loss="listnet", gamma=0.1,
                 negative_pool=16, pool_negatives=True, ppr_k=9, ppr_sample=3, ppr_walks=500,
                 ppr_steps=3, device="cpu", seed=3)
    model = train(graph, cfg, log=lambda m: None, lists=build_lists(path, graph, cfg, log=lambda m: None))
    assert separation(node_embeddings(model, graph)) > 0.3


@pytest.mark.parametrize("bad", [dict(ppr_sample=11), dict(ppr_sample=0), dict(ppr_sample_by="top"),
                                 dict(ppr_steps=2.5), dict(ppr_steps=0.0)])
def test_invalid_sampling_and_steps_are_rejected(cliques, bad):
    path, graph = cliques
    cfg = Config(dims=4, passes=1, loss="listnet", device="cpu", ppr_k=10, **bad)
    with pytest.raises(ValueError):
        train(graph, cfg, log=lambda m: None, lists=build_lists(path, graph, Config(ppr_k=10, ppr_walks=10)))


@pytest.mark.parametrize("loss", ["listnet", "approxndcg"])
@pytest.mark.parametrize("pool_negatives", [False, True])
def test_listwise_losses_train_with_negative_types(tmp_path, loss, pool_negatives):
    """Two markets of queries and products; each side draws only the other side as negatives.
    Each query's nearest products are then its own market's."""
    path = tmp_path / "m.edges"
    path.write_text("".join(f"query\tq{c}-{i}\tproduct\tp{c}-{j}\t1\n"
                            for c in range(2) for i in range(6) for j in range(6)))
    graph = data.load(str(path))
    cfg = Config(dims=8, passes=40, batch_size=8, lr=0.05, negatives=3, loss=loss, gamma=0.1,
                 negative_pool=16, pool_negatives=pool_negatives, ppr_k=6, ppr_walks=300, ppr_steps=3,
                 negative_types="query=product;product=query", device="cpu", seed=3)
    lists = build_lists(str(path), graph, cfg, log=lambda m: None)
    model = train(graph, cfg, log=lambda m: None, lists=lists)
    X = torch.nn.functional.normalize(node_embeddings(model, graph), dim=1)
    types, names = graph.node_types, graph.node_names
    queries = [i for i, t in enumerate(types) if t == "query"]
    products = [i for i, t in enumerate(types) if t == "product"]
    right = 0
    for q in queries:
        sims = X[products] @ X[q]
        top = [products[j] for j in sims.topk(6).indices.tolist()]
        right += sum(names[p][1] == names[q][1] for p in top)
    assert right >= 0.85 * 6 * len(queries), f"{right} of {6 * len(queries)}"
