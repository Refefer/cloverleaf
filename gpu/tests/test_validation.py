import pytest
import torch

from ep_gpu import data
from ep_gpu.train import Config, train
from ep_gpu.validation import RankingProbe, hold_out_edges


@pytest.fixture
def two_cliques(tmp_path):
    """Two 10-node cliques joined by one edge; every node's only feature is its id."""
    lines = [f"n\t{c * 10 + a}\tn\t{c * 10 + b}\t1" for c in range(2) for a in range(10) for b in range(a + 1, 10)]
    lines.append("n\t0\tn\t10\t1")
    path = tmp_path / "cliques.edges"
    path.write_text("\n".join(lines) + "\n")
    return data.load(str(path))


def test_trained_neighbours_outrank_random_nodes(tmp_path):
    # Five 8-node cliques in a ring: four in five random nodes are in another clique.
    lines = [f"n\t{c * 8 + a}\tn\t{c * 8 + b}\t1" for c in range(5) for a in range(8) for b in range(a + 1, 8)]
    lines += [f"n\t{c * 8}\tn\t{(c + 1) % 5 * 8}\t1" for c in range(5)]
    path = tmp_path / "cliques.edges"
    path.write_text("\n".join(lines) + "\n")
    g = data.load(str(path))
    valid = torch.arange(1, 40, 5)
    probe = RankingProbe.from_nodes(g, valid, positives=3, negatives=400, seed=1)
    assert len(probe) == 24
    hr0, auc0 = probe.evaluate(train(g, Config(dims=16, passes=0, device="cpu", seed=3)), g)
    cfg = Config(dims=16, passes=60, batch_size=8, lr=0.05, loss="starspace", gamma=0.5, negatives=3,
                 device="cpu", seed=3)
    hr, auc = probe.evaluate(train(g, cfg, log=lambda m: None), g)
    assert 0.3 < auc0 < 0.7, auc0              # random embeddings: about half the negatives
    assert auc > 0.75 and auc > auc0 + 0.2, (auc0, auc)
    assert hr >= hr0 and 0.0 <= hr <= 1.0


def test_negatives_are_drawn_from_the_neighbours_type(tmp_path):
    edges = tmp_path / "g.edges"
    edges.write_text("".join(f"q\tq{i}\tp\tp{(i + j) % 7}\t1\n" for i in range(6) for j in range(2)))
    g = data.load(str(edges))
    queries = torch.tensor([g.node_id("q", f"q{i}") for i in range(6)])
    probe = RankingProbe.from_nodes(g, queries, positives=2, negatives=50, seed=0)
    assert set(probe.negatives) == {int(probe.positive_type[0])}       # one type: products
    assert all(g.node_types[n] == "p" for n in probe.negatives[int(probe.positive_type[0])].tolist())
    assert all(g.node_types[n] == "p" for n in probe.positive.tolist())


def test_validation_log_reports_the_ranking(two_cliques):
    lines = []
    train(two_cliques, Config(dims=8, passes=2, batch_size=4, valid_pct=0.2, device="cpu"), log=lines.append)
    assert all("valid hr@10" in m and " auc " in m for m in lines)
    lines.clear()
    train(two_cliques, Config(dims=8, passes=1, batch_size=4, valid_pct=0.2, valid_negatives=0, device="cpu"),
          log=lines.append)
    assert "hr@10" not in lines[0]


def test_hold_out_edges_removes_both_directions_and_nothing_else(two_cliques):
    g, src, dst = hold_out_edges(two_cliques, 0.1, torch.Generator().manual_seed(2))
    assert len(src) == int(len(two_cliques.adj_idx) * 0.1)

    def edges(graph):
        out = set()
        for v in range(graph.num_nodes):
            for u in graph.adj_idx[graph.adj_ptr[v]:graph.adj_ptr[v + 1]].tolist():
                out.add((v, u))
        return out
    before, after = edges(two_cliques), edges(g)
    held = {(int(a), int(b)) for a, b in zip(src, dst)} | {(int(b), int(a)) for a, b in zip(src, dst)}
    assert after == before - held
    assert len(g.adj_w) == len(g.adj_idx) == int(g.adj_ptr[-1])


def test_training_with_held_out_edges_reports_their_ranking(two_cliques):
    lines = []
    train(two_cliques, Config(dims=8, passes=2, batch_size=4, valid_edges=0.1, device="cpu"), log=lines.append)
    assert all("held-out edges hr@10" in m and "valid " not in m.split("held-out")[0][-8:] for m in lines)
    with pytest.raises(ValueError, match="valid_edges"):
        train(two_cliques, Config(dims=4, passes=1, valid_edges=1.0, device="cpu"), log=lambda m: None)


def test_are_neighbours(two_cliques):
    from ep_gpu import sampling
    keys = sampling.edge_keys(two_cliques)
    a = torch.tensor([0, 0, 15])
    b = torch.tensor([[1, 10, 15], [11, 0, 9], [10, 0, 19]])
    assert sampling.are_neighbours(keys, two_cliques.num_nodes, a, b).tolist() == \
        [[True, True, False], [False, False, True], [True, False, True]]


def test_infonce_and_starspace_ignore_excluded_negatives():
    from ep_gpu import losses
    pos = torch.tensor([0.5, 0.5])
    neg = torch.tensor([[0.9, 0.0], [0.9, 0.0]])
    keep_all = losses.infonce_from_cosines(pos, neg, 0.1)
    drop_close = losses.infonce_from_cosines(pos, neg, 0.1, torch.tensor([[True, False], [False, False]]))
    assert drop_close[0] < keep_all[0] and drop_close[1] == keep_all[1]
    hinge = losses.starspace_from_cosines(pos, neg, 0.2, torch.tensor([[True, False], [False, False]]))
    assert hinge[0] == 0 and hinge[1] > 0      # only the close negative violated the margin


@pytest.mark.parametrize("kwargs", [dict(pool_negatives=True), dict(negatives=4), dict(negatives=4, negative_pool=0)])
def test_masked_and_mixed_infonce_trains(two_cliques, kwargs):
    cfg = Config(dims=16, passes=40, batch_size=4, lr=0.1, loss="infonce", gamma=0.2, mask_neighbours=True,
                 starspace_weight=0.5, device="cpu", seed=3, **kwargs)
    from ep_gpu.train import node_embeddings
    emb = torch.nn.functional.normalize(node_embeddings(train(two_cliques, cfg, log=lambda m: None), two_cliques), dim=1)
    d = 1 - emb @ emb.T
    assert float(d[:10, 10:].mean() - (d[:10, :10].sum() + d[10:, 10:].sum()) / 180) > 0.3


def test_keep_best_probe_restores_the_best_pass(two_cliques):
    lines = []
    cfg = Config(dims=8, passes=6, batch_size=4, lr=0.05, valid_edges=0.1, keep_best_probe=True, device="cpu", seed=2)
    model = train(two_cliques, cfg, log=lines.append)
    hrs = [float(m.split("hr@10 ")[1].split()[0]) for m in lines if m.startswith("pass")]
    kept = [m for m in lines if m.startswith("kept pass")]
    assert len(kept) == 1
    best_pass = int(kept[0].split()[2].rstrip(","))
    assert hrs[best_pass - 1] == max(hrs)


@pytest.mark.parametrize("kwargs", [dict(loss="starspace", mask_neighbours=True), dict(loss="infonce", starspace_weight=1.5),
                                    dict(keep_best_probe=True, valid_negatives=0)])
def test_infonce_options_are_checked(two_cliques, kwargs):
    with pytest.raises(ValueError):
        train(two_cliques, Config(dims=4, passes=1, device="cpu", **kwargs), log=lambda m: None)
