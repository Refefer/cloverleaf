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
