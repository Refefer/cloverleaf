import numpy as np
import pytest
import torch

from ep_gpu import data
from ep_gpu.train import Config, embed_feature_sets, export, node_embeddings, train


@pytest.fixture
def two_cliques(tmp_path):
    """Two 10-node cliques joined by one edge; every node's only feature is its id."""
    lines = []
    for c in range(2):
        for a in range(10):
            for b in range(a + 1, 10):
                lines.append(f"n\t{c * 10 + a}\tn\t{c * 10 + b}\t1")
    lines.append("n\t0\tn\t10\t1")
    path = tmp_path / "cliques.edges"
    path.write_text("\n".join(lines) + "\n")
    return data.load(str(path))


def separation(emb):
    """Mean distance to the other clique minus mean distance within a node's own clique."""
    e = torch.nn.functional.normalize(emb, dim=1)
    d = 1 - e @ e.T
    within = torch.cat([d[:10, :10][~torch.eye(10, dtype=torch.bool)], d[10:, 10:][~torch.eye(10, dtype=torch.bool)]])
    across = d[:10, 10:].flatten()
    return float(across.mean() - within.mean())


@pytest.mark.parametrize("pooling,loss,gamma", [("mean", "margin", 1.0), ("mean", "starspace", 0.5),
                                               ("mean", "rankspace", 0.5), ("attention", "margin", 1.0)])
def test_learns_the_cliques(two_cliques, pooling, loss, gamma):
    before = separation(node_embeddings(train(two_cliques, Config(dims=16, passes=0, device="cpu", seed=3)),
                                        two_cliques))
    cfg = Config(dims=16, passes=60, batch_size=8, lr=0.05, loss=loss, gamma=gamma, negatives=3,
                 pooling=pooling, heads=2, device="cpu", seed=3)
    after = separation(node_embeddings(train(two_cliques, cfg, log=lambda m: None), two_cliques))
    assert after > before + 0.3 and after > 0.3, (before, after)


def test_sampling_options_train(two_cliques):
    cfg = Config(dims=8, passes=3, batch_size=4, max_nodes=3, max_features=0.5, weighted_sampling=True,
                 weighted_averaging=True, valid_pct=0.1, device="cpu")
    model = train(two_cliques, cfg, log=lambda m: None)
    assert torch.isfinite(node_embeddings(model, two_cliques)).all()


def test_export_round_trips_and_mean_pooling_matches_the_feature_table(two_cliques, tmp_path):
    cfg = Config(dims=8, passes=2, batch_size=4, device="cpu")
    model = train(two_cliques, cfg, log=lambda m: None)
    prefix = str(tmp_path / "ep")
    export(model, two_cliques, cfg, prefix)

    def read(path):
        rows = {}
        for line in open(path):
            t, n, e = line.rstrip("\n").split("\t")
            rows[(t, n)] = np.array([float(x) for x in e.strip("[]").split(",")])
        return rows
    nodes, feats = read(prefix + ".node-embeddings"), read(prefix + ".feature-embeddings")
    assert len(nodes) == two_cliques.num_nodes
    # Each node's only feature is ("node", id): its embedding is that feature's row.
    for v, (t, n) in enumerate(zip(two_cliques.node_types, two_cliques.node_names)):
        assert np.allclose(nodes[(t, n)], feats[("node", str(v))], atol=1e-6)
    # embed_feature_sets reproduces a node from its features, and skips unknown ones.
    index = two_cliques.feature_index()
    again = embed_feature_sets(model, index, [[("node", "3"), ("node", "nope")]])
    assert np.allclose(again[0].numpy(), nodes[("n", "3")], atol=1e-6)


def test_nested_mean_reconstruction_is_the_weighted_mean_of_neighbour_means(two_cliques):
    from ep_gpu import sampling
    from ep_gpu.model import EPModel
    m = EPModel(two_cliques.num_features, 4, seed=0)
    nodes = torch.tensor([0, 5])
    fidx, fmask, w = sampling.neighbour_features(two_cliques, nodes)
    per_neighbour = m.embed(fidx, fmask.float()).view(2, w.shape[1], -1)
    t = m.combine_neighbours(per_neighbour, w)
    expected = (per_neighbour * w[..., None]).sum(1) / w.sum(1, keepdim=True)
    assert torch.allclose(t, expected, atol=1e-6)


def test_flat_attention_still_trains(two_cliques):
    cfg = Config(dims=16, passes=40, batch_size=8, lr=0.05, negatives=3, pooling="attention", heads=2,
                 reconstruction="flat", device="cpu", seed=3)
    assert separation(node_embeddings(train(two_cliques, cfg, log=lambda m: None), two_cliques)) > 0.3


def test_unknown_reconstruction_is_rejected(two_cliques):
    with pytest.raises(ValueError):
        train(two_cliques, Config(dims=4, passes=1, reconstruction="sideways", device="cpu"), log=lambda m: None)


def test_attention_weights_use_their_own_learning_rate(two_cliques, monkeypatch):
    seen = []
    real_adam = torch.optim.Adam

    class Spy(real_adam):
        def step(self, *a, **k):
            seen.append(self.param_groups[0]["lr"])
            return super().step(*a, **k)
    monkeypatch.setattr(torch.optim, "Adam", Spy)
    cfg = Config(dims=8, passes=1, batch_size=10, lr=0.5, dense_lr=0.001, lr_schedule="constant",
                 pooling="attention", heads=2, device="cpu")
    train(two_cliques, cfg, log=lambda m: None)
    assert seen and all(abs(r - 0.001) < 1e-12 for r in seen)


def test_unknown_precision_is_rejected(two_cliques):
    with pytest.raises(ValueError):
        train(two_cliques, Config(dims=4, passes=1, precision="fp8", device="cpu"), log=lambda m: None)


@pytest.mark.parametrize("optimizer,table_dtype", [("rowwise_adam", "float32"), ("rowwise_adagrad", "float32"),
                                                   ("rowwise_adagrad", "bfloat16"), ("adam", "bfloat16")])
def test_memory_saving_options_learn(two_cliques, optimizer, table_dtype):
    cfg = Config(dims=16, passes=60, batch_size=8, lr=0.05, negatives=3, optimizer=optimizer,
                 table_dtype=table_dtype, device="cpu", seed=3)
    model = train(two_cliques, cfg, log=lambda m: None)
    assert model.features.weight.dtype == (torch.bfloat16 if table_dtype == "bfloat16" else torch.float32)
    assert separation(node_embeddings(model, two_cliques).float()) > 0.3


def test_attention_with_mean_neighbour_pooling(two_cliques):
    from ep_gpu import sampling
    from ep_gpu.model import EPModel, MeanPool
    m = EPModel(two_cliques.num_features, 8, pooling="attention", heads=2, seed=0)
    assert isinstance(m.neighbour_pool, MeanPool)
    torch.nn.init.normal_(m.pool.out.weight)
    fidx, fmask, w = sampling.neighbour_features(two_cliques, torch.tensor([0, 5]))
    per_neighbour = m.embed(fidx, fmask.float()).view(2, w.shape[1], -1)
    expected = (per_neighbour * w[..., None]).sum(1) / w.sum(1, keepdim=True)
    assert torch.allclose(m.combine_neighbours(per_neighbour, w), expected, atol=1e-6)
    cfg = Config(dims=16, passes=40, batch_size=8, lr=0.05, negatives=3, pooling="attention", heads=2,
                 device="cpu", seed=3)
    assert separation(node_embeddings(train(two_cliques, cfg, log=lambda m: None), two_cliques)) > 0.3


def test_unknown_neighbour_pooling_is_rejected(two_cliques):
    with pytest.raises(ValueError):
        train(two_cliques, Config(dims=4, passes=1, pooling="attention", neighbour_pooling="max", device="cpu"),
              log=lambda m: None)


@pytest.mark.parametrize("pool_negatives", [False, True])
def test_infonce_trains(two_cliques, pool_negatives):
    cfg = Config(dims=16, passes=40, batch_size=8, lr=0.05, negatives=3, loss="infonce", gamma=0.1,
                 negative_pool=16, pool_negatives=pool_negatives, device="cpu", seed=3)
    assert separation(node_embeddings(train(two_cliques, cfg, log=lambda m: None), two_cliques)) > 0.3


def test_pool_negatives_needs_infonce(two_cliques):
    with pytest.raises(ValueError):
        train(two_cliques, Config(dims=4, passes=1, loss="starspace", pool_negatives=True, device="cpu"),
              log=lambda m: None)


@pytest.mark.parametrize("pooling", ["mean", "attention"])
def test_offset_features_start_at_zero_and_learn(tmp_path, pooling):
    edges = tmp_path / "g.edges"
    edges.write_text("".join(f"q\tq{i}\tp\tp{(i + j) % 6}\t1\n" for i in range(6) for j in range(2)))
    feats = tmp_path / "g.features"
    feats.write_text("".join(f"q\tq{i}\tw{i} w{(i + 1) % 6}\np\tp{i}\tw{i} t{i}\n" for i in range(6)))
    g = data.load(str(edges), str(feats), shared_namespace="feat")
    offsets = g.zero_init
    shared = torch.tensor([i for i, (ns, _) in enumerate(g.feature_keys) if ns == "feat"])
    start = train(g, Config(dims=8, passes=0, pooling=pooling, heads=2, device="cpu"), log=lambda m: None)
    assert (start.features.weight[offsets] == 0).all()
    assert (start.features.weight[shared].float().norm(dim=1) > 0.5).all()
    cfg = Config(dims=8, passes=5, batch_size=4, negatives=2, pooling=pooling, heads=2, device="cpu")
    model = train(g, cfg, log=lambda m: None)
    assert (model.features.weight[offsets].float().norm(dim=1) > 0).any()
    assert torch.isfinite(node_embeddings(model, g)).all()



@pytest.mark.parametrize("pooling", ["mean", "attention"])
def test_epoch_updates_equal_batch_updates_with_one_batch_per_pass(two_cliques, pooling):
    common = dict(dims=8, passes=3, batch_size=64, lr=0.1, negatives=2, pooling=pooling, heads=2,
                  optimizer="rowwise_adagrad", device="cpu", seed=5)
    batch = train(two_cliques, Config(**common), log=lambda m: None)
    epoch = train(two_cliques, Config(**common, update_frequency="epoch"), log=lambda m: None)
    for a, b in zip(batch.state_dict().values(), epoch.state_dict().values()):
        assert torch.allclose(a.float(), b.float(), atol=1e-6)


def test_epoch_updates_step_once_per_pass(two_cliques, monkeypatch):
    from ep_gpu import optim
    steps, applies = [], []
    monkeypatch.setattr(optim.SparseTableOptimizer, "step", lambda self: steps.append(1))
    real_apply = optim.EpochGradients.apply
    monkeypatch.setattr(optim.EpochGradients, "apply", lambda self, o: (applies.append(1), real_apply(self, o)))
    cfg = Config(dims=8, passes=3, batch_size=4, update_frequency="epoch", device="cpu")
    train(two_cliques, cfg, log=lambda m: None)
    assert not steps and len(applies) == 3


@pytest.mark.parametrize("pooling", ["mean", "attention"])
def test_epoch_updates_learn_the_cliques(two_cliques, pooling):
    cfg = Config(dims=16, passes=60, batch_size=4, lr=0.2, loss="starspace", gamma=0.5, negatives=3,
                 pooling=pooling, heads=2, optimizer="rowwise_adagrad", update_frequency="epoch",
                 device="cpu", seed=3)
    assert separation(node_embeddings(train(two_cliques, cfg, log=lambda m: None), two_cliques)) > 0.3


def test_unknown_update_frequency_is_rejected(two_cliques):
    with pytest.raises(ValueError):
        train(two_cliques, Config(dims=4, passes=1, update_frequency="step", device="cpu"), log=lambda m: None)


def test_epoch_after_switches_from_batch_to_epoch_updates(two_cliques, monkeypatch):
    from ep_gpu import optim
    steps, applies = [], []
    real_step, real_apply = optim.SparseTableOptimizer.step, optim.EpochGradients.apply
    monkeypatch.setattr(optim.SparseTableOptimizer, "step", lambda self: (steps.append(1), real_step(self)))
    monkeypatch.setattr(optim.EpochGradients, "apply", lambda self, o: (applies.append(1), real_apply(self, o)))
    cfg = Config(dims=8, passes=5, batch_size=4, update_frequency="epoch", epoch_after=2, device="cpu")
    train(two_cliques, cfg, log=lambda m: None)
    assert len(steps) == 2 * 5 and len(applies) == 3     # 20 nodes in batches of 4: 5 per pass


def test_batch_then_epoch_updates_learn_the_cliques(two_cliques):
    cfg = Config(dims=16, passes=60, batch_size=4, lr=0.2, loss="starspace", gamma=0.5, negatives=3,
                 optimizer="rowwise_adagrad", update_frequency="epoch", epoch_after=20, device="cpu", seed=3)
    assert separation(node_embeddings(train(two_cliques, cfg, log=lambda m: None), two_cliques)) > 0.3


@pytest.mark.parametrize("kwargs", [dict(epoch_after=3), dict(update_frequency="epoch", epoch_after=0)])
def test_epoch_after_needs_epoch_updates_and_a_positive_count(two_cliques, kwargs):
    with pytest.raises(ValueError, match="epoch_after"):
        train(two_cliques, Config(dims=4, passes=1, device="cpu", **kwargs), log=lambda m: None)
