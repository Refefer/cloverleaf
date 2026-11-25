import math

import pytest
import torch

from ep_gpu import geometry, losses
from ep_gpu.geometry import EUCLIDEAN, Lorentz, Poincare


def mobius_distance(x, y, c):
    """Reference: d(x, y) = 2/sqrt(c) artanh(sqrt(c) |(-x) (+) y|), in float64."""
    x, y = x.double(), y.double()
    xy, xx, yy = (-x * y).sum(-1), (x * x).sum(-1), (y * y).sum(-1)
    num = (1 + 2 * c * xy + c * yy)[..., None] * -x + (1 - c * xx)[..., None] * y
    add = num / (1 + 2 * c * xy + c * c * xx * yy)[..., None]
    return 2 / math.sqrt(c) * torch.atanh(math.sqrt(c) * add.norm(dim=-1))


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_exp_map_lands_in_the_ball_and_inverts(c):
    space = Poincare(c)
    v = torch.randn(64, 7) * torch.logspace(-3, 1, 64)[:, None]
    x = space.to_space(v)
    assert (x.norm(dim=-1) < 1 / math.sqrt(c)).all()
    small = v.norm(dim=-1) < 3
    assert torch.allclose(space.log0(x)[small], v[small], atol=1e-4, rtol=1e-3)
    # Far out, points stop short of the edge rather than reaching it.
    far = space.to_space(torch.full((1, 3), 1e6))
    assert far.norm() < 1 / math.sqrt(c)


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_distance_matches_the_mobius_formula(c):
    space = Poincare(c)
    x, y = space.to_space(torch.randn(50, 5)), space.to_space(torch.randn(50, 5) * 2)
    expected = mobius_distance(x, y, c).float()
    assert torch.allclose(space.distance(x, y), expected, rtol=1e-3, atol=1e-3)
    assert torch.allclose(space.distance(y, x), expected, rtol=1e-3, atol=1e-3)
    assert torch.allclose(space.pairwise_distance(x, y).diagonal(), expected, rtol=1e-3, atol=1e-3)
    assert space.distance(x, x).abs().max() < 1e-3


def test_distance_grows_towards_the_edge():
    """Equal Euclidean steps cover ever more hyperbolic distance near the boundary."""
    space = Poincare()
    r = torch.tensor([[0.0, 0], [0.5, 0], [0.9, 0], [0.99, 0]])
    steps = space.distance(r[:-1], r[1:])
    assert (steps[1:] > steps[:-1]).all()
    assert torch.allclose(space.distance(r[:1], r[1:2]), torch.tensor([2 * math.atanh(0.5)]), atol=1e-5)


def test_gradients_stay_finite_where_points_coincide_or_vanish():
    space = Poincare()
    v = torch.randn(4, 6, requires_grad=True)
    zero = torch.zeros(2, 6, requires_grad=True)
    x = space.to_space(torch.cat([v, zero]))
    pool = torch.cat([x, space.to_space(torch.full((1, 6), 50.0))])   # itself and a near-edge point
    d = space.pairwise_distance(x, pool)
    # Masked out, as a pool slot holding the anchor itself is.
    d = d.masked_fill(torch.eye(len(x), len(pool), dtype=torch.bool), 0.0)
    (d.sum() + space.distance(x, x).sum()).backward()
    assert torch.isfinite(v.grad).all() and torch.isfinite(zero.grad).all()


def test_euclidean_space_keeps_cosine_and_euclidean_scores():
    t, h, hu = torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 5, 4)
    cos = torch.nn.functional.cosine_similarity
    pos, neg = losses.similarities(t, h, hu=hu)
    assert torch.allclose(pos, cos(h, t, dim=-1), atol=1e-6)
    assert torch.allclose(neg, cos(hu, h[:, None], dim=-1), atol=1e-6)
    assert torch.allclose(EUCLIDEAN.distance(t, h), (t - h).norm(dim=-1))


def test_poincare_losses_score_by_distance():
    space = Poincare()
    t, h, hu = (space.to_space(torch.randn(*s)) for s in [(3, 4), (3, 4), (3, 5, 4)])
    pos, neg = losses.similarities(t, h, hu=hu, space=space)
    assert torch.allclose(pos, -space.distance(h, t))
    expected = torch.relu(1.0 + space.distance(t, h)[:, None] - space.distance(t[:, None], hu))
    got = losses.margin(t, h, hu, 1.0, space)
    count = (expected > 0).sum(1).clamp(min=1)
    assert torch.allclose(got, expected.sum(1) / count)


def test_rank_loss_refuses_hyperbolic_space():
    space = Poincare()
    with pytest.raises(ValueError, match="Euclidean"):
        losses.rank(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 3, 2), 0.5, space)


def test_make_rejects_unknown_geometries_and_bad_curvature():
    assert geometry.make("euclidean") is EUCLIDEAN
    with pytest.raises(ValueError, match="geometry"):
        geometry.make("spherical")
    with pytest.raises(ValueError, match="curvature"):
        geometry.make("poincare", 0.0)


def test_gyromidpoint_bisects_the_geodesic_and_keeps_a_single_point():
    from ep_gpu.model import weighted_mean
    torch.manual_seed(0)
    space = Poincare(1.5, pooling="midpoint")
    a, b = space.exp0(torch.randn(20, 4)), space.exp0(torch.randn(20, 4) * 2)
    m = weighted_mean(torch.stack([a, b], 1), torch.ones(20, 2), space)
    half = space.distance(a, b) / 2
    assert torch.allclose(space.distance(a, m), half, rtol=1e-3, atol=1e-3)
    assert torch.allclose(space.distance(m, b), half, rtol=1e-3, atol=1e-3)
    alone = weighted_mean(torch.stack([a, b], 1), torch.tensor([[1.0, 0.0]] * 20), space)
    assert torch.allclose(alone, a, atol=1e-3)   # float32, through artanh near the edge
    opposite = weighted_mean(torch.stack([a, -a], 1), torch.ones(20, 2), space)
    assert opposite.norm(dim=-1).max() < 1e-5


@pytest.mark.parametrize("pooling", ["tangent", "midpoint"])
def test_scatter_pooling_matches_dense_pooling(pooling):
    from ep_gpu.model import EPModel, weighted_mean
    space = Poincare(1.0, pooling=pooling)
    model = EPModel(30, 6, space=space, seed=0)
    ids = torch.randint(0, 30, (8, 5))
    w = (torch.rand(8, 5) > 0.3).float()
    w[0] = 0                                                       # a row without tokens
    got = model.embed(ids, w)
    rows = space.feature_points(model.features.weight[ids].float())
    expected = space.to_space(weighted_mean(rows, w, space))
    assert torch.allclose(got, expected, atol=1e-5)
    assert got[0].abs().max() == 0                                 # the origin


def test_midpoint_pooling_refuses_attention():
    from ep_gpu.model import EPModel
    with pytest.raises(ValueError, match="midpoint"):
        EPModel(10, 8, pooling="attention", space=Poincare(pooling="midpoint"))


def test_clipping_bounds_the_distance_from_the_origin():
    space = Poincare(1.0, max_tangent_norm=1.5)
    x = space.to_space(torch.randn(100, 8) * 10)
    assert space.origin_distance(x).max() <= 2 * 1.5 + 1e-4      # d(0, exp0(v)) = 2|v| at c = 1


def test_ball_mean_matches_the_scatter_midpoint():
    from ep_gpu.model import weighted_mean
    torch.manual_seed(1)
    space = Poincare(0.5, pooling="midpoint")
    x, w = space.exp0(torch.randn(6, 4, 3)), torch.rand(6, 4)
    assert torch.allclose(space.ball_mean(x, w), weighted_mean(x, w, space), atol=1e-5)


def test_neighbour_midpoint_reconstructs_from_the_neighbours_points():
    from ep_gpu.model import EPModel
    space = Poincare(1.0, pooling="neighbour_midpoint")
    model = EPModel(10, 4, space=space, seed=0)
    pooled = torch.randn(3, 5, 4)                                   # tangent vectors per neighbour
    w = torch.rand(3, 5)
    expected = space.ball_mean(space.exp0(pooled), w)
    assert torch.allclose(model.reconstruct(pooled, w), expected)
    # Features still pool in the tangent space: a single feature's node is exp0 of its row.
    row = model.features.weight[2:3].float()
    assert torch.allclose(model.embed(torch.tensor([[2]]), torch.ones(1, 1)), space.exp0(row), atol=1e-6)


def test_margin_sq_with_the_adaptive_margin():
    space = Poincare()
    torch.manual_seed(2)
    t, h, hu = (space.to_space(torch.randn(*s)) for s in [(4, 3), (4, 3), (4, 6, 3)])
    plain = losses.margin_sq(t, h, hu, 1.0, space)
    d_pos, d_neg = space.distance(t, h), space.distance(t[:, None], hu)
    terms = torch.relu(1.0 + d_pos[:, None] ** 2 - d_neg ** 2)
    assert torch.allclose(plain, terms.sum(1) / (terms > 0).sum(1).clamp(min=1))
    adaptive = losses.margin_sq(t, h, hu, 1.0, space, adaptive=True)
    assert torch.isfinite(adaptive).all() and (adaptive <= plain + 1e-6).all()   # sigmoid < 1


@pytest.mark.parametrize("kwargs", [
    dict(loss="margin_sq", gamma=0.5, negatives=4),
    dict(loss="margin_sq", gamma=0.5, hard_negatives=4, adaptive_margin=True),
    dict(loss="starspace", gamma=1.0, hard_negatives=2, max_tangent_norm=2.0, curvature=0.3),
    dict(loss="margin", gamma=1.0, negatives=4, reconstruction="nested", poincare_pooling="neighbour_midpoint"),
])
def test_poincare_variants_train_and_separate_two_cliques(tmp_path, kwargs):
    from ep_gpu import data
    from ep_gpu.train import Config, node_embeddings, train
    path = tmp_path / "g.edges"
    with open(path, "w") as f:
        for base in (0, 10):
            for i in range(base, base + 10):
                for j in range(i + 1, base + 10):
                    f.write(f"n\t{i}\tn\t{j}\t1\n")
    graph = data.load(str(path))
    cfg = Config(dims=8, passes=40, batch_size=4, lr=0.1, negative_pool=16, geometry="poincare",
                 device="cpu", **kwargs)
    model = train(graph, cfg, log=lambda m: None)
    X = node_embeddings(model, graph)
    D = model.space.pairwise_distance(X, X)
    same = torch.tensor([[(i < 10) == (j < 10) for j in range(20)] for i in range(20)])
    off = ~torch.eye(20, dtype=torch.bool)
    assert D[same & off].mean() < D[~same].mean()



def test_attention_keeping_the_norm_changes_only_direction():
    from ep_gpu.model import AttentionPool, weighted_mean
    torch.manual_seed(5)
    pool = AttentionPool(16, 4, keeps_norm=True)
    torch.nn.init.normal_(pool.out.weight, std=0.5)                 # as if trained
    x, w = torch.randn(6, 5, 16), (torch.rand(6, 5) > 0.2).float()
    w[0] = 0                                                         # a row without tokens
    mean, pooled = weighted_mean(x, w), pool(x, w)
    assert torch.allclose(pooled.norm(dim=-1), mean.norm(dim=-1), atol=1e-5)
    assert pooled[0].abs().max() == 0
    free = AttentionPool(16, 4)
    free.load_state_dict(pool.state_dict())
    assert torch.allclose(torch.nn.functional.normalize(free(x, w)[1:], dim=-1),
                          torch.nn.functional.normalize(pooled[1:], dim=-1), atol=1e-5)
    assert not torch.allclose(free(x, w)[1:].norm(dim=-1), mean[1:].norm(dim=-1))


def test_attention_keeping_the_norm_trains_in_the_ball(tmp_path):
    from ep_gpu import data
    from ep_gpu.train import Config, node_embeddings, train
    path, feats = tmp_path / "g.edges", tmp_path / "g.features"
    with open(path, "w") as f, open(feats, "w") as g:
        for base in (0, 10):
            for i in range(base, base + 10):
                g.write(f"n\t{i}\tid{i} group{base} shared\n")
                for j in range(i + 1, base + 10):
                    f.write(f"n\t{i}\tn\t{j}\t1\n")
    graph = data.load(str(path), str(feats))
    cfg = Config(dims=16, passes=40, batch_size=4, lr=0.1, negative_pool=16, geometry="poincare", curvature=0.3,
                 pooling="attention", heads=2, attention_keeps_norm=True, loss="starspace", gamma=1.0,
                 negatives=4, device="cpu")
    model = train(graph, cfg, log=lambda m: None)
    X = node_embeddings(model, graph)
    D = model.space.pairwise_distance(X, X)
    same = torch.tensor([[(i < 10) == (j < 10) for j in range(20)] for i in range(20)])
    off = ~torch.eye(20, dtype=torch.bool)
    assert D[same & off].mean() < D[~same].mean()
    from ep_gpu.train import check_loss
    with pytest.raises(ValueError, match="attention_keeps_norm"):
        check_loss(Config(attention_keeps_norm=True))


def test_attention_dropout_only_in_training():
    from ep_gpu.model import AttentionPool, weighted_mean
    torch.manual_seed(6)
    pool = AttentionPool(16, 4, dropout=0.5)
    torch.nn.init.normal_(pool.out.weight, std=0.5)
    x, w = torch.randn(8, 5, 16), torch.ones(8, 5)
    pool.eval()
    a, b = pool(x, w), pool(x, w)
    assert torch.equal(a, b)                                          # inference: deterministic, whole term
    pool.train()
    c = pool(x, w)
    assert not torch.allclose(c, a)
    mean = weighted_mean(x, w)
    # Some of the added term's entries are dropped: there the pooled vector is the mean itself.
    assert ((c - mean).abs() < 1e-7).float().mean() > 0.3
    with pytest.raises(ValueError, match="dropout"):
        AttentionPool(16, 4, dropout=1.0)


def test_attention_weights_mode_is_a_convex_combination_that_starts_as_the_mean():
    from ep_gpu.model import AttentionPool, weighted_mean
    torch.manual_seed(7)
    pool = AttentionPool(16, 4, mode="weights")
    x, w = torch.randn(6, 5, 16), (torch.rand(6, 5) > 0.2).float()
    w[0] = 0
    assert torch.allclose(pool(x, w), weighted_mean(x, w), atol=1e-6)      # keys at zero: the mean
    torch.nn.init.normal_(pool.qkv.weight, std=0.5)                        # as if trained
    b = pool.importance(pool.qkv(x), w)
    assert torch.allclose(b[1:].sum(1), torch.ones(5), atol=1e-5) and (b >= 0).all() and (b[w == 0] == 0).all()
    assert torch.allclose(pool(x, w), (b[..., None] * x).sum(1), atol=1e-5)
    assert not torch.allclose(pool(x, w)[1:], weighted_mean(x, w)[1:], atol=1e-3)
    assert pool(x, w)[0].abs().max() == 0
    assert not hasattr(pool, "out")
    with pytest.raises(ValueError, match="transform"):
        AttentionPool(16, 4, keeps_norm=True, mode="weights")


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_attention_weights_mode_learns_feature_importance_in_the_ball(tmp_path, seed):
    """Every node has its clique's feature, one shared by all and a noise feature: weighting
    learns the clique feature matters more than the noise, and the cliques separate."""
    from ep_gpu import data, sampling
    from ep_gpu.train import Config, node_embeddings, train
    path, feats = tmp_path / "g.edges", tmp_path / "g.features"
    with open(path, "w") as f, open(feats, "w") as g:
        for base in (0, 10):
            for i in range(base, base + 10):
                g.write(f"n\t{i}\tgroup{base} shared noise{i % 4}\n")
                for j in range(i + 1, base + 10):
                    f.write(f"n\t{i}\tn\t{j}\t1\n")
    graph = data.load(str(path), str(feats))
    cfg = Config(dims=16, passes=40, batch_size=4, lr=0.1, dense_lr=0.01, negative_pool=16, geometry="poincare",
                 curvature=0.3, max_tangent_norm=2.0, pooling="attention", heads=2, attention_mode="weights",
                 loss="starspace", gamma=1.0, negatives=4, device="cpu", seed=seed)
    model = train(graph, cfg, log=lambda m: None)
    model.eval()
    X = node_embeddings(model, graph)
    D = model.space.pairwise_distance(X, X)
    same = torch.tensor([[(i < 10) == (j < 10) for j in range(20)] for i in range(20)])
    off = ~torch.eye(20, dtype=torch.bool)
    assert D[same & off].mean() < D[~same].mean()
    fidx, fmask = sampling.sample_features(graph, torch.arange(graph.num_nodes), None)
    with torch.no_grad():
        b = model.pool.importance(model.pool.qkv(model.features(fidx).float()), fmask.float())
    kind = [[graph.feature_keys[int(f)][1].rstrip("0123456789") for f in row] for row in fidx]
    mean_weight = lambda name: float(torch.stack([b[r, j] for r in range(len(kind)) for j in range(len(kind[r]))
                                                  if fmask[r, j] and kind[r][j] == name]).mean())
    assert mean_weight("group") > mean_weight("noise")


def test_feature_importance_starts_as_the_mean_and_weights_features():
    from ep_gpu.model import EPModel
    model = EPModel(10, 4, feature_importance=True, seed=0)
    ids, w = torch.tensor([[1, 2, 3], [4, 5, 0]]), torch.tensor([[1.0, 1, 1], [1, 1, 0]])
    rows = model.features.weight[ids].float()
    mean = (rows * w[..., None]).sum(1) / w.sum(1, keepdim=True)
    assert torch.allclose(model.embed(ids, w), mean, atol=1e-6)
    with torch.no_grad():
        model.importance.weight[2] = 2.0
    e = torch.exp(model.importance.weight[ids, 0].detach()) * w
    expected = (rows * e[..., None]).sum(1) / e.sum(1, keepdim=True)
    assert torch.allclose(model.embed(ids, w), expected, atol=1e-6)
    assert all(not n.startswith("importance") for n, _ in model.named_parameters() if any(p is _ for p in model.dense_parameters()))


@pytest.mark.parametrize("geometry", ["euclidean", "poincare"])
def test_feature_importance_learns_and_trains(tmp_path, geometry):
    from ep_gpu import data
    from ep_gpu.train import Config, node_embeddings, train
    path, feats = tmp_path / "g.edges", tmp_path / "g.features"
    with open(path, "w") as f, open(feats, "w") as g:
        for base in (0, 10):
            for i in range(base, base + 10):
                g.write(f"n\t{i}\tgroup{base} shared noise{i % 4}\n")
                for j in range(i + 1, base + 10):
                    f.write(f"n\t{i}\tn\t{j}\t1\n")
    graph = data.load(str(path), str(feats))
    extra = dict(curvature=0.3, max_tangent_norm=2.0, gamma=1.0) if geometry == "poincare" else dict(gamma=0.5)
    cfg = Config(dims=16, passes=40, batch_size=4, lr=0.1, negative_pool=16, geometry=geometry, loss="starspace",
                 negatives=4, feature_importance=True, device="cpu", **extra)
    model = train(graph, cfg, log=lambda m: None)
    # The scalars learn.  Which features they favour depends on the objective: with random negatives
    # drawn from the anchor's own clique, features that tell clique-mates apart (noise) are what
    # push those negatives away, so they can gain weight; the test checks mechanics, not that.
    assert model.importance.weight.abs().max() > 0.05
    X = node_embeddings(model, graph)
    D = model.space.pairwise_distance(X, X)
    same = torch.tensor([[(i < 10) == (j < 10) for j in range(20)] for i in range(20)])
    off = ~torch.eye(20, dtype=torch.bool)
    assert D[same & off].mean() < D[~same].mean()


@pytest.mark.parametrize("kwargs, representable", [
    (dict(), True),
    (dict(pooling="attention", heads=2), True),
    (dict(pooling="attention", heads=2, attention_mode="weights"), False),
    (dict(pooling="attention", heads=2, attention_keeps_norm=True), False),
    (dict(feature_importance=True), False),
    (dict(geometry="poincare"), True),                       # tangent pooling: the aggregator records the ball
    (dict(geometry="poincare", poincare_pooling="midpoint"), False),
    (dict(geometry="lorentz"), True),                        # written as the isometric ball
])
def test_export_writes_the_aggregator_only_when_cloverleaf_can_reproduce_it(tmp_path, kwargs, representable):
    from ep_gpu import data
    from ep_gpu.train import Config, export, train
    path = tmp_path / "g.edges"
    with open(path, "w") as f:
        for i in range(6):
            f.write(f"n\t{i}\tn\t{(i + 1) % 6}\t1\n")
    graph = data.load(str(path))
    cfg = Config(dims=8, passes=1, batch_size=4, device="cpu", **kwargs)
    model = train(graph, cfg, log=lambda m: None)
    prefix = str(tmp_path / "m")
    export(model, graph, cfg, prefix)
    import os
    assert os.path.exists(prefix + ".pt") and os.path.exists(prefix + ".node-embeddings")
    assert os.path.exists(prefix + ".aggregator") == representable


# The hyperboloid (Lorentz) model


@pytest.mark.parametrize("c", [0.3, 1.0, 2.0])
def test_lorentz_points_lie_on_the_hyperboloid_and_invert(c):
    space = Lorentz(c)
    v = torch.randn(64, 7) * torch.logspace(-3, 0, 64)[:, None] / math.sqrt(c)
    x = space.to_space(v)
    assert x.shape == (64, 8) and x.dtype == torch.float64 and (x[:, 0] > 0).all()
    inner = -x[:, 0] ** 2 + (x[:, 1:] ** 2).sum(-1)
    assert torch.allclose(inner * c, torch.full_like(inner, -1.0), rtol=1e-6)
    assert torch.allclose(space.log0(x), v, atol=1e-4, rtol=1e-3)
    # The origin is the zero tangent vector.
    assert torch.allclose(space.to_space(torch.zeros(1, 7))[0, 1:].float(), torch.zeros(7))


@pytest.mark.parametrize("c", [0.3, 1.0, 2.0])
def test_lorentz_and_poincare_are_isometric(c):
    """The same tangent vectors give the same points (through the isometry) and distances."""
    lorentz, ball = Lorentz(c), Poincare(c)
    u = torch.randn(40, 6) * 0.8
    v = torch.randn(40, 6) * 0.8
    xl, yl = lorentz.to_space(u), lorentz.to_space(v)
    xb, yb = ball.to_space(u), ball.to_space(v)
    assert torch.allclose(lorentz.to_poincare(xl), xb, atol=1e-5)
    assert torch.allclose(lorentz.distance(xl, yl), ball.distance(xb, yb), atol=1e-3, rtol=1e-3)
    assert torch.allclose(lorentz.pairwise_distance(xl, yl), lorentz.distance(xl[:, None], yl[None, :]),
                          atol=1e-3, rtol=1e-3)
    assert torch.allclose(lorentz.origin_distance(xl), ball.origin_distance(xb), atol=1e-4, rtol=1e-4)


def test_lorentz_keeps_precision_where_the_ball_clamps():
    """Past the ball's clamp (sqrt(c) |v| > 6.1) it pins points at its edge; the hyperboloid
    still tells them apart."""
    lorentz, ball = Lorentz(1.0), Poincare(1.0)
    v = torch.tensor([[7.0, 0.0], [8.0, 0.0]])
    assert ball.distance(*ball.to_space(v)) < 0.01
    assert torch.isclose(lorentz.distance(*lorentz.to_space(v)), torch.tensor(2.0), atol=1e-3)  # 2 per unit of |v|


def test_lorentz_gradients_stay_finite_where_points_coincide_or_vanish():
    space = Lorentz(0.5)
    v = torch.tensor([[0.3, -0.2], [0.3, -0.2], [0.0, 0.0]], requires_grad=True)
    x = space.to_space(v)
    (space.distance(x[0], x[1]) + space.distance(x[0], x[2]) + space.origin_distance(x).sum()).backward()
    assert torch.isfinite(v.grad).all()


def test_lorentz_rejects_ball_pooling_and_bad_curvature():
    with pytest.raises(ValueError):
        geometry.make("lorentz", 1.0, "midpoint")
    with pytest.raises(ValueError):
        geometry.make("lorentz", 0.0)
    from ep_gpu.train import Config, check_loss
    with pytest.raises(ValueError):
        check_loss(Config(geometry="lorentz", poincare_pooling="midpoint"))
    with pytest.raises(ValueError):
        check_loss(Config(geometry="lorentz", loss="rank"))


@pytest.mark.parametrize("kwargs", [
    dict(loss="starspace", gamma=1.0, hard_negatives=2, curvature=0.3),
    dict(loss="margin", gamma=1.0, negatives=4, max_tangent_norm=2.0),
])
def test_lorentz_trains_and_separates_two_cliques(tmp_path, kwargs):
    from ep_gpu import data
    from ep_gpu.train import Config, node_embeddings, train
    path = tmp_path / "g.edges"
    with open(path, "w") as f:
        for base in (0, 10):
            for i in range(base, base + 10):
                for j in range(i + 1, base + 10):
                    f.write(f"n\t{i}\tn\t{j}\t1\n")
    graph = data.load(str(path))
    cfg = Config(dims=8, passes=40, batch_size=4, lr=0.1, negative_pool=16, geometry="lorentz",
                 device="cpu", **kwargs)
    model = train(graph, cfg, log=lambda m: None)
    X = node_embeddings(model, graph)
    assert X.shape == (20, 9)
    D = model.space.pairwise_distance(X, X)
    same = torch.tensor([[(i < 10) == (j < 10) for j in range(20)] for i in range(20)])
    off = ~torch.eye(20, dtype=torch.bool)
    assert D[same & off].mean() < D[~same].mean()


@pytest.mark.parametrize("kwargs", [
    dict(loss="starspace", gamma=1.0, hard_negatives=2, curvature=0.3),
    dict(loss="margin_sq", gamma=0.5, pooling="attention", heads=2, dense_lr=0.01),
])
def test_lorentz_trains_as_the_isometric_poincare_model(tmp_path, kwargs):
    """Same parameters, seed and batches: the hyperboloid and the ball learn the same model, to
    float rounding, since every distance agrees."""
    from ep_gpu import data
    from ep_gpu.train import Config, node_embeddings, train
    path = tmp_path / "g.edges"
    with open(path, "w") as f:
        for i in range(12):
            for j in (i + 1, i + 3):
                f.write(f"n\t{i}\tn\t{j % 12}\t1\n")
    graph = data.load(str(path))
    D = {}
    for geo in ("poincare", "lorentz"):
        cfg = Config(dims=6, passes=10, batch_size=4, lr=0.05, negative_pool=8, geometry=geo,
                     precision="fp32", device="cpu", **kwargs)
        model = train(graph, cfg, log=lambda m: None)
        X = node_embeddings(model, graph)
        D[geo] = model.space.distance(X[:, None], X[None, :])
    assert torch.allclose(D["poincare"], D["lorentz"], atol=1e-2, rtol=1e-2)
