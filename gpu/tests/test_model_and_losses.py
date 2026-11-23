import math

import pytest
import torch

from ep_gpu import losses
from ep_gpu.model import AttentionPool, EPModel, MeanPool, weighted_mean


def test_mean_pool_ignores_padding_and_empty_rows_are_zero():
    x = torch.tensor([[[1., 0.], [3., 2.], [100., 100.]],
                      [[5., 5.], [7., 7.], [9., 9.]]])
    w = torch.tensor([[1., 1., 0.], [0., 0., 0.]])
    out = MeanPool()(x, w)
    assert torch.allclose(out[0], torch.tensor([2., 1.]))
    assert torch.allclose(out[1], torch.zeros(2))


def test_weighted_mean_uses_weights():
    x = torch.tensor([[[0.], [4.]]])
    assert torch.allclose(weighted_mean(x, torch.tensor([[3., 1.]])), torch.tensor([[1.]]))


def test_attention_starts_as_mean_pooling():
    torch.manual_seed(0)
    pool = AttentionPool(8, 2)
    x = torch.randn(3, 5, 8)
    w = torch.tensor([[1., 1., 1., 0., 0.], [1., 0., 0., 0., 0.], [2., 1., 1., 1., 1.]])
    assert torch.allclose(pool(x, w), weighted_mean(x, w), atol=1e-6)


def trained_pool():
    torch.manual_seed(0)
    pool = AttentionPool(8, 2)
    torch.nn.init.normal_(pool.out.weight)
    return pool


def test_attention_ignores_padded_tokens():
    pool = trained_pool()
    x = torch.randn(2, 5, 8)
    w = torch.tensor([[1., 1., 1., 0., 0.], [1., 1., 0., 0., 0.]])
    changed = x.clone()
    changed[:, 3:] = torch.randn(2, 2, 8) * 100
    assert torch.allclose(pool(x, w), pool(changed, w), atol=1e-5)


def test_attention_is_permutation_invariant():
    pool = trained_pool()
    x = torch.randn(1, 4, 8)
    w = torch.ones(1, 4)
    perm = torch.tensor([2, 0, 3, 1])
    assert torch.allclose(pool(x, w), pool(x[:, perm], w[:, perm]), atol=1e-5)


def test_attention_row_without_tokens_is_zero_and_finite():
    pool = trained_pool()
    out = pool(torch.randn(1, 3, 8), torch.zeros(1, 3))
    assert torch.isfinite(out).all() and torch.allclose(out, torch.zeros(1, 8))


@pytest.mark.parametrize("init", ["unit", "glorot"])
def test_init(init):
    m = EPModel(50, 16, init=init, seed=1)
    w = m.features.weight
    if init == "unit":
        assert torch.allclose(w.norm(dim=1), torch.ones(50), atol=1e-5)
    else:
        assert w.abs().max() <= math.sqrt(6 / 66) + 1e-6


def test_margin_averages_only_positive_terms():
    t = torch.tensor([[0., 0.]])
    h = torch.tensor([[1., 0.]])                      # |t - h| = 1
    hu = torch.tensor([[[3., 0.], [0., 1.5]]])        # distances 3 and 1.5
    # gamma 1: terms max(0, 2 - 3) = 0 and max(0, 2 - 1.5) = 0.5 -> mean of positive = 0.5
    assert torch.allclose(losses.margin(t, h, hu, 1.0), torch.tensor([0.5]))
    # gamma 0.1: both terms zero -> 0
    assert torch.allclose(losses.margin(t, h, hu, 0.1), torch.tensor([0.0]))


def test_starspace_compares_negatives_with_h():
    t = torch.tensor([[1., 0.]])
    h = torch.tensor([[1., 1.]])                      # cos(t, h) = 0.7071
    hu = torch.tensor([[[0., 2.], [-1., -1.]]])       # cos(h, hu) = 0.7071 and -1
    out = losses.starspace(t, h, hu, 0.5)
    # terms: max(0, 0.5 - 0) = 0.5 and max(0, 0.5 - 1.7071) = 0 -> 0.5
    assert torch.allclose(out, torch.tensor([0.5]), atol=1e-5)


def test_rank_loss_is_negative_log_probability_below_tau():
    t = torch.tensor([[1., 0.]])
    h = torch.tensor([[1., 0.]])
    hu = torch.tensor([[[0., 0.]]])
    # logits [0, 1] -> p(positive) = e / (1 + e) = 0.731
    p = math.e / (1 + math.e)
    assert torch.allclose(losses.rank(t, h, hu, 0.9), torch.tensor([-math.log(p)]), atol=1e-5)
    assert torch.allclose(losses.rank(t, h, hu, 0.5), torch.tensor([0.0]))


def test_rankspace_is_the_sum():
    torch.manual_seed(0)
    t, h, hu = torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 2, 3)
    assert torch.allclose(losses.rankspace(t, h, hu, 0.4),
                          losses.starspace(t, h, hu, 0.4) + losses.rank(t, h, hu, 0.4))


@pytest.mark.parametrize("pooling", ["mean", "attention"])
def test_deduplicated_embedding_matches_the_padded_reference(pooling):
    torch.manual_seed(0)
    m = EPModel(30, 8, pooling=pooling, heads=2, seed=0)
    if pooling == "attention":
        torch.nn.init.normal_(m.pool.out.weight)
    # Repeated features within and across rows and groups, padding, and a row with no tokens.
    a = (torch.tensor([[1, 2, 2, 0], [3, 4, 0, 0]]), torch.tensor([[1., 1., 2., 0.], [1., 1., 0., 0.]]))
    b = (torch.tensor([[2, 5, 1], [0, 0, 0]]), torch.tensor([[0.5, 1., 1.], [0., 0., 0.]]))
    got = m.embed_groups([a, b])
    for (f, w), out in zip([a, b], got):
        assert torch.allclose(out, m.pool(m.features(f), w), atol=1e-5)


def test_deduplicated_embedding_gives_the_same_gradients():
    torch.manual_seed(0)
    f = torch.tensor([[1, 2, 2], [2, 3, 0]])
    w = torch.tensor([[1., 1., 1.], [1., 2., 0.]])
    grads = []
    for dedup in (True, False):
        m = EPModel(5, 4, seed=1)
        out = m.embed(f, w) if dedup else m.pool(m.features(f), w)
        (out ** 2).sum().backward()
        grads.append(m.features.weight.grad.to_dense())
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


@pytest.mark.parametrize("dims,heads", [(151, 1), (151, 4), (8, 3), (3, 4)])
def test_attention_handles_any_dims_and_heads(dims, heads):
    pool = AttentionPool(dims, heads)
    assert pool.head_dims % 8 == 0
    torch.nn.init.normal_(pool.out.weight)
    x, w = torch.randn(2, 5, dims), torch.ones(2, 5)
    out = pool(x, w)
    assert out.shape == (2, dims) and torch.isfinite(out).all()


def test_infonce_is_cross_entropy_over_scaled_cosines():
    t = torch.tensor([[1., 0.]])
    h = torch.tensor([[2., 0.]])
    hu = torch.tensor([[[0., 3.], [-1., 0.]]])
    logits = torch.tensor([1.0, 0.0, -1.0]) / 0.5
    expected = -torch.log_softmax(logits, dim=0)[0]
    assert torch.allclose(losses.infonce(t, h, hu, 0.5), expected[None], atol=1e-6)


def test_infonce_pool_matches_gathering_every_pool_node():
    torch.manual_seed(0)
    t, h, pool = torch.randn(3, 4), torch.randn(3, 4), torch.randn(5, 4)
    nodes, pool_nodes = torch.tensor([7, 2, 9]), torch.tensor([1, 2, 3, 4, 5])
    exclude = pool_nodes[None, :] == nodes[:, None]
    got = losses.infonce_pool(t, h, pool, exclude, 0.1)
    for i in range(3):
        keep = ~exclude[i]
        ref = losses.infonce(t[i:i + 1], h[i:i + 1], pool[keep][None], 0.1)
        assert torch.allclose(got[i], ref[0], atol=1e-5)


def _listwise_case():
    torch.manual_seed(0)
    pos_logits = torch.randn(3, 4, dtype=torch.float64)
    neg_logits = torch.randn(3, 5, dtype=torch.float64)
    pos_w = torch.tensor([[0.5, 0.3, 0.2, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                         dtype=torch.float64)
    return pos_logits, pos_w, neg_logits


def test_listnet_is_cross_entropy_against_the_list_weights():
    pos_logits, pos_w, neg_logits = _listwise_case()
    got = losses.listnet(pos_logits, pos_w, neg_logits)
    for b in range(3):
        k = int((pos_w[b] > 0).sum())
        log_p = torch.log_softmax(torch.cat([pos_logits[b, :k], neg_logits[b]]), 0)
        assert torch.allclose(got[b], -(pos_w[b, :k] * log_p[:k]).sum())


def test_passive_listnet_skips_positives_already_as_likely_as_their_weight():
    pos_logits = torch.tensor([[5.0, -5.0]])
    pos_w = torch.tensor([[0.5, 0.5]])
    neg_logits = torch.tensor([[-5.0]])
    log_p = torch.log_softmax(torch.tensor([5.0, -5.0, -5.0]), 0)
    assert torch.allclose(losses.listnet(pos_logits, pos_w, neg_logits, passive=True), -0.5 * log_p[1])
    assert torch.allclose(losses.listnet(pos_logits, pos_w, neg_logits), -0.5 * log_p[:2].sum())


def test_listmle_scores_only_the_positives_positions():
    pos_logits, pos_w, neg_logits = _listwise_case()
    got = losses.listmle(pos_logits, pos_w, neg_logits)
    for b in range(3):
        k = int((pos_w[b] > 0).sum())
        logits = torch.cat([pos_logits[b, :k], neg_logits[b]])
        want = sum(torch.logsumexp(logits[i:], 0) - logits[i] for i in range(k))
        assert torch.allclose(got[b], torch.as_tensor(want, dtype=got.dtype))
    shuffled = losses.listmle(pos_logits, pos_w, neg_logits.flip(1))
    assert torch.allclose(got, shuffled)


def test_masked_negatives_drop_out_and_keep_gradients_finite():
    pos_logits, pos_w, neg_logits = _listwise_case()
    pos_logits.requires_grad_(True)
    masked = torch.cat([neg_logits, torch.full((3, 2), losses.MASKED, dtype=torch.float64)], 1)
    for loss in (losses.listnet, losses.listmle):
        assert torch.allclose(loss(pos_logits, pos_w, masked), loss(pos_logits, pos_w, neg_logits))
        loss(pos_logits, pos_w, masked).sum().backward()
        assert torch.isfinite(pos_logits.grad).all()
        pos_logits.grad = None
