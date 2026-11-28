"""EP's losses, batched, with cloverleaf's semantics (src/algos/ep/loss.rs).

Each takes t = h~(v) (B, d), h = h(v) (B, d) and hu = h(u) for the negatives (B, n, d), and returns
one loss per node (B,).  Margin and StarSpace average only their non-zero per-negative terms, and
are 0 when all are zero, as cloverleaf does.

`space` (geometry.py) says how embeddings compare: in Euclidean space, by cosine similarity, and
by Euclidean distance for the margin loss, as cloverleaf does; in the Poincaré ball, by negative
hyperbolic distance, and by that distance for the margin loss.  The embeddings passed in are
already in that space (space.to_space).
"""
import torch

from .geometry import EUCLIDEAN


def _mean_of_positive(terms):
    positive = terms > 0
    count = positive.sum(dim=1)
    return torch.where(count > 0, (terms * positive).sum(dim=1) / count.clamp(min=1),
                       torch.zeros_like(count, dtype=terms.dtype))


def margin(t, h, hu, gamma, space=EUCLIDEAN):
    """max(0, gamma + d(t, h) - d(t, hu_i)), with the space's distance."""
    d_pos = space.distance(t, h)
    d_neg = space.distance(t[:, None, :], hu)
    return _mean_of_positive(torch.relu(gamma + d_pos[:, None] - d_neg))


def margin_sq(t, h, hu, gamma, space=EUCLIDEAN, adaptive=False):
    """max(0, m + d(t, h)^2 - d(t, hu_i)^2), the hyperbolic recommenders' margin ranking loss
    (HGCF, HICF): squared distance is smooth where d -> 0, where d's own slope is infinite.
    m = gamma, or with `adaptive`, HICF's geometry-aware margin gamma * sigmoid(delta),
      delta = (d0(t)^2 + d0(h)^2 - d(t, h)^2) / (cosh d0(t) cosh d0(h)),
    d0 the distance from the origin: pairs near the origin (popular, general nodes) get larger
    margins, pairs out by the edge (the tail) smaller ones."""
    d_pos = space.distance(t, h)
    d_neg = space.distance(t[:, None, :], hu)
    m = gamma
    if adaptive:
        r_t, r_h = space.origin_distance(t), space.origin_distance(h)
        delta = (r_t ** 2 + r_h ** 2 - d_pos ** 2) / (torch.cosh(r_t) * torch.cosh(r_h))
        m = gamma * torch.sigmoid(delta)[:, None]
    return _mean_of_positive(torch.relu(m + d_pos[:, None] ** 2 - d_neg ** 2))


def starspace(t, h, hu, gamma, space=EUCLIDEAN):
    """max(0, gamma - (s(t, h) - s(h, hu_i))), s the space's similarity (cosine in Euclidean
    space, negative distance in the Poincaré ball)."""
    pos = space.similarity(t, h)
    neg = space.similarity(hu, h[:, None, :])
    return _mean_of_positive(torch.relu(gamma - (pos[:, None] - neg)))


def rank(t, h, hu, tau, space=EUCLIDEAN):
    """-log p(positive) under a softmax over [h.hu_1, ..., h.hu_n, h.t], when p < tau.  Dot
    products, so Euclidean space only."""
    if space is not EUCLIDEAN:
        raise ValueError("the rank loss scores by dot product, so it needs Euclidean space")
    logits = torch.cat([(hu * h[:, None, :]).sum(-1), (h * t).sum(-1, keepdim=True)], dim=1)
    log_p = torch.log_softmax(logits, dim=1)[:, -1]
    return torch.where(log_p.exp() < tau, -log_p, torch.zeros_like(log_p))


def rankspace(t, h, hu, gamma, space=EUCLIDEAN):
    return starspace(t, h, hu, gamma, space) + rank(t, h, hu, gamma, space)


def similarities(t, h, hu=None, pool=None, space=EUCLIDEAN):
    """(s(h, t) (B,), s(h, negatives) (B, K)), the negatives given per node (hu, (B, n, d))
    or as a pool shared by every node (pool, (P, d)), scored with one matrix multiply."""
    pos = space.similarity(h, t)
    if pool is not None:
        return pos, space.pairwise_similarity(h, pool)
    return pos, space.similarity(hu, h[:, None, :])


def infonce_from_similarities(pos, neg, temperature, exclude=None):
    """-log p(positive) under a softmax over [pos, neg_1, ..., neg_K] / temperature; negatives
    where `exclude` (B, K) is True don't count."""
    if exclude is not None:
        neg = neg.masked_fill(exclude, float("-inf"))
    return -torch.log_softmax(torch.cat([pos[:, None], neg], dim=1) / temperature, dim=1)[:, 0]


def starspace_from_similarities(pos, neg, gamma, exclude=None):
    """StarSpace's hinge, mean of the non-zero max(0, gamma - (pos - neg_i)), over negatives not
    in `exclude`."""
    terms = torch.relu(gamma - (pos[:, None] - neg))
    if exclude is not None:
        terms = terms.masked_fill(exclude, 0.0)
    return _mean_of_positive(terms)


def infonce(t, h, hu, temperature, space=EUCLIDEAN):
    """-log p(positive) under a softmax over similarities divided by the temperature:
    [s(h, t), s(h, hu_1), ..., s(h, hu_n)] / temperature.  Always on, unlike rank."""
    return infonce_from_similarities(*similarities(t, h, hu=hu, space=space), temperature)


def infonce_pool(t, h, pool, exclude, temperature, space=EUCLIDEAN):
    """infonce against every node of a shared negative pool at once: one (B, P) matrix multiply
    rather than gathering P negatives per node.  exclude: (B, P), True where the pool node is the
    anchor itself (or must otherwise not count)."""
    return infonce_from_similarities(*similarities(t, h, pool=pool, space=space), temperature, exclude)


LOSSES = {"margin": margin, "margin_sq": margin_sq, "starspace": starspace, "rank": rank, "rankspace": rankspace,
          "infonce": infonce}


# Listwise losses over each node's PPR list (ppr.py).  Unlike the losses above there's no
# reconstruction: h(v) is scored against its listed nodes' own embeddings, as retrieval scores
# them, with logits s(h(v), h(u)) / temperature (s the space's similarity).  pos_w (B, k) are the list's weights, 0 for
# padding; the negatives' logits (B, n) have MASKED where a negative must not count.

MASKED = -1e4   # finite, so logcumsumexp's backward pass stays finite


def logits(h, others, temperature, space=EUCLIDEAN):
    """(B, d) against (B, n, d) -> (B, n)."""
    return space.similarity(others, h[:, None, :]) / temperature


def pool_logits(h, pool, temperature, space=EUCLIDEAN):
    """(B, d) against a shared (P, d) pool -> (B, P)."""
    return space.pairwise_similarity(h, pool) / temperature


def listnet(pos_logits, pos_w, neg_logits, passive=False):
    """Cross entropy between the list's weights (0 for every negative) and one softmax over the
    positives and negatives.  passive skips positives already predicted at least as likely as
    their weight, as cloverleaf's Python bindings do."""
    valid = pos_w > 0
    k = pos_logits.shape[1]
    log_p = torch.log_softmax(torch.cat([pos_logits.masked_fill(~valid, MASKED), neg_logits], 1), 1)[:, :k]
    if passive:
        valid = valid & (log_p.detach().exp() <= pos_w)
    return -torch.where(valid, pos_w * log_p, torch.zeros_like(log_p)).sum(1)


def listmle(pos_logits, pos_w, neg_logits):
    """Top-k ListMLE: the Plackett-Luce likelihood of the positives coming first, in list order,
    each chosen from itself, the positives after it and every negative.  The negatives' own order
    is arbitrary, so it isn't scored."""
    valid = pos_w > 0
    pos = pos_logits.masked_fill(~valid, MASKED)
    after = torch.logcumsumexp(pos.flip(1), 1).flip(1)           # log sum over positives j >= i
    denom = torch.logaddexp(after, torch.logsumexp(neg_logits, 1, keepdim=True))
    return torch.where(valid, denom - pos, torch.zeros_like(pos)).sum(1)


def approxndcg(pos_logits, pos_w, neg_logits):
    """ApproxNDCG (Qin, Liu and Li, 2010): 1 - DCG / IDCG over the positives and negatives,
    with the list's weights as gains (2^w - 1; 0 for every negative) and each positive's rank the
    smooth estimate 1 + sum_{j != i} sigmoid(l_j - l_i).  The logits are already divided by the
    temperature (gamma), which sets how closely that tracks the true rank.  In [0, 1]; 0 for an
    empty list.  As cloverleaf's PprRank "approxndcg"."""
    valid = pos_w > 0
    pos = pos_logits.masked_fill(~valid, MASKED)
    every = torch.cat([pos, neg_logits], 1)                          # (B, k + n)
    # sigmoid(l_j - l_i) summed over every j; j = i adds sigmoid(0) = 1/2.
    rank = torch.sigmoid(every[:, None, :] - pos[:, :, None]).sum(-1) + 0.5
    gain = torch.where(valid, torch.exp2(pos_w) - 1, torch.zeros_like(pos_w))
    dcg = torch.where(valid, gain / torch.log2(1 + rank), torch.zeros_like(gain)).sum(1)
    ideal = gain.sort(1, descending=True).values
    discount = torch.log2(torch.arange(ideal.shape[1], device=ideal.device, dtype=ideal.dtype) + 2)
    idcg = (ideal / discount).sum(1)
    return torch.where(idcg > 0, 1 - dcg / idcg.clamp(min=1e-12), torch.zeros_like(idcg))


LISTWISE = {"listnet": listnet, "listmle": listmle, "approxndcg": approxndcg}
