"""EP's losses, batched, with cloverleaf's semantics (src/algos/ep/loss.rs).

Each takes t = h~(v) (B, d), h = h(v) (B, d) and hu = h(u) for the negatives (B, n, d), and returns
one loss per node (B,).  Margin and StarSpace average only their non-zero per-negative terms, and
are 0 when all are zero, as cloverleaf does.
"""
import torch
import torch.nn.functional as F


def _mean_of_positive(terms):
    positive = terms > 0
    count = positive.sum(dim=1)
    return torch.where(count > 0, (terms * positive).sum(dim=1) / count.clamp(min=1),
                       torch.zeros_like(count, dtype=terms.dtype))


def margin(t, h, hu, gamma):
    """max(0, gamma + |t - h| - |t - hu_i|), Euclidean."""
    d_pos = torch.linalg.vector_norm(t - h, dim=-1)
    d_neg = torch.linalg.vector_norm(t[:, None, :] - hu, dim=-1)
    return _mean_of_positive(torch.relu(gamma + d_pos[:, None] - d_neg))


def starspace(t, h, hu, gamma):
    """max(0, gamma - (cos(t, h) - cos(h, hu_i)))."""
    tn, hn, hun = F.normalize(t, dim=-1), F.normalize(h, dim=-1), F.normalize(hu, dim=-1)
    pos = (tn * hn).sum(-1)
    neg = (hun * hn[:, None, :]).sum(-1)
    return _mean_of_positive(torch.relu(gamma - (pos[:, None] - neg)))


def rank(t, h, hu, tau):
    """-log p(positive) under a softmax over [h.hu_1, ..., h.hu_n, h.t], when p < tau."""
    logits = torch.cat([(hu * h[:, None, :]).sum(-1), (h * t).sum(-1, keepdim=True)], dim=1)
    log_p = torch.log_softmax(logits, dim=1)[:, -1]
    return torch.where(log_p.exp() < tau, -log_p, torch.zeros_like(log_p))


def rankspace(t, h, hu, gamma):
    return starspace(t, h, hu, gamma) + rank(t, h, hu, gamma)


def cosines(t, h, hu=None, pool=None):
    """(cos(h, t) (B,), cos(h, negatives) (B, K)), the negatives given per node (hu, (B, n, d))
    or as a pool shared by every node (pool, (P, d)), scored with one matrix multiply."""
    hn = F.normalize(h, dim=-1)
    pos = (hn * F.normalize(t, dim=-1)).sum(-1)
    if pool is not None:
        return pos, hn @ F.normalize(pool, dim=-1).T
    return pos, (F.normalize(hu, dim=-1) * hn[:, None, :]).sum(-1)


def infonce_from_cosines(pos, neg, temperature, exclude=None):
    """-log p(positive) under a softmax over [pos, neg_1, ..., neg_K] / temperature; negatives
    where `exclude` (B, K) is True don't count."""
    if exclude is not None:
        neg = neg.masked_fill(exclude, float("-inf"))
    return -torch.log_softmax(torch.cat([pos[:, None], neg], dim=1) / temperature, dim=1)[:, 0]


def starspace_from_cosines(pos, neg, gamma, exclude=None):
    """StarSpace's hinge, mean of the non-zero max(0, gamma - (pos - neg_i)), over negatives not
    in `exclude`."""
    terms = torch.relu(gamma - (pos[:, None] - neg))
    if exclude is not None:
        terms = terms.masked_fill(exclude, 0.0)
    return _mean_of_positive(terms)


def infonce(t, h, hu, temperature):
    """-log p(positive) under a softmax over cosine similarities divided by the temperature:
    [cos(h, t), cos(h, hu_1), ..., cos(h, hu_n)] / temperature.  Always on, unlike rank."""
    return infonce_from_cosines(*cosines(t, h, hu=hu), temperature)


def infonce_pool(t, h, pool, exclude, temperature):
    """infonce against every node of a shared negative pool at once: one (B, P) matrix multiply
    rather than gathering P negatives per node.  exclude: (B, P), True where the pool node is the
    anchor itself (or must otherwise not count)."""
    return infonce_from_cosines(*cosines(t, h, pool=pool), temperature, exclude)


LOSSES = {"margin": margin, "starspace": starspace, "rank": rank, "rankspace": rankspace,
          "infonce": infonce}


# Listwise losses over each node's PPR list (ppr.py).  Unlike the losses above there's no
# reconstruction: h(v) is scored against its listed nodes' own embeddings, as retrieval scores
# them, with logits cos(h(v), h(u)) / temperature.  pos_w (B, k) are the list's weights, 0 for
# padding; the negatives' logits (B, n) have MASKED where a negative must not count.

MASKED = -1e4   # finite, so logcumsumexp's backward pass stays finite


def cosine_logits(h, others, temperature):
    """(B, d) against (B, n, d) -> (B, n)."""
    return (F.normalize(others, dim=-1) * F.normalize(h, dim=-1)[:, None, :]).sum(-1) / temperature


def pool_logits(h, pool, temperature):
    """(B, d) against a shared (P, d) pool -> (B, P)."""
    return F.normalize(h, dim=-1) @ F.normalize(pool, dim=-1).T / temperature


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


LISTWISE = {"listnet": listnet, "listmle": listmle}
