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


LOSSES = {"margin": margin, "starspace": starspace, "rank": rank, "rankspace": rankspace}
