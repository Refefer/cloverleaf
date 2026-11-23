"""Batched, on-device sampling of features, neighbours and negatives.

Every function takes a 1-D tensor of node ids and returns padded tensors plus a mask (or token
weights, 0 for padding), so a whole batch is handled with a few tensor operations.
"""
import torch


def _ranges(ptr, nodes):
    starts = ptr[nodes]
    return starts, ptr[nodes + 1] - starts


def sample_features(graph, nodes, max_features=None, generator=None):
    """Each node's features, or a random subset of them.

    max_features: None keeps every feature; a float in (0, 1) keeps ceil(fraction * count); an
    int keeps at most that many.  Every node keeps at least one feature.
    Returns (feature ids, mask), both (len(nodes), L).
    """
    starts, counts = _ranges(graph.feat_ptr, nodes)
    if max_features is None:
        keep = counts
    elif isinstance(max_features, float):
        keep = torch.ceil(counts.float() * max_features).long()
    else:
        keep = torch.clamp(counts, max=int(max_features))
    keep = torch.clamp(keep, min=1)
    width = int(counts.max()) if len(nodes) else 0
    pos = torch.arange(width, device=nodes.device).expand(len(nodes), width)
    valid = pos < counts[:, None]
    if max_features is not None:
        # A random subset: rank the valid positions by random keys and keep the first `keep`.
        keys = torch.rand(len(nodes), width, device=nodes.device, generator=generator)
        keys = keys.masked_fill(~valid, 2.0)
        pos = torch.argsort(keys, dim=1)
    width_kept = int(keep.max()) if len(nodes) else 0
    pos = pos[:, :width_kept]
    mask = torch.arange(width_kept, device=nodes.device)[None, :] < keep[:, None]
    idx = graph.feat_idx[(starts[:, None] + pos).masked_fill(~mask, 0).clamp(max=len(graph.feat_idx) - 1)]
    return idx.masked_fill(~mask, 0), mask


def sample_neighbours(graph, nodes, max_nodes=None, weighted=False, generator=None):
    """Each node's neighbours, or a sample of them.

    max_nodes: None keeps every neighbour; an int keeps every neighbour of nodes with at most
    that many and otherwise samples that many, with replacement (cloverleaf samples without
    replacement; the difference is small when degree >> max_nodes).  weighted samples in
    proportion to edge weight.
    Returns (neighbour ids, edge weights, mask), each (len(nodes), K), and whether each node's
    neighbours were sampled rather than all taken.
    """
    starts, degree = _ranges(graph.adj_ptr, nodes)
    device = nodes.device
    if max_nodes is None:
        width = int(degree.max()) if len(nodes) else 0
        offsets = torch.arange(width, device=device).expand(len(nodes), width)
        mask = offsets < degree[:, None]
        sampled_nodes = torch.zeros(len(nodes), dtype=torch.bool, device=device)
    else:
        width = int(max_nodes)
        take_all = degree <= width
        all_offsets = torch.arange(width, device=device).expand(len(nodes), width)
        u = torch.rand(len(nodes), width, device=device, generator=generator)
        if weighted:
            cum = torch.cumsum(graph.adj_w, 0)
            before = torch.where(starts > 0, cum[(starts - 1).clamp(min=0)], torch.zeros_like(starts, dtype=cum.dtype))
            total = cum[(starts + degree - 1).clamp(min=0)] - before
            target = before[:, None] + u * total[:, None]
            sampled = torch.searchsorted(cum, target.contiguous(), right=True) - starts[:, None]
            sampled = torch.minimum(sampled, (degree - 1).clamp(min=0)[:, None])
        else:
            sampled = (u * degree[:, None]).long()
        offsets = torch.where(take_all[:, None], all_offsets, sampled)
        mask = torch.where(take_all[:, None], all_offsets < degree[:, None],
                           torch.ones_like(all_offsets, dtype=torch.bool))
        mask &= degree[:, None] > 0
        sampled_nodes = ~take_all
    pos = (starts[:, None] + offsets).masked_fill(~mask, 0).clamp(max=max(len(graph.adj_idx) - 1, 0))
    nbrs = graph.adj_idx[pos].masked_fill(~mask, 0)
    weights = graph.adj_w[pos].masked_fill(~mask, 0.0)
    return nbrs, weights, mask, sampled_nodes


def sample_negatives(num_nodes, nodes, count, generator=None):
    """`count` uniformly random nodes per anchor, never the anchor itself."""
    shift = torch.randint(1, max(num_nodes, 2), (len(nodes), count), device=nodes.device,
                          generator=generator)
    return (nodes[:, None] + shift) % num_nodes


def pick_from_pool(pool, nodes, count, generator=None):
    """For each anchor, `count` positions in `pool` (a shared set of negative nodes) whose node
    isn't the anchor.  Returns (len(nodes), count) indices into pool."""
    pick = torch.randint(0, len(pool), (len(nodes), count), device=nodes.device, generator=generator)
    for _ in range(8):
        clash = pool[pick] == nodes[:, None]
        if not clash.any():
            break
        pick = torch.where(clash, torch.randint(0, len(pool), pick.shape, device=nodes.device,
                                                generator=generator), pick)
    return pick


def neighbour_features(graph, nodes, max_nodes=None, max_features=None, weighted_sampling=False,
                       weighted_averaging=False, generator=None):
    """Each sampled neighbour's own features, one row per neighbour, and each neighbour's weight
    (0 for padding): its edge weight with weighted_averaging, 1 otherwise.  As in cloverleaf,
    edge weights aren't applied again to neighbours already sampled in proportion to them.
    Returns (feature ids, feature mask), both (len(nodes) * K, L), and weights (len(nodes), K).
    """
    nbrs, edge_w, nmask, sampled = sample_neighbours(graph, nodes, max_nodes, weighted_sampling,
                                                     generator)
    fidx, fmask = sample_features(graph, nbrs.reshape(-1), max_features, generator)
    fmask = fmask & nmask.reshape(-1, 1)
    w = nmask.float()
    if weighted_averaging:
        use_edge_w = ~sampled if weighted_sampling else torch.ones_like(sampled)
        w = torch.where(use_edge_w[:, None], edge_w, w)
    return fidx.masked_fill(~fmask, 0), fmask, w * nmask


def neighbourhood_tokens(graph, nodes, max_nodes=None, max_features=None, weighted_sampling=False,
                         weighted_averaging=False, generator=None):
    """The features of a sample of each node's neighbours, flattened into one token set per node,
    with a weight per token (0 for padding): the neighbour's weight, its edge weight when
    weighted_averaging and 1 otherwise.  This is what h~(v) pools over, as in cloverleaf, where a
    feature shared by several neighbours counts once per neighbour.
    Returns (feature ids, token weights), both (len(nodes), K * L).
    """
    fidx, fmask, w = neighbour_features(graph, nodes, max_nodes, max_features, weighted_sampling,
                                        weighted_averaging, generator)
    b, k = w.shape
    l = fidx.shape[1]
    tok_w = (w.reshape(-1, 1) * fmask).reshape(b, k * l)
    return fidx.reshape(b, k * l), tok_w
