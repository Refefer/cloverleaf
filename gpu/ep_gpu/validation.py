"""A retrieval check on the validation nodes, alongside their loss.

The validation loss measures the training objective, which has repeatedly ranked models
differently from retrieval.  This measures retrieval directly: for each validation node, do its
real neighbours outrank random nodes?  Each neighbour is compared with `negatives` random nodes
of its own type (a query's products against random products), every node embedded from all its
features as the exported embeddings are.

  hr@10  the fraction of (node, neighbour) pairs where fewer than 10 negatives score at least as
         high as the neighbour (ties count against it), as the retrieval evaluation's HitRate@10
  auc    the mean fraction of negatives the neighbour outscores

The pairs and negatives are drawn once, so the numbers compare from pass to pass.

The pairs come from one of two places:

  held-out edges  (`hold_out_edges`) edges removed from the training graph, in both directions:
                  link prediction on edges the model never saw, the question the retrieval
                  evaluation asks of test-period edges.
  validation nodes (`RankingProbe.from_nodes`) the neighbours of nodes left out as anchors.
                  Those nodes stay in their neighbours' reconstructions, so the edges were
                  trained on, and the check mostly measures how well training edges are fit.
"""
import dataclasses

import numpy as np
import torch
import torch.nn.functional as F

from . import sampling


def hold_out_edges(graph, fraction, generator=None):
    """(graph without them, sources, targets): `fraction` of the edges, drawn uniformly and
    removed together with their reverse edges where those exist (an undirected graph stores
    both), so nothing in training connects the pair."""
    n, degrees = graph.num_nodes, graph.degrees()
    device = graph.adj_ptr.device
    src = torch.repeat_interleave(torch.arange(n, device=device), degrees)
    dst = graph.adj_idx.long()
    count = int(len(dst) * fraction)
    pick = torch.randperm(len(dst), device=device, generator=generator)[:count]
    keys = src * n + dst
    dropped = torch.cat([keys[pick], dst[pick] * n + src[pick]])
    keep = ~torch.isin(keys, dropped)
    ptr = torch.zeros(n + 1, dtype=graph.adj_ptr.dtype, device=device)
    ptr[1:] = torch.cumsum(torch.bincount(src[keep], minlength=n), 0)
    held = dataclasses.replace(graph, adj_ptr=ptr, adj_idx=graph.adj_idx[keep], adj_w=graph.adj_w[keep])
    return held, src[pick], dst[pick]


class RankingProbe:
    def __init__(self, graph, anchor, positive, negatives=1000, seed=0):
        """graph: on the device validation runs on; (anchor, positive): the pairs to score."""
        device = graph.adj_ptr.device
        gen = torch.Generator(device=device).manual_seed(seed)
        self.anchor, self.positive = anchor, positive
        _, types = np.unique(np.asarray(graph.node_types), return_inverse=True)
        node_type = torch.from_numpy(types).to(device)
        self.positive_type = node_type[self.positive]
        # `negatives` random nodes of each neighbour type: (types, negatives).
        self.negatives = {}
        for t in self.positive_type.unique().tolist():
            members = (node_type == t).nonzero(as_tuple=True)[0]
            pick = torch.randint(0, len(members), (negatives,), device=device, generator=gen)
            self.negatives[t] = members[pick]
        self.count = negatives

    @classmethod
    def from_nodes(cls, graph, valid, positives=5, negatives=1000, seed=0):
        """Up to `positives` neighbours of each validation node."""
        gen = torch.Generator(device=graph.adj_ptr.device).manual_seed(seed)
        nbrs, _, mask, _ = sampling.sample_neighbours(graph, valid, positives, generator=gen)
        rows, cols = mask.nonzero(as_tuple=True)
        return cls(graph, valid[rows], nbrs[rows, cols], negatives, seed + 1)

    def __len__(self):
        return len(self.anchor)

    @torch.no_grad()
    def evaluate(self, model, graph, batch_size=4096):
        """(hr@10, auc) over every (node, neighbour) pair."""
        if not len(self.anchor):
            return float("nan"), float("nan")
        model.eval()

        def embed(nodes):
            out = []
            for s in range(0, len(nodes), batch_size):
                chunk = nodes[s:s + batch_size]
                fidx, fmask = sampling.sample_features(graph, chunk, None)
                out.append(model.embed(fidx.to(model.device), fmask.float().to(model.device)).float())
            return F.normalize(torch.cat(out), dim=1)

        negatives = {t: embed(n) for t, n in self.negatives.items()}
        hits = above = 0.0
        for s in range(0, len(self.anchor), batch_size):
            a = embed(self.anchor[s:s + batch_size])
            p = embed(self.positive[s:s + batch_size])
            types = self.positive_type[s:s + batch_size].to(model.device)
            score = (a * p).sum(1)
            beaten = torch.zeros(len(a), device=model.device)
            for t, neg in negatives.items():
                rows = types == t
                if rows.any():
                    beaten[rows] = (a[rows] @ neg.T >= score[rows, None]).sum(1).float()
            hits += float((beaten < 10).sum())
            above += float((self.count - beaten).sum() / self.count)
        return hits / len(self.anchor), above / len(self.anchor)
