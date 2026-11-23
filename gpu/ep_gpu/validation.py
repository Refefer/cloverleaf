"""A retrieval check on the validation nodes, alongside their loss.

The validation loss measures the training objective, which has repeatedly ranked models
differently from retrieval.  This measures retrieval directly: for each validation node, do its
real neighbours outrank random nodes?  Each neighbour is compared with `negatives` random nodes
of its own type (a query's products against random products), every node embedded from all its
features as the exported embeddings are.

  hr@10  the fraction of (node, neighbour) pairs where fewer than 10 negatives score at least as
         high as the neighbour (ties count against it), as the retrieval evaluation's HitRate@10
  auc    the mean fraction of negatives the neighbour outscores

The neighbours and negatives are drawn once, so the numbers compare from pass to pass.
Validation nodes are left out as anchors but stay in the graph, so their edges still shape
their neighbours' training: this is a transductive check, like the evaluation's transductive
queries, not an inductive one.
"""
import numpy as np
import torch
import torch.nn.functional as F

from . import sampling


class RankingProbe:
    def __init__(self, graph, valid, positives=5, negatives=1000, seed=0):
        """graph: on the device validation runs on; valid: validation node ids."""
        device = graph.adj_ptr.device
        gen = torch.Generator(device=device).manual_seed(seed)
        nbrs, _, mask, _ = sampling.sample_neighbours(graph, valid, positives, generator=gen)
        rows, cols = mask.nonzero(as_tuple=True)
        self.anchor = valid[rows]                   # (P,) each pair's validation node
        self.positive = nbrs[rows, cols]            # (P,) and its neighbour
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
