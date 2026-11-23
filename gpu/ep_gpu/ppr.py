"""Personalised PageRank lists for the listwise losses, built by cloverleaf (cloverleaf.ppr_lists,
the lists its PprRankLearner trains on).

Random walks with restart from each node, following edges in proportion to their weight, score
each node reached count / (walks * degree^beta), so beta > 0 discounts popular nodes.  A node's
list is its top k other nodes, weighted score^compression and normalised to sum to 1.
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PprLists:
    nodes: torch.Tensor     # (N, k) node ids; padding is 0
    weights: torch.Tensor   # (N, k) each row sums to 1, or is all 0 for a node reaching nothing;
                            # padding is 0.  Rows are in descending weight.

    def to(self, device):
        return PprLists(self.nodes.to(device), self.weights.to(device))


def ppr_lists(edges_path, graph, k, walks, steps, beta=0.8, compression=1.0,
              other_type_only=False, directed=False, seed=2017):
    """The top-k PPR list of every node of `graph`, which was loaded from edges_path.
    other_type_only keeps only nodes of a different type from the start node, e.g. a query's
    products.  Uses the cloverleaf graph `graph` was loaded from, or reloads it (a cached graph)."""
    import cloverleaf

    cl_graph = graph.cloverleaf
    if cl_graph is None:
        edge_type = cloverleaf.EdgeType.Directed if directed else cloverleaf.EdgeType.Undirected
        cl_graph = cloverleaf.Graph.load(edges_path, edge_type, chunk_size=100_000)
    if cl_graph.nodes() != graph.num_nodes:
        raise ValueError(f"cloverleaf loaded {cl_graph.nodes()} nodes, but the graph has {graph.num_nodes}")
    nodes, weights = cloverleaf.ppr_lists(cl_graph, k, walks, float(steps), beta=beta,
                                          compression=compression, other_type_only=other_type_only,
                                          seed=seed, indicator=False)
    nodes = torch.from_numpy(np.frombuffer(nodes, np.int64).reshape(-1, k).copy())
    weights = torch.from_numpy(np.frombuffer(weights, np.float32).reshape(-1, k).copy())
    return PprLists(nodes.to(graph.adj_idx.dtype), weights)


SAMPLE_BY = ("weight", "uniform")


def subsample(nodes, weights, count, by="weight", generator=None):
    """`count` entries of each list, without replacement: in proportion to their weight (by the
    Gumbel top-k trick) or uniformly.  The drawn entries' weights are renormalised to sum to 1 and
    each row is sorted by descending weight, as the listwise losses expect; rows with fewer than
    `count` entries keep them all, padded with weight 0."""
    if by not in SAMPLE_BY:
        raise ValueError(f"by must be one of {', '.join(SAMPLE_BY)}, got {by!r}")
    valid = weights > 0
    u = torch.rand(weights.shape, device=weights.device, generator=generator).clamp(min=1e-20)
    gumbel = -torch.log(-torch.log(u))
    keys = gumbel + (torch.log(weights.clamp(min=1e-30)) if by == "weight" else 0)
    keys = keys.masked_fill(~valid, float("-inf"))
    pick = keys.topk(min(count, weights.shape[1]), dim=1).indices
    drawn_w = torch.gather(weights, 1, pick)
    drawn_w = drawn_w / drawn_w.sum(1, keepdim=True).clamp(min=1e-30)
    order = torch.argsort(drawn_w, dim=1, descending=True, stable=True)
    return torch.gather(torch.gather(nodes, 1, pick), 1, order), torch.gather(drawn_w, 1, order)
