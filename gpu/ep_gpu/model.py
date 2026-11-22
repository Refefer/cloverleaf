"""The feature embedding table and the pooling that turns a set of features into one embedding.

h(v) pools v's own features.  h~(v) pools its neighbours' features, either flat, all the sampled
neighbours' features as one token set (cloverleaf's reconstruction, sampling.neighbourhood_tokens),
or nested, each neighbour pooled from its own features and the neighbours then pooled together
(sampling.neighbour_features, EPModel.combine_neighbours).  Nested keeps attention's token sets
to one node's features, which flat attention over every neighbour's features can't fit in memory
at useful batch sizes.

  mean       the weighted mean of the feature embeddings: cloverleaf's averaged model, EP-B.
  attention  multi-head self-attention over the tokens, then their weighted mean.  Unlike
             cloverleaf's attention, which stores per-head query/key slices in every feature's
             embedding, the query, key, value and output projections are shared across features,
             so the table stays one `dims`-wide vector per feature.  The output projection starts
             at zero, so training starts from mean pooling.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


def weighted_mean(x, w):
    """x: (B, T, d); w: (B, T), 0 for padding.  Rows without tokens give zeros."""
    total = w.sum(dim=1, keepdim=True)
    return (x * w[..., None]).sum(dim=1) / total.clamp(min=1e-12)


class MeanPool(nn.Module):
    def forward(self, x, w):
        return weighted_mean(x, w)


class AttentionPool(nn.Module):
    def __init__(self, dims, heads):
        super().__init__()
        self.heads = heads
        # Each head's width is dims / heads rounded up to a multiple of 8: the fused attention
        # kernels need that, and otherwise attention falls back to PyTorch's slow math path.
        self.head_dims = -(-dims // heads // 8) * 8 if dims >= heads else 8
        inner = heads * self.head_dims
        self.qkv = nn.Linear(dims, 3 * inner, bias=False)
        self.out = nn.Linear(inner, dims, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x, w):
        return self.attend(self.qkv(x), weighted_mean(x, w), w)

    def attend(self, qkv, x_mean, w):
        """Attention from already-projected tokens.  qkv: (B, T, 3 * heads * head_dims), the
        query, key and value projections of the tokens; x_mean: (B, d), the tokens' weighted
        mean.  The output projection is linear, so it's applied once to each row's weighted
        mean context rather than to every token: weighted_mean(x + out(ctx)) equals
        weighted_mean(x) + out(weighted_mean(ctx))."""
        b, t, _ = qkv.shape
        mask = w > 0
        # A row without tokens would softmax over nothing; give it one masked-in pad token.
        # Its weight is still 0, so the row pools to zeros.
        empty = ~mask.any(dim=1)
        attend = mask.clone()
        attend[empty, 0] = True
        q, k, v = qkv.view(b, t, 3, self.heads, self.head_dims).permute(2, 0, 3, 1, 4)
        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attend[:, None, None, :])
        ctx = ctx.transpose(1, 2).reshape(b, t, self.heads * self.head_dims)
        return x_mean + self.out(weighted_mean(ctx, w))


class EPModel(nn.Module):
    def __init__(self, num_features, dims, pooling="mean", heads=4, init="unit", seed=None):
        super().__init__()
        self.features = nn.Embedding(num_features, dims, sparse=True)
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        with torch.no_grad():
            if init == "unit":
                # Uniform in [-1, 1), then normalised to unit length, as cloverleaf's "unit".
                weight = torch.rand(num_features, dims, generator=g) * 2 - 1
                self.features.weight.copy_(F.normalize(weight, dim=1))
            elif init == "glorot":
                limit = math.sqrt(6 / (num_features + dims))
                self.features.weight.copy_((torch.rand(num_features, dims, generator=g) * 2 - 1) * limit)
            else:
                raise ValueError(f"init must be 'unit' or 'glorot', got {init!r}")
        if pooling == "mean":
            self.pool, self.neighbour_pool = MeanPool(), MeanPool()
        elif pooling == "attention":
            # Separate parameters for pooling features and for pooling neighbours.
            self.pool, self.neighbour_pool = AttentionPool(dims, heads), AttentionPool(dims, heads)
        else:
            raise ValueError(f"pooling must be 'mean' or 'attention', got {pooling!r}")

    def embed(self, feature_ids, weights):
        """feature_ids, weights: (B, T); weights are 0 for padding."""
        return self.embed_groups([(feature_ids, weights)])[0]

    def embed_groups(self, groups):
        """Embeds several (feature ids, weights) groups, e.g. anchors, reconstructions and
        negatives, looking each distinct feature up once across all of them.

        A batch repeats features heavily (shared words, neighbours shared between anchors, a
        neighbour's features counted for every anchor it neighbours), so looking up unique
        features keeps the gathered rows, and the sparse gradient the optimiser has to merge,
        to the batch's distinct features rather than one row per token.
        """
        valids = [w > 0 for _, w in groups]
        ids = torch.cat([f[v] for (f, _), v in zip(groups, valids)])
        unique, inverse = torch.unique(ids, return_inverse=True)
        table = self.features(unique)
        attention = isinstance(self.pool, AttentionPool)
        # A token's query, key and value depend only on its feature, so they're projected once
        # per distinct feature rather than once per (padded) token.
        projected = self.pool.qkv(table) if attention else None
        out, start = [], 0
        for (f, w), v in zip(groups, valids):
            rows = v.nonzero(as_tuple=True)[0]
            count = len(rows)
            index = inverse[start:start + count]
            start += count
            # The weighted mean of the tokens: a scatter-sum, without padding.
            summed = torch.zeros(f.shape[0], table.shape[1], device=table.device, dtype=table.dtype)
            summed.index_add_(0, rows, table[index] * w[v][:, None])
            mean = summed / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
            if attention:
                padded = torch.zeros(*f.shape, projected.shape[1], device=table.device, dtype=projected.dtype)
                padded[v] = projected[index]
                out.append(self.pool.attend(padded, mean, w))
            else:
                out.append(mean)
        return out

    def combine_neighbours(self, neighbour_embeddings, weights):
        """Nested reconstruction's second level: (B, K, d) neighbour embeddings and (B, K)
        weights, 0 for padding, pooled into one (B, d) embedding per node."""
        return self.neighbour_pool(neighbour_embeddings, weights)

    def dense_parameters(self):
        return [p for name, p in self.named_parameters() if not name.startswith("features.")]
