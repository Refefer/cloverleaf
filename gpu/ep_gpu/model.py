"""The feature embedding table and the pooling that turns a set of features into one embedding.

h(v) pools v's own features.  h~(v) pools its neighbours' features, either flat, all the sampled
neighbours' features as one token set (cloverleaf's reconstruction, sampling.neighbourhood_tokens),
or nested, each neighbour pooled from its own features and the neighbours then pooled together
(sampling.neighbour_features, EPModel.combine_neighbours).  Nested keeps attention's token sets
to one node's features, which flat attention over every neighbour's features can't fit in memory
at useful batch sizes.

  mean       the weighted mean of the feature embeddings: cloverleaf's averaged model, EP-B.
  attention  multi-head self-attention over the tokens, then their weighted mean: cloverleaf's
             attention model (src/algos/ep/attention.rs).  The query, key, value and output
             projections are shared across features, so the table stays one `dims`-wide vector
             per feature.  The output projection starts at zero, so training starts from mean
             pooling.

The pooled vector becomes the node's embedding through the model's space (geometry.py): as is in
Euclidean space, or mapped into the Poincaré ball.  embed_groups returns pooled vectors, so nested
reconstructions combine neighbours before the map; embed returns embeddings in the space.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from .geometry import EUCLIDEAN


def weighted_mean(x, w, space=EUCLIDEAN):
    """x: (B, T, d); w: (B, T), 0 for padding.  Rows without tokens give zeros.  The space's mean:
    arithmetic, or in the Poincaré ball the gyromidpoint (geometry.py)."""
    if not space.pools_in_space:
        return (x * w[..., None]).sum(dim=1) / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
    num, den = space.mean_terms(x.float())
    return space.mean_finish((num * w[..., None]).sum(dim=1), (den * w[..., None]).sum(dim=1))


class MeanPool(nn.Module):
    def __init__(self, space=EUCLIDEAN):
        super().__init__()
        self.space = space

    def forward(self, x, w):
        return weighted_mean(x, w, self.space)


ATTENTION_MODES = ("transform", "weights")


class AttentionPool(nn.Module):
    def __init__(self, dims, heads, keeps_norm=False, dropout=0.0, mode="transform"):
        """keeps_norm: attention changes only the pooled vector's direction; its length stays the
        mean's.  In the Poincaré ball the length is the distance from the origin, which encodes how
        general a node is; left free, the shared output projection learns to stretch every node
        outwards (distances grow, satisfying margins cheaply), so that distance stops following
        the node's own features.

        dropout: in training, dropout on attention's added term out(ctx) (residual dropout), so
        the projection every node shares can't co-adapt into one consistent shift that fits the
        training nodes; at inference the term is used whole.

        mode: "transform" adds out(ctx), the output projection of the tokens' attention context,
        to their mean (cloverleaf's attention).  Since out and the value projection are linear,
        that term is M (sum_j a_j x_j) with one matrix M = out V shared by every node, which can
        stretch and rotate any node.  "weights" keeps only the per-node importance: the node is
        sum_j b_j x_j, b_j proportional to w_j times the attention token j receives within the
        node, a convex combination of its own feature embeddings.  There's no value or output
        projection, and the key projection starts at zero, so it starts as the mean."""
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError(f"attention dropout must be in [0, 1), got {dropout}")
        if mode not in ATTENTION_MODES:
            raise ValueError(f"attention mode must be one of {', '.join(ATTENTION_MODES)}, got {mode!r}")
        if mode == "weights" and (keeps_norm or dropout):
            raise ValueError("keeps_norm and dropout apply to attention mode 'transform' only")
        self.mode = mode
        self.heads = heads
        self.keeps_norm = keeps_norm
        self.dropout = dropout
        # Each head's width is dims / heads rounded up to a multiple of 8: the fused attention
        # kernels need that, and otherwise attention falls back to PyTorch's slow math path.
        self.head_dims = -(-dims // heads // 8) * 8 if dims >= heads else 8
        inner = heads * self.head_dims
        if mode == "weights":
            # Queries and keys only.  The keys start at zero, so every score is equal and the
            # weights start uniform (mean pooling); the queries don't, or neither would ever get
            # a gradient (d score / d key is the query, and the reverse).
            self.qkv = nn.Linear(dims, 2 * inner, bias=False)
            with torch.no_grad():
                self.qkv.weight[inner:] = 0
        else:
            self.qkv = nn.Linear(dims, 3 * inner, bias=False)
            self.out = nn.Linear(inner, dims, bias=False)
            nn.init.zeros_(self.out.weight)

    def forward(self, x, w):
        return self.attend(self.qkv(x), weighted_mean(x, w), w, x)

    def importance(self, qk, w):
        """(B, T) weights summing to 1 over each row's tokens: w_j times the mean attention token j
        receives from the row's tokens (weighted by w, over heads).  Rows without tokens get 0."""
        b, t, _ = qk.shape
        mask = w > 0
        q, k = qk.view(b, t, 2, self.heads, self.head_dims).permute(2, 0, 3, 1, 4)    # (B, H, T, hd)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dims)
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)                         # (B, H, T, T)
        received = (probs * w[:, None, :, None]).sum(2).mean(1) / w.sum(1, keepdim=True).clamp(min=1e-12)
        b_ = received * w
        return b_ / b_.sum(1, keepdim=True).clamp(min=1e-12)

    def attend(self, qkv, x_mean, w, x=None):
        """Attention from already-projected tokens.  qkv: (B, T, 3 * heads * head_dims), the
        query, key and value projections of the tokens; x_mean: (B, d), the tokens' weighted
        mean.  The output projection is linear, so it's applied once to each row's weighted
        mean context rather than to every token: weighted_mean(x + out(ctx)) equals
        weighted_mean(x) + out(weighted_mean(ctx)).  In mode "weights", qkv holds only queries and
        keys, and x (B, T, d) the tokens themselves, which the node is a weighted sum of."""
        if self.mode == "weights":
            return (self.importance(qkv, w)[..., None] * x).sum(1)
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
        pooled = x_mean + F.dropout(self.out(weighted_mean(ctx, w)), self.dropout, self.training)
        if self.keeps_norm:
            length = torch.sqrt((x_mean * x_mean).sum(-1, keepdim=True) + 1e-30)
            pooled = pooled * (length / torch.sqrt((pooled * pooled).sum(-1, keepdim=True) + 1e-30))
        return pooled


class EPModel(nn.Module):
    def __init__(self, num_features, dims, pooling="mean", heads=4, init="unit", seed=None,
                 table_dtype=torch.float32, neighbour_pooling="mean", space=EUCLIDEAN,
                 attention_keeps_norm=False, attention_dropout=0.0, attention_mode="transform",
                 feature_importance=False):
        super().__init__()
        if space.pools_in_space and "attention" in (pooling, neighbour_pooling):
            raise ValueError("pooling in the Poincaré ball (midpoint) needs mean pooling and mean neighbour pooling")
        if space.neighbours_in_space and neighbour_pooling != "mean":
            raise ValueError("combining neighbours by their midpoint needs neighbour_pooling 'mean'")
        self.space = space
        # One learned scalar per feature: mean pooling weights feature j by w_j exp(s_j).  Zero,
        # so it starts as the plain mean.
        self.importance = None
        if feature_importance:
            if pooling != "mean" or neighbour_pooling != "mean" or space.pools_in_space:
                raise ValueError("feature_importance needs mean pooling, mean neighbour pooling and "
                                 "pooling outside the ball")
            self.importance = nn.Embedding(num_features, 1, sparse=True)
            nn.init.zeros_(self.importance.weight)
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
        # Initialised in float32, then stored as table_dtype (bfloat16 halves the table).
        self.features.to(table_dtype)
        if pooling == "mean":
            self.pool = MeanPool(space)
        elif pooling == "attention":
            self.pool = AttentionPool(dims, heads, attention_keeps_norm, attention_dropout, attention_mode)
        else:
            raise ValueError(f"pooling must be 'mean' or 'attention', got {pooling!r}")
        # How a nested reconstruction combines its neighbours.  A weighted mean keeps h~(v) in the
        # same space as the node embeddings it's compared with, so training optimises node-to-node
        # similarity directly; attention here learns its own transform of that space.
        if neighbour_pooling == "mean":
            self.neighbour_pool = MeanPool(space)
        elif neighbour_pooling == "attention":
            self.neighbour_pool = AttentionPool(dims, heads, attention_keeps_norm, attention_dropout, attention_mode)
        else:
            raise ValueError(f"neighbour_pooling must be 'mean' or 'attention', got {neighbour_pooling!r}")

    @property
    def device(self):
        """Where embeddings are computed: set by train.build, else wherever the table is."""
        return self.__dict__.get("_compute_device") or self.features.weight.device

    @device.setter
    def device(self, value):
        self.__dict__["_compute_device"] = torch.device(value)

    def embed(self, feature_ids, weights):
        """Node embeddings in the model's space.  feature_ids, weights: (B, T); weights are 0 for
        padding."""
        return self.space.to_space(self.embed_groups([(feature_ids, weights)])[0])

    def embed_groups(self, groups):
        """Pools several (feature ids, weights) groups, e.g. anchors, reconstructions and
        negatives, looking each distinct feature up once across all of them.

        A batch repeats features heavily (shared words, neighbours shared between anchors, a
        neighbour's features counted for every anchor it neighbours), so looking up unique
        features keeps the gathered rows, and the sparse gradient the optimiser has to merge,
        to the batch's distinct features rather than one row per token.
        """
        valids = [w > 0 for _, w in groups]
        ids = torch.cat([f[v] for (f, _), v in zip(groups, valids)])
        unique, inverse = torch.unique(ids, return_inverse=True)
        # Computed in float32 whatever the table's storage type.  The table may live in host
        # memory: only this batch's distinct rows are copied over, and their gradients copied
        # back, through autograd.
        table_device = self.features.weight.device
        table = self.features(unique.to(table_device)).to(unique.device, non_blocking=True).float()
        # Each distinct feature's contribution to its rows' means: itself and a weight of 1, or
        # in the Poincaré ball the gyromidpoint's terms of its point there.
        space = self.space
        if space.pools_in_space:
            num, den = space.mean_terms(space.feature_points(table))
        importance = None
        if self.importance is not None:
            importance = torch.exp(self.importance(unique.to(table_device)).to(unique.device).float()[:, 0])
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
            wv = w[v][:, None]
            summed = torch.zeros(f.shape[0], table.shape[1], device=table.device, dtype=table.dtype)
            if space.pools_in_space:
                summed.index_add_(0, rows, num[index] * wv)
                total = torch.zeros(f.shape[0], 1, device=table.device, dtype=table.dtype)
                total.index_add_(0, rows, den[index] * wv)
                mean = space.mean_finish(summed, total)
            elif importance is not None:
                wi = wv * importance[index][:, None]
                summed.index_add_(0, rows, table[index] * wi)
                total = torch.zeros(f.shape[0], 1, device=table.device, dtype=table.dtype)
                total.index_add_(0, rows, wi)
                mean = summed / total.clamp(min=1e-12)
            else:
                summed.index_add_(0, rows, table[index] * wv)
                mean = summed / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
            if attention:
                padded = torch.zeros(*f.shape, projected.shape[1], device=table.device, dtype=projected.dtype)
                padded[v] = projected[index]
                tokens = None
                if self.pool.mode == "weights":
                    tokens = torch.zeros(*f.shape, table.shape[1], device=table.device, dtype=table.dtype)
                    tokens[v] = table[index]
                out.append(self.pool.attend(padded, mean, w, tokens))
            else:
                out.append(mean)
        return out

    def reconstruct(self, neighbours, weights):
        """A nested reconstruction h~(v) as an embedding in the model's space, from (B, K, d)
        pooled neighbours (embed_groups' output) and their (B, K) weights, 0 for padding: the
        gyromidpoint of the neighbours' points where the space pools neighbours there, else
        combine_neighbours mapped into the space."""
        if self.space.neighbours_in_space:
            return self.space.ball_mean(self.space.to_space(neighbours), weights)
        return self.space.to_space(self.combine_neighbours(neighbours, weights))

    def combine_neighbours(self, neighbour_embeddings, weights):
        """Nested reconstruction's second level: (B, K, d) neighbour embeddings and (B, K)
        weights, 0 for padding, pooled into one (B, d) embedding per node."""
        return self.neighbour_pool(neighbour_embeddings, weights)

    def dense_parameters(self):
        return [p for name, p in self.named_parameters()
                if not name.startswith(("features.", "importance."))]
