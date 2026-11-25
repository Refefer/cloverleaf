"""Trains EP on the GPU and exports the embeddings in cloverleaf's formats.

    python -m ep_gpu.train --edges GRAPH --features FEATURES --out PREFIX [options]

writes PREFIX.node-embeddings (every node, pooled from all its features), PREFIX.feature-embeddings
(the feature table; with mean pooling, cloverleaf's averaged NodeEmbedder reproduces the node
embeddings from it), PREFIX.aggregator (the pooling, for cloverleaf's FeatureAggregator.load, so
cloverleaf's NodeEmbedder embeds new nodes as the model does, attention included) and PREFIX.pt
(the model).
"""
import argparse
import math
import time
from dataclasses import asdict, dataclass

import torch

from . import data, geometry, losses, ppr, sampling
from .validation import RankingProbe, hold_out_edges
from .model import AttentionPool, EPModel
from .optim import TableOptimizers, EpochGradients, SparseTableOptimizer


@dataclass
class Config:
    dims: int = 128
    passes: int = 100
    batch_size: int = 1024
    lr: float = 0.01
    dense_lr: float = 1e-3           # the attention weights' rate; they share lr's schedule shape.
                                     # Dense matrices need far smaller steps than sparse embeddings.
    lr_schedule: str = "cosine"      # "cosine" (20% warmup from lr/100, then cosine to lr/100),
                                     # "constant", or "sawtooth": lr_cycles equal cycles, each a
                                     # linear decay from lr to lr/100 (the first after the warmup),
                                     # so the rate jumps back to lr at every restart
    lr_cycles: int = 1               # sawtooth only: the number of cycles
    init: str = "unit"               # "unit" or "glorot"
    loss: str = "margin"             # margin, starspace, rank, rankspace, infonce; or listnet or
                                     # listmle over each node's PPR list rather than reconstructing
    gamma: float = 1.0               # the loss's margin / threshold
    negatives: int = 1
    negative_pool: int | None = 2048  # nodes embedded once per batch that every anchor draws its
                                      # negatives from; 0 or None draws independent ones per anchor
    pool_negatives: bool = False     # infonce only: use the whole pool as each node's negatives
                                     # (one matrix multiply) instead of `negatives` drawn from it
    mask_neighbours: bool = False    # infonce only: a negative that is one of the anchor's own
                                     # neighbours doesn't count (a false negative)
    starspace_weight: float = 0.0    # infonce only: the loss is (1 - w) infonce + w StarSpace over
    starspace_margin: float = 0.1    # the same negatives, with this margin
    ppr_k: int = 10                  # listwise losses: the PPR list's length,
    ppr_sample: int | None = None    # entries drawn from it per node each time it's used (None: all),
    ppr_sample_by: str = "weight"    # drawn in proportion to their PPR weight, or "uniform",
    ppr_walks: int = 1000            # walks per node estimating it,
    ppr_steps: float = 0.5           # < 1: each walk's restart probability after each step, so walks
                                     # average 1/p steps; a whole number >= 1: a fixed walk length
                                     # (odd lengths on a bipartite graph always end on the other side),
    ppr_beta: float = 0.8            # the popularity discount: scores / degree^beta,
    ppr_compression: float = 1.0     # weights are score^compression, normalised,
    ppr_other_type: bool = False     # only nodes of another type, e.g. a query's products,
    ppr_passive: bool = False        # listnet: skip positives already as likely as their weight
    max_nodes: int | None = None     # neighbours per reconstruction; None for all
    max_features: float | int | None = None   # features per node; a fraction, a count or None for all
    weighted_sampling: bool = False
    weighted_averaging: bool = False
    pooling: str = "mean"            # "mean" or "attention"
    reconstruction: str = "auto"     # h~(v): "flat" (cloverleaf), "nested", or "auto": flat for
                                     # mean pooling, nested for attention
    neighbour_pooling: str = "mean"  # nested h~(v)'s second level: a weighted "mean" of the
                                     # neighbours (same space as h(v)) or "attention"
    heads: int = 4
    attention_keeps_norm: bool = False  # attention changes only the pooled vector's direction, its
                                     # length staying the features' mean's: in the Poincaré ball,
                                     # the distance from the origin then follows the node's own
                                     # features rather than the shared output projection
    attention_dropout: float = 0.0   # in training, dropout on attention's added term (residual
                                     # dropout), regularising the output projection every node shares
    feature_importance: bool = False  # mean pooling: a learned scalar per feature, weighting it by
                                     # exp(s) in every mean it's pooled into (a learned IDF); starts
                                     # at 0, the plain mean.  Batch updates, table on the device.
    attention_mode: str = "transform"  # "transform": mean + out(ctx), a shared linear map of the
                                     # attended tokens; "weights": per-node feature importance
                                     # only, the node a convex combination of its own features
    precision: str = "bf16"          # "bf16": the forward pass under bfloat16 autocast (fastest for
                                     # attention); "tf32": float32 matrix multiplies on tensor
                                     # cores; "fp32": exact.  The feature table, optimiser state
                                     # and losses stay float32.
    optimizer: str = "adam"          # the feature table's: adam, rowwise_adam or rowwise_adagrad
                                     # (see optim.py for the memory each needs)
    table_dtype: str = "float32"     # "float32" or "bfloat16" (updates stochastically rounded)
    graph_device: str = "same"       # "same" (as device) or "cpu": keep the graph in host memory
                                     # and sample there, copying each batch's ids over
    table_device: str = "same"       # "same" or "cpu": keep the feature table and its optimiser
                                     # state in pinned host memory, copying each batch's rows over
    update_frequency: str = "batch"  # "batch": an update after every batch; "epoch": every batch of
                                     # a pass computes its gradients against the same embeddings,
                                     # they're summed, and one update applies them at the pass's end
                                     # (cloverleaf's update_frequency).  Only `passes` updates, so it
                                     # needs a larger lr or more passes; keeps a float32 gradient
                                     # table the size of the feature table.
    epoch_after: int | None = None   # update_frequency "epoch" only: batch updates for this many
                                     # passes first, then epoch updates.  The schedule runs over
                                     # passes either way, so the switch doesn't jump the rate.
    valid_pct: float = 0.0           # nodes held out of the anchors; their loss is reported, and
                                     # how well their neighbours outrank random nodes
    valid_positives: int = 5         # neighbours per validation node that ranking check scores,
    valid_negatives: int = 1000      # against this many random nodes of each neighbour's type
                                     # (0 turns the ranking check off; see validation.py)
    keep_best_probe: bool = False    # export the pass whose ranking check had the best hr@10
                                     # rather than the last (early stopping; needs the check)
    valid_edges: float = 0.0         # this fraction of edges is removed from training (with their
                                     # reverses) and the ranking check scores them instead: link
                                     # prediction on unseen edges rather than on trained ones
    geometry: str = "euclidean"      # the embeddings' space (geometry.py): "euclidean", compared by
                                     # cosine, or "poincare", the Poincaré ball, compared by
                                     # hyperbolic distance (gamma is then in distance units for
                                     # margin and starspace, and infonce's and the listwise
                                     # losses' logits are -distance / gamma)
    curvature: float = 1.0           # poincare: the ball's curvature is -curvature
    poincare_pooling: str = "tangent"  # poincare: "tangent" (a mean of tangent vectors, mapped into
                                     # the ball), "midpoint" (features mapped into the ball, then
                                     # their gyromidpoint; mean pooling only) or
                                     # "neighbour_midpoint" (features as tangent; a nested h~(v) is
                                     # the gyromidpoint of its neighbours' embeddings)
    max_tangent_norm: float | None = None  # poincare: clip pooled tangent vectors to this norm
                                     # before mapping them into the ball
    hard_negatives: int = 0          # margin, margin_sq, starspace: also use the k pool nodes
                                     # nearest h~(v) as negatives, other than v's own neighbours,
                                     # alongside the `negatives` random ones (0 random: hard only)
    negative_types: str | None = None  # which node types each anchor type's negatives may be, as
                                     # "query=product;product=product,query"; anchor types left out
                                     # may draw any type.  Needs a negative_pool.
    adaptive_margin: bool = False    # margin_sq: HICF's geometry-aware margin, gamma * sigmoid(.)
    seed: int = 2017
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


LR_SCHEDULES = ("cosine", "constant", "sawtooth")


def learning_rate(cfg, step, total):
    if cfg.lr_schedule == "constant" or total == 0:
        return cfg.lr
    low = cfg.lr / 100
    if cfg.lr_schedule == "sawtooth":
        cycle = total / cfg.lr_cycles
        k = min(int(step // cycle), cfg.lr_cycles - 1)
        start, warmup = k * cycle, (cycle // 5 if k == 0 else 0)
        if step - start < warmup:
            return low + (cfg.lr - low) * (step - start) / warmup
        ratio = min((step - start - warmup) / max(cycle - warmup, 1), 1.0)
        return cfg.lr - (cfg.lr - low) * ratio
    warmup = total // 5
    if step < warmup:
        return low + (cfg.lr - low) * step / warmup
    ratio = min((step - warmup) / max(total - warmup, 1), 1.0)
    return low + 0.5 * (cfg.lr - low) * (1 + math.cos(math.pi * ratio))


def set_precision(cfg, device):
    if cfg.precision not in ("fp32", "tf32", "bf16"):
        raise ValueError(f"precision must be fp32, tf32 or bf16, got {cfg.precision!r}")
    # Process-wide, as PyTorch's matmul precision is.
    torch.set_float32_matmul_precision("highest" if cfg.precision == "fp32" else "high")


def autocast(cfg, device):
    enabled = cfg.precision == "bf16" and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


TABLE_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
UPDATE_FREQUENCIES = ("batch", "epoch")


def listwise(cfg):
    return cfg.loss in losses.LISTWISE


def check_loss(cfg):
    names = {**losses.LOSSES, **losses.LISTWISE}
    if cfg.loss not in names:
        raise ValueError(f"loss must be one of {', '.join(names)}, got {cfg.loss!r}")
    if cfg.pool_negatives and (cfg.loss != "infonce" and not listwise(cfg) or not cfg.negative_pool):
        raise ValueError("pool_negatives needs loss infonce, listnet or listmle, and a negative_pool")
    if listwise(cfg) and not cfg.negative_pool:
        raise ValueError(f"{cfg.loss} draws its negatives from a negative_pool")
    if (cfg.mask_neighbours or cfg.starspace_weight) and cfg.loss != "infonce":
        raise ValueError("mask_neighbours and starspace_weight apply only to infonce")
    if not 0 <= cfg.starspace_weight <= 1:
        raise ValueError(f"starspace_weight must be in [0, 1], got {cfg.starspace_weight}")
    if cfg.ppr_passive and cfg.loss != "listnet":
        raise ValueError("ppr_passive applies only to listnet")
    if not (0 < cfg.ppr_steps < 1 or (cfg.ppr_steps >= 1 and float(cfg.ppr_steps).is_integer())):
        raise ValueError(f"ppr_steps must be in (0, 1) or a whole number >= 1, got {cfg.ppr_steps}")
    if cfg.ppr_sample is not None and not 0 < cfg.ppr_sample <= cfg.ppr_k:
        raise ValueError(f"ppr_sample must be in 1..ppr_k ({cfg.ppr_k}), got {cfg.ppr_sample}")
    if cfg.geometry not in geometry.GEOMETRIES:
        raise ValueError(f"geometry must be one of {', '.join(geometry.GEOMETRIES)}, got {cfg.geometry!r}")
    if cfg.geometry != "euclidean" and cfg.loss in ("rank", "rankspace"):
        raise ValueError(f"{cfg.loss} scores by dot product, so it needs geometry 'euclidean'")
    if not cfg.curvature > 0:
        raise ValueError(f"curvature must be > 0, got {cfg.curvature}")
    if cfg.poincare_pooling not in geometry.POINCARE_POOLING:
        raise ValueError(f"poincare_pooling must be one of {', '.join(geometry.POINCARE_POOLING)}, got {cfg.poincare_pooling!r}")
    if cfg.poincare_pooling != "tangent" and cfg.geometry != "poincare":
        raise ValueError("poincare_pooling applies only to geometry 'poincare'")
    if cfg.poincare_pooling == "midpoint" and "attention" in (cfg.pooling, cfg.neighbour_pooling):
        raise ValueError(f"poincare_pooling {cfg.poincare_pooling!r} needs pooling and neighbour_pooling 'mean'")
    if cfg.poincare_pooling == "neighbour_midpoint" and cfg.neighbour_pooling != "mean":
        raise ValueError("poincare_pooling 'neighbour_midpoint' needs neighbour_pooling 'mean'")
    if cfg.poincare_pooling == "neighbour_midpoint" and not nested_reconstruction(cfg):
        raise ValueError("poincare_pooling 'neighbour_midpoint' needs the nested reconstruction "
                         "(reconstruction 'nested', or attention pooling)")
    if cfg.max_tangent_norm is not None and (cfg.geometry != "poincare" or not cfg.max_tangent_norm > 0):
        raise ValueError("max_tangent_norm needs geometry 'poincare' and a value > 0")
    if cfg.hard_negatives < 0 or (cfg.hard_negatives and (
            cfg.loss not in ("margin", "margin_sq", "starspace") or not cfg.negative_pool
            or cfg.hard_negatives > cfg.negative_pool)):
        raise ValueError("hard_negatives needs loss margin, margin_sq or starspace, and a "
                         "negative_pool at least that large")
    if cfg.negatives < 0 or (cfg.negatives + cfg.hard_negatives < 1 and not cfg.pool_negatives
                             and not listwise(cfg)):
        raise ValueError("needs at least one negative: negatives, hard_negatives or pool_negatives")
    if cfg.negative_types is not None:
        sampling.parse_negative_types(cfg.negative_types)
        if not cfg.negative_pool or listwise(cfg):
            raise ValueError("negative_types needs a negative_pool and a non-listwise loss")
    if (cfg.attention_keeps_norm or cfg.attention_dropout) and "attention" not in (cfg.pooling, cfg.neighbour_pooling):
        raise ValueError("attention_keeps_norm and attention_dropout need attention pooling")
    if cfg.feature_importance and (cfg.pooling != "mean" or cfg.neighbour_pooling != "mean"
                                   or cfg.poincare_pooling == "midpoint" or cfg.update_frequency != "batch"
                                   or cfg.table_device != "same"):
        raise ValueError("feature_importance needs mean pooling, mean neighbour pooling, no midpoint "
                         "pooling, batch updates and table_device 'same'")
    if cfg.attention_mode != "transform" and "attention" not in (cfg.pooling, cfg.neighbour_pooling):
        raise ValueError("attention_mode needs attention pooling")
    if cfg.attention_mode == "weights" and (cfg.attention_keeps_norm or cfg.attention_dropout):
        raise ValueError("attention_keeps_norm and attention_dropout apply to attention_mode 'transform' only")
    if not 0 <= cfg.attention_dropout < 1:
        raise ValueError(f"attention_dropout must be in [0, 1), got {cfg.attention_dropout}")
    if cfg.adaptive_margin and cfg.loss != "margin_sq":
        raise ValueError("adaptive_margin applies only to margin_sq")
    if cfg.ppr_sample_by not in ppr.SAMPLE_BY:
        raise ValueError(f"ppr_sample_by must be one of {', '.join(ppr.SAMPLE_BY)}, got {cfg.ppr_sample_by!r}")


def placement(cfg, setting):
    """The device for graph_device or table_device: "same" as cfg.device, or "cpu"."""
    value = getattr(cfg, setting)
    if value not in ("same", "cpu"):
        raise ValueError(f"{setting} must be 'same' or 'cpu', got {value!r}")
    return torch.device(cfg.device if value == "same" else "cpu")


def build(cfg, num_features, device, generator=None, zero_init=None):
    """The model, the feature table's optimiser and, with attention, an Adam for its weights.
    zero_init: feature ids whose rows start at zero (Graph.zero_init)."""
    if cfg.table_dtype not in TABLE_DTYPES:
        raise ValueError(f"table_dtype must be float32 or bfloat16, got {cfg.table_dtype!r}")
    model = EPModel(num_features, cfg.dims, cfg.pooling, cfg.heads, cfg.init, cfg.seed,
                    TABLE_DTYPES[cfg.table_dtype], cfg.neighbour_pooling,
                    geometry.make(cfg.geometry, cfg.curvature, cfg.poincare_pooling,
                                  cfg.max_tangent_norm), cfg.attention_keeps_norm,
                    cfg.attention_dropout, cfg.attention_mode, cfg.feature_importance).to(device)
    if zero_init is not None:
        with torch.no_grad():
            model.features.weight[zero_init.to(model.features.weight.device)] = 0
    model.device = device
    table_device = placement(cfg, "table_device")
    if table_device != device:
        model.features.to(table_device)
        if device.type == "cuda":
            # Pinned, so each batch's rows copy to the GPU asynchronously.
            model.features.weight.data = model.features.weight.data.pin_memory()
        generator = torch.Generator(device=table_device).manual_seed(cfg.seed + 2)
    table_opt = SparseTableOptimizer(model.features.weight, cfg.optimizer, cfg.lr, generator=generator)
    if model.importance is not None:
        table_opt = TableOptimizers(table_opt, SparseTableOptimizer(model.importance.weight, cfg.optimizer,
                                                                    cfg.lr, generator=generator))
    dense = model.dense_parameters()
    dense_opt = torch.optim.Adam(dense, lr=cfg.dense_lr) if dense else None
    return model, table_opt, dense_opt


def nested_reconstruction(cfg):
    if cfg.reconstruction not in ("auto", "flat", "nested"):
        raise ValueError(f"reconstruction must be auto, flat or nested, got {cfg.reconstruction!r}")
    return cfg.reconstruction == "nested" or (cfg.reconstruction == "auto" and cfg.pooling == "attention")


def listwise_loss(model, graph, lists, nodes, cfg, generator):
    """Mean listwise loss over `nodes` whose PPR list isn't empty."""
    pos, pos_w = lists.nodes[nodes], lists.weights[nodes]
    if cfg.ppr_sample is not None and cfg.ppr_sample < cfg.ppr_k:
        pos, pos_w = ppr.subsample(pos, pos_w, cfg.ppr_sample, cfg.ppr_sample_by, generator)
    fidx, fmask = sampling.sample_features(graph, nodes, cfg.max_features, generator)
    pidx, pmask = sampling.sample_features(graph, pos.reshape(-1), cfg.max_features, generator)
    pmask &= (pos_w.reshape(-1) > 0)[:, None]
    pool = torch.randint(0, graph.num_nodes, (cfg.negative_pool,), device=nodes.device,
                         generator=generator)
    nidx, nmask = sampling.sample_features(graph, pool, cfg.max_features, generator)
    valid = pos_w > 0
    if cfg.pool_negatives:
        # Every pool node but the anchor and its own positives.
        clash = (pool[None, :] == nodes[:, None]) | (
            (pool[None, None, :] == pos[:, :, None]) & valid[:, :, None]).any(1)
    else:
        pick = sampling.pick_from_pool(pool, nodes, cfg.negatives, generator)
        clash = ((pool[pick][:, :, None] == pos[:, None, :]) & valid[:, None, :]).any(2)
    device = model.device
    move = lambda x: x.to(device, non_blocking=True)
    fidx, fmask, pidx, pmask, nidx, nmask, pos_w, clash = map(
        move, (fidx, fmask, pidx, pmask, nidx, nmask, pos_w, clash))
    with autocast(cfg, device):
        h, hp, neg = model.embed_groups([(fidx, fmask.float()), (pidx, pmask.float()),
                                         (nidx, nmask.float())])
    space = model.space
    h, hp, neg = (space.to_space(x.float()) for x in (h, hp, neg))
    pos_logits = losses.logits(h, hp.view(len(nodes), pos_w.shape[1], -1), cfg.gamma, space)
    if cfg.pool_negatives:
        neg_logits = losses.pool_logits(h, neg, cfg.gamma, space)
    else:
        neg_logits = losses.logits(h, neg[move(pick)], cfg.gamma, space)
    neg_logits = neg_logits.masked_fill(clash, losses.MASKED)
    if cfg.loss == "listnet":
        per_node = losses.listnet(pos_logits, pos_w, neg_logits, cfg.ppr_passive)
    else:
        per_node = losses.listmle(pos_logits, pos_w, neg_logits)
    has_list = pos_w.sum(1) > 0
    return (per_node * has_list).sum() / has_list.sum().clamp(min=1)


def build_lists(edges_path, graph, cfg, directed=False, log=print):
    """The PPR lists a listwise loss trains on, built by cloverleaf from the graph's edge file."""
    start = time.time()
    lists = ppr.ppr_lists(edges_path, graph, cfg.ppr_k, cfg.ppr_walks, cfg.ppr_steps, cfg.ppr_beta,
                          cfg.ppr_compression, cfg.ppr_other_type, directed, cfg.seed)
    empty = int((lists.weights.sum(1) == 0).sum())
    log(f"ppr lists in {time.time() - start:.1f}s; {empty} of {graph.num_nodes} nodes reach nothing")
    return lists


def batch_loss(model, graph, nodes, cfg, generator, lists=None, keys=None, negative_types=None):
    """Mean loss over `nodes` that have neighbours (or, for a listwise loss, a PPR list).  Sampling
    runs where the graph is (nodes and generator are on its device); the embedding and loss run
    on model.device.  keys: sampling.edge_keys(graph), for mask_neighbours and hard_negatives;
    negative_types: a sampling.NegativeTypes restricting each anchor's negatives by node type."""
    if listwise(cfg):
        return listwise_loss(model, graph, lists, nodes, cfg, generator)
    fidx, fmask = sampling.sample_features(graph, nodes, cfg.max_features, generator)
    nested = nested_reconstruction(cfg)
    neighbours = (sampling.neighbour_features if nested else sampling.neighbourhood_tokens)(
        graph, nodes, cfg.max_nodes, cfg.max_features, cfg.weighted_sampling,
        cfg.weighted_averaging, generator)
    if nested:
        tidx, tmask, nw = neighbours
        tw = tmask.float()
    else:
        tidx, tw = neighbours
    allowed = None
    if cfg.negative_pool:
        if negative_types is not None:
            pool = negative_types.pool(cfg.negative_pool, generator)
            allowed = negative_types.allowed_mask(nodes, pool)
            pick = sampling.pick_allowed(allowed & (pool[None, :] != nodes[:, None]), cfg.negatives, generator)
        else:
            pool = torch.randint(0, graph.num_nodes, (cfg.negative_pool,), device=nodes.device,
                                 generator=generator)
            pick = sampling.pick_from_pool(pool, nodes, cfg.negatives, generator)
        neg_nodes = pool
    else:
        neg_nodes = sampling.sample_negatives(graph.num_nodes, nodes, cfg.negatives, generator).reshape(-1)
    nidx, nmask = sampling.sample_features(graph, neg_nodes, cfg.max_features, generator)
    device = model.device
    move = lambda x: x.to(device, non_blocking=True)
    fidx, fmask, tidx, tw, nidx, nmask = map(move, (fidx, fmask, tidx, tw, nidx, nmask))
    if nested:
        nw = move(nw)
    if cfg.negative_pool:
        pick = move(pick)
    with autocast(cfg, device):
        h, t, neg = model.embed_groups([(fidx, fmask.float()), (tidx, tw), (nidx, nmask.float())])
        if nested:
            t = model.reconstruct(t.view(len(nodes), nw.shape[1], -1), nw)
            tw = nw
    # Losses in float32: they normalise and take logs.  Pooled vectors become embeddings in the
    # model's space only now (a nested reconstruction already is one).
    space = model.space
    h, neg = (space.to_space(x.float()) for x in (h, neg))
    t = t.float() if nested else space.to_space(t.float())
    if cfg.loss == "infonce" and (cfg.pool_negatives or cfg.mask_neighbours or cfg.starspace_weight):
        # Against the whole pool, or `negatives` picked from it, leaving out the anchor itself
        # and, with mask_neighbours, its own neighbours.
        if cfg.pool_negatives:
            neg_ids = pool[None, :].expand(len(nodes), -1)
            pos, neg_sim = losses.similarities(t, h, pool=neg, space=space)
        else:
            neg_ids = pool[pick.to(pool.device)] if cfg.negative_pool else neg_nodes.view(len(nodes), -1)
            hu = neg[pick] if cfg.negative_pool else neg.view(len(nodes), cfg.negatives, -1)
            pos, neg_sim = losses.similarities(t, h, hu=hu, space=space)
        exclude = neg_ids == nodes[:, None]
        if allowed is not None:
            exclude = exclude | ~(allowed if cfg.pool_negatives else allowed.gather(1, pick.to(allowed.device)))
        if cfg.mask_neighbours:
            exclude = exclude | sampling.are_neighbours(keys, graph.num_nodes, nodes, neg_ids)
        exclude = move(exclude)
        per_node = losses.infonce_from_similarities(pos, neg_sim, cfg.gamma, exclude)
        if cfg.starspace_weight:
            w = cfg.starspace_weight
            per_node = (1 - w) * per_node + w * losses.starspace_from_similarities(pos, neg_sim, cfg.starspace_margin, exclude)
    else:
        if cfg.hard_negatives:
            # The pool nodes nearest the reconstruction, leaving out the anchor itself and its own
            # neighbours: the nearest nodes are often real neighbours, and pushing those away
            # would fight the positive term (false negatives).
            with torch.no_grad():
                d = space.pairwise_distance(t, neg)
                pool_ids = pool[None, :].expand(len(nodes), -1)
                own = (pool_ids == nodes[:, None]) | sampling.are_neighbours(keys, graph.num_nodes, nodes, pool_ids)
                if allowed is not None:
                    own = own | ~allowed
                d = d.masked_fill(move(own), float("inf"))
                hard = d.topk(cfg.hard_negatives, dim=1, largest=False).indices
            # Mixed: the random negatives and the hard ones together.
            hu = torch.cat([neg[pick], neg[hard]], dim=1)
        else:
            hu = neg[pick] if cfg.negative_pool else neg.view(len(nodes), cfg.negatives, -1)
        if cfg.loss == "margin_sq":
            per_node = losses.margin_sq(t, h, hu, cfg.gamma, space, cfg.adaptive_margin)
        else:
            per_node = losses.LOSSES[cfg.loss](t, h, hu, cfg.gamma, space)
    has_neighbours = tw.sum(dim=1) > 0
    return (per_node * has_neighbours).sum() / has_neighbours.sum().clamp(min=1)


def train(graph, cfg, log=print, lists=None):
    """lists: the PPR lists (build_lists) a listwise loss needs."""
    device = torch.device(cfg.device)
    check_loss(cfg)
    if listwise(cfg) and lists is None:
        raise ValueError(f"{cfg.loss} needs PPR lists; see build_lists")
    if cfg.update_frequency not in UPDATE_FREQUENCIES:
        raise ValueError(f"update_frequency must be one of {', '.join(UPDATE_FREQUENCIES)}, "
                         f"got {cfg.update_frequency!r}")
    if cfg.lr_schedule not in LR_SCHEDULES:
        raise ValueError(f"lr_schedule must be one of {', '.join(LR_SCHEDULES)}, got {cfg.lr_schedule!r}")
    if cfg.lr_cycles < 1 or (cfg.lr_cycles > 1 and cfg.lr_schedule != "sawtooth"):
        raise ValueError(f"lr_cycles must be >= 1, and above 1 only with lr_schedule 'sawtooth', got {cfg.lr_cycles}")
    if cfg.epoch_after is not None and (cfg.update_frequency != "epoch" or cfg.epoch_after < 1):
        raise ValueError(f"epoch_after needs update_frequency 'epoch' and a value >= 1, got {cfg.epoch_after!r}")
    set_precision(cfg, device)
    graph_device = placement(cfg, "graph_device")
    g = graph.to(graph_device)
    if not 0 <= cfg.valid_edges < 1:
        raise ValueError(f"valid_edges must be in [0, 1), got {cfg.valid_edges}")
    held_out = None
    if cfg.valid_edges > 0:
        edge_gen = torch.Generator(device=graph_device).manual_seed(cfg.seed + 4)
        g, *held_out = hold_out_edges(g, cfg.valid_edges, edge_gen)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=graph_device).manual_seed(cfg.seed)
    model, table_opt, dense_opt = build(cfg, g.num_features, device,
                                        torch.Generator(device=device).manual_seed(cfg.seed + 2),
                                        graph.zero_init)

    if lists is not None:
        lists = lists.to(graph_device)
    order = torch.randperm(g.num_nodes, device=graph_device, generator=gen)
    n_valid = int(g.num_nodes * cfg.valid_pct)
    valid, anchors = order[:n_valid], order[n_valid:]
    keys = sampling.edge_keys(g) if cfg.mask_neighbours or cfg.hard_negatives else None
    negative_types = (sampling.NegativeTypes(g.node_types, sampling.parse_negative_types(cfg.negative_types),
                                             graph_device) if cfg.negative_types is not None else None)
    probe = None
    if cfg.valid_negatives > 0:
        if held_out is not None:
            probe = RankingProbe(g, *held_out, cfg.valid_negatives, cfg.seed + 3)
        elif len(valid) and cfg.valid_positives > 0:
            probe = RankingProbe.from_nodes(g, valid, cfg.valid_positives, cfg.valid_negatives, cfg.seed + 3)
    steps_per_pass = math.ceil(len(anchors) / cfg.batch_size)
    # Epoch updates: the table's gradients are summed here; the attention weights' accumulate in
    # their .grad, which is cleared only once a pass.  They start after `epoch_after` passes of
    # batch updates, or from the first pass.
    epoch = EpochGradients(model.features.weight) if cfg.update_frequency == "epoch" else None
    epoch_from = (cfg.epoch_after or 0) if epoch else cfg.passes
    # The schedule's position is in batches, so it advances at the same rate per pass whichever
    # kind of update a pass makes; an epoch update uses the rate at the start of its pass.
    total = steps_per_pass * cfg.passes

    def set_lr(lr):
        table_opt.lr = lr
        if dense_opt is not None:
            for group in dense_opt.param_groups:
                group["lr"] = lr * cfg.dense_lr / cfg.lr

    if cfg.keep_best_probe and probe is None:
        raise ValueError("keep_best_probe needs the ranking check: valid_edges or valid_pct, and valid_negatives")
    best = (-1.0, 0, None)   # (hr@10, pass, model state) with keep_best_probe
    for p in range(cfg.passes):
        per_pass = p >= epoch_from
        start, running, lr = time.time(), 0.0, learning_rate(cfg, p * steps_per_pass, total)
        perm = anchors[torch.randperm(len(anchors), device=graph_device, generator=gen)]
        model.train()
        if per_pass and dense_opt is not None:
            dense_opt.zero_grad()
        for b in range(steps_per_pass):
            nodes = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            table_opt.zero_grad()
            if not per_pass:
                lr = learning_rate(cfg, p * steps_per_pass + b, total)
                set_lr(lr)
                if dense_opt is not None:
                    dense_opt.zero_grad()
            loss = batch_loss(model, g, nodes, cfg, gen, lists, keys, negative_types)
            loss.backward()
            if per_pass:
                epoch.add(model.features.weight.grad)
            else:
                table_opt.step()
                if dense_opt is not None:
                    dense_opt.step()
            running = running + loss.detach() * len(nodes)
        if per_pass:
            set_lr(lr)
            table_opt.zero_grad()
            epoch.apply(table_opt)
            if dense_opt is not None:
                dense_opt.step()
        message = f"pass {p + 1}/{cfg.passes}: loss {float(running) / len(anchors):.5f}, lr {lr:.5f}"
        if len(valid):
            model.eval()
            vgen = torch.Generator(device=graph_device).manual_seed(cfg.seed + 1)
            total_vl = torch.zeros((), device=device)
            with torch.no_grad():
                # In training-sized batches: attention pads every row, so all the held-out nodes
                # at once may not fit.
                for vs in range(0, len(valid), cfg.batch_size):
                    chunk = valid[vs:vs + cfg.batch_size]
                    total_vl += batch_loss(model, g, chunk, cfg, vgen, lists, keys, negative_types) * len(chunk)
            message += f", valid {float(total_vl) / len(valid):.5f}"
        if probe is not None:
            hr, auc = probe.evaluate(model, g)
            message += f", {'held-out edges' if held_out is not None else 'valid'} hr@10 {hr:.4f} auc {auc:.4f}"
            if cfg.keep_best_probe and hr > best[0]:
                best = (hr, p + 1, {k: v.detach().clone() for k, v in model.state_dict().items()})
        log(f"{message} ({time.time() - start:.1f}s)")
    if best[2] is not None:
        model.load_state_dict(best[2])
        log(f"kept pass {best[1]}, held-out hr@10 {best[0]:.4f}")
    return model


@torch.no_grad()
def node_embeddings(model, graph, batch_size=8192):
    """Every node, pooled from all its features, in the model's space.  Gathered in host memory a batch at a time, so
    exporting needs no GPU memory for the whole set."""
    model.eval()
    out = []
    for s in range(0, graph.num_nodes, batch_size):
        nodes = torch.arange(s, min(s + batch_size, graph.num_nodes), device=graph.adj_ptr.device)
        fidx, fmask = sampling.sample_features(graph, nodes, None)
        out.append(model.embed(fidx.to(model.device), fmask.float().to(model.device)).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def embed_feature_sets(model, feature_index, feature_sets):
    """Embeds nodes that weren't in the graph from their features, e.g. nodes added after
    training, in the model's space.  Features missing from the training vocabulary are skipped; a
    set with none left embeds to zeros (the origin)."""
    model.eval()
    device = model.device
    ids = [[feature_index[f] for f in fs if f in feature_index] for fs in feature_sets]
    width = max((len(x) for x in ids), default=0) or 1
    idx = torch.zeros(len(ids), width, dtype=torch.long, device=device)
    w = torch.zeros(len(ids), width, device=device)
    for i, x in enumerate(ids):
        idx[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        w[i, :len(x)] = 1
    return model.embed(idx, w)


def aggregator_representable(model):
    """Whether cloverleaf's FeatureAggregator reproduces the model's node embeddings: it pools
    features (mean, or attention as mean + out(ctx)) and maps the result into its geometry,
    Euclidean or the Poincaré ball after tangent pooling; it has no per-feature importance,
    norm-keeping or weighting-only attention, or midpoint pooling."""
    pool, space = model.pool, model.space
    space_ok = space is geometry.EUCLIDEAN or (space.name == "poincare" and space.pooling == "tangent")
    return (space_ok and model.importance is None
            and (not isinstance(pool, AttentionPool) or (pool.mode == "transform" and not pool.keeps_norm)))


def export(model, graph, cfg, prefix):
    # The model first: it's what everything else can be rebuilt from.
    torch.save({"config": asdict(cfg), "state": model.state_dict(),
                "feature_keys": graph.feature_keys}, prefix + ".pt")
    keys = list(zip(graph.node_types, graph.node_names))
    data.save_embeddings(prefix + ".node-embeddings", keys, node_embeddings(model, graph))
    data.save_embeddings(prefix + ".feature-embeddings", graph.feature_keys, model.features.weight.float())
    if aggregator_representable(model):
        data.save_aggregator(prefix + ".aggregator", model)


def main():
    p = argparse.ArgumentParser(description="Embedding Propagation on the GPU.")
    p.add_argument("--edges", required=True)
    p.add_argument("--features", default=None)
    p.add_argument("--out", default=None, help="Output prefix; required unless --no-export.")
    p.add_argument("--no-export", action="store_true", help="Train only, e.g. to time it.")
    p.add_argument("--graph-cache", default=None,
                   help="Save the parsed graph here and reuse it while the inputs are unchanged.")
    p.add_argument("--directed", action="store_true")
    p.add_argument("--prune-min-count", type=int, default=0,
                   help="Drop features fewer than this many nodes have (cloverleaf's prune_min_count).")
    p.add_argument("--namespace", default="node_type",
                   help="'node_type' namespaces features by node type; anything else is one shared namespace.")
    p.add_argument("--shared-namespace", default=None,
                   help="With --namespace node_type, also give every token a feature in this shared "
                        "namespace; the node-type features start at zero, as offsets from it.")
    defaults = Config()
    for name, value in asdict(defaults).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            p.add_argument(flag, action="store_true")
        elif name == "max_features":
            p.add_argument(flag, type=lambda s: float(s) if "." in s else int(s), default=None)
        elif name in ("max_nodes", "ppr_sample", "epoch_after"):
            p.add_argument(flag, type=int, default=None)
        elif name == "max_tangent_norm":
            p.add_argument(flag, type=float, default=None)
        elif name == "negative_types":
            p.add_argument(flag, type=str, default=None)
        else:
            p.add_argument(flag, type=type(value), default=value)
    args = p.parse_args()
    if args.out is None and not args.no_export:
        p.error("--out is required unless --no-export")
    cfg = Config(**{k: getattr(args, k) for k in asdict(defaults)})
    start = time.time()
    if args.graph_cache:
        graph = data.load_cached(args.graph_cache, args.edges, args.features, args.directed,
                                 args.namespace, args.prune_min_count, args.shared_namespace)
    else:
        graph = data.load(args.edges, args.features, args.directed, args.namespace, args.prune_min_count,
                          args.shared_namespace)
    print(f"loaded in {time.time() - start:.0f}s", flush=True)
    print(f"{graph.num_nodes} nodes, {len(graph.adj_idx)} directed edges, "
          f"{graph.num_features} features; {cfg}", flush=True)
    log = lambda m: print(m, flush=True)
    check_loss(cfg)
    lists = build_lists(args.edges, graph, cfg, args.directed, log) if listwise(cfg) else None
    start = time.time()
    model = train(graph, cfg, log=log, lists=lists)
    peak = (f", peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
            if torch.device(cfg.device).type == "cuda" else "")
    print(f"trained in {time.time() - start:.0f}s{peak}", flush=True)
    if not args.no_export:
        export(model, graph, cfg, args.out)


if __name__ == "__main__":
    main()
