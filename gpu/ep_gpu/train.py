"""Trains EP on the GPU and exports the embeddings in cloverleaf's formats.

    python -m ep_gpu.train --edges GRAPH --features FEATURES --out PREFIX [options]

writes PREFIX.node-embeddings (every node, pooled from all its features), PREFIX.feature-embeddings
(the feature table; with mean pooling, cloverleaf's averaged NodeEmbedder reproduces the node
embeddings from it) and PREFIX.pt (the model, to embed new nodes with attention pooling).
"""
import argparse
import math
import time
from dataclasses import asdict, dataclass

import torch

from . import data, losses, sampling
from .model import EPModel
from .optim import SparseTableOptimizer


@dataclass
class Config:
    dims: int = 128
    passes: int = 100
    batch_size: int = 1024
    lr: float = 0.01
    dense_lr: float = 1e-3           # the attention weights' rate; they share lr's schedule shape.
                                     # Dense matrices need far smaller steps than sparse embeddings.
    lr_schedule: str = "cosine"      # "cosine" (20% warmup from lr/100, then cosine to lr/100) or "constant"
    init: str = "unit"               # "unit" or "glorot"
    loss: str = "margin"             # margin, starspace, rank, rankspace
    gamma: float = 1.0               # the loss's margin / threshold
    negatives: int = 1
    negative_pool: int | None = 2048  # nodes embedded once per batch that every anchor draws its
                                      # negatives from; 0 or None draws independent ones per anchor
    pool_negatives: bool = False     # infonce only: use the whole pool as each node's negatives
                                     # (one matrix multiply) instead of `negatives` drawn from it
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
    valid_pct: float = 0.0           # nodes held out of the anchors; their loss is reported
    seed: int = 2017
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def learning_rate(cfg, step, total):
    if cfg.lr_schedule == "constant" or total == 0:
        return cfg.lr
    low, warmup = cfg.lr / 100, total // 5
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


def check_loss(cfg):
    if cfg.loss not in losses.LOSSES:
        raise ValueError(f"loss must be one of {', '.join(losses.LOSSES)}, got {cfg.loss!r}")
    if cfg.pool_negatives and (cfg.loss != "infonce" or not cfg.negative_pool):
        raise ValueError("pool_negatives needs loss infonce and a negative_pool")


def placement(cfg, setting):
    """The device for graph_device or table_device: "same" as cfg.device, or "cpu"."""
    value = getattr(cfg, setting)
    if value not in ("same", "cpu"):
        raise ValueError(f"{setting} must be 'same' or 'cpu', got {value!r}")
    return torch.device(cfg.device if value == "same" else "cpu")


def build(cfg, num_features, device, generator=None):
    """The model, the feature table's optimiser and, with attention, an Adam for its weights."""
    if cfg.table_dtype not in TABLE_DTYPES:
        raise ValueError(f"table_dtype must be float32 or bfloat16, got {cfg.table_dtype!r}")
    model = EPModel(num_features, cfg.dims, cfg.pooling, cfg.heads, cfg.init, cfg.seed,
                    TABLE_DTYPES[cfg.table_dtype], cfg.neighbour_pooling).to(device)
    model.device = device
    table_device = placement(cfg, "table_device")
    if table_device != device:
        model.features.to(table_device)
        if device.type == "cuda":
            # Pinned, so each batch's rows copy to the GPU asynchronously.
            model.features.weight.data = model.features.weight.data.pin_memory()
        generator = torch.Generator(device=table_device).manual_seed(cfg.seed + 2)
    table_opt = SparseTableOptimizer(model.features.weight, cfg.optimizer, cfg.lr, generator=generator)
    dense = model.dense_parameters()
    dense_opt = torch.optim.Adam(dense, lr=cfg.dense_lr) if dense else None
    return model, table_opt, dense_opt


def nested_reconstruction(cfg):
    if cfg.reconstruction not in ("auto", "flat", "nested"):
        raise ValueError(f"reconstruction must be auto, flat or nested, got {cfg.reconstruction!r}")
    return cfg.reconstruction == "nested" or (cfg.reconstruction == "auto" and cfg.pooling == "attention")


def batch_loss(model, graph, nodes, cfg, generator):
    """Mean EP loss over `nodes` that have neighbours.  Sampling runs where the graph is (nodes and
    generator are on its device); the embedding and loss run on model.device."""
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
    if cfg.negative_pool:
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
            t = model.combine_neighbours(t.view(len(nodes), nw.shape[1], -1), nw)
            tw = nw
    # Losses in float32: they normalise and take logs.
    h, t, neg = h.float(), t.float(), neg.float()
    if cfg.pool_negatives:
        exclude = pool.to(device)[None, :] == nodes.to(device)[:, None]
        per_node = losses.infonce_pool(t, h, neg, exclude, cfg.gamma)
    else:
        hu = neg[pick] if cfg.negative_pool else neg.view(len(nodes), cfg.negatives, -1)
        per_node = losses.LOSSES[cfg.loss](t, h, hu, cfg.gamma)
    has_neighbours = tw.sum(dim=1) > 0
    return (per_node * has_neighbours).sum() / has_neighbours.sum().clamp(min=1)


def train(graph, cfg, log=print):
    device = torch.device(cfg.device)
    check_loss(cfg)
    set_precision(cfg, device)
    graph_device = placement(cfg, "graph_device")
    g = graph.to(graph_device)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=graph_device).manual_seed(cfg.seed)
    model, table_opt, dense_opt = build(cfg, g.num_features, device,
                                        torch.Generator(device=device).manual_seed(cfg.seed + 2))

    order = torch.randperm(g.num_nodes, device=graph_device, generator=gen)
    n_valid = int(g.num_nodes * cfg.valid_pct)
    valid, anchors = order[:n_valid], order[n_valid:]
    steps_per_pass = math.ceil(len(anchors) / cfg.batch_size)
    total, step = steps_per_pass * cfg.passes, 0
    for p in range(cfg.passes):
        start, running, lr = time.time(), 0.0, learning_rate(cfg, step, total)
        perm = anchors[torch.randperm(len(anchors), device=graph_device, generator=gen)]
        model.train()
        for b in range(steps_per_pass):
            nodes = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            lr = learning_rate(cfg, step, total)
            table_opt.lr = lr
            table_opt.zero_grad()
            if dense_opt is not None:
                for group in dense_opt.param_groups:
                    group["lr"] = lr * cfg.dense_lr / cfg.lr
                dense_opt.zero_grad()
            loss = batch_loss(model, g, nodes, cfg, gen)
            loss.backward()
            table_opt.step()
            if dense_opt is not None:
                dense_opt.step()
            running = running + loss.detach() * len(nodes)
            step += 1
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
                    total_vl += batch_loss(model, g, chunk, cfg, vgen) * len(chunk)
            message += f", valid {float(total_vl) / len(valid):.5f}"
        log(f"{message} ({time.time() - start:.1f}s)")
    return model


@torch.no_grad()
def node_embeddings(model, graph, batch_size=8192):
    """Every node, pooled from all its features.  Gathered in host memory a batch at a time, so
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
    training.  Features missing from the training vocabulary are skipped; a set with none left
    embeds to zeros."""
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


def export(model, graph, cfg, prefix):
    keys = list(zip(graph.node_types, graph.node_names))
    data.save_embeddings(prefix + ".node-embeddings", keys, node_embeddings(model, graph))
    data.save_embeddings(prefix + ".feature-embeddings", graph.feature_keys, model.features.weight.float())
    torch.save({"config": asdict(cfg), "state": model.state_dict(),
                "feature_keys": graph.feature_keys}, prefix + ".pt")


def main():
    p = argparse.ArgumentParser(description="Embedding Propagation on the GPU.")
    p.add_argument("--edges", required=True)
    p.add_argument("--features", default=None)
    p.add_argument("--out", default=None, help="Output prefix; required unless --no-export.")
    p.add_argument("--no-export", action="store_true", help="Train only, e.g. to time it.")
    p.add_argument("--graph-cache", default=None,
                   help="Save the parsed graph here and reuse it while the inputs are unchanged.")
    p.add_argument("--directed", action="store_true")
    p.add_argument("--namespace", default="node_type",
                   help="'node_type' namespaces features by node type; anything else is one shared namespace.")
    defaults = Config()
    for name, value in asdict(defaults).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            p.add_argument(flag, action="store_true")
        elif name == "max_features":
            p.add_argument(flag, type=lambda s: float(s) if "." in s else int(s), default=None)
        elif name == "max_nodes":
            p.add_argument(flag, type=int, default=None)
        else:
            p.add_argument(flag, type=type(value), default=value)
    args = p.parse_args()
    if args.out is None and not args.no_export:
        p.error("--out is required unless --no-export")
    cfg = Config(**{k: getattr(args, k) for k in asdict(defaults)})
    start = time.time()
    if args.graph_cache:
        graph = data.load_cached(args.graph_cache, args.edges, args.features, args.directed, args.namespace)
    else:
        graph = data.load(args.edges, args.features, args.directed, args.namespace)
    print(f"loaded in {time.time() - start:.0f}s", flush=True)
    print(f"{graph.num_nodes} nodes, {len(graph.adj_idx)} directed edges, "
          f"{graph.num_features} features; {cfg}", flush=True)
    start = time.time()
    model = train(graph, cfg, log=lambda m: print(m, flush=True))
    peak = (f", peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
            if torch.device(cfg.device).type == "cuda" else "")
    print(f"trained in {time.time() - start:.0f}s{peak}", flush=True)
    if not args.no_export:
        export(model, graph, cfg, args.out)


if __name__ == "__main__":
    main()
