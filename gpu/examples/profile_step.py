"""Profiles a few GPU EP training steps and prints where the time goes.

    python profile_step.py --edges GRAPH [--features FEATURES] [--graph-cache PATH] \
        [--steps 5] [any ep_gpu.train Config option, e.g. --pooling attention --batch-size 1024]
"""
import argparse
import os
import sys
from dataclasses import asdict

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ep_gpu import data  # noqa: E402
from ep_gpu.model import EPModel  # noqa: E402
from ep_gpu.train import Config, batch_loss, set_precision  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--edges", required=True)
    p.add_argument("--features", default=None)
    p.add_argument("--graph-cache", default=None)
    p.add_argument("--namespace", default="node_type")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--rows", type=int, default=25)
    for name, value in asdict(Config()).items():
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
    cfg = Config(**{k: getattr(args, k) for k in asdict(Config())})
    graph = (data.load_cached(args.graph_cache, args.edges, args.features, namespace=args.namespace)
             if args.graph_cache else data.load(args.edges, args.features, namespace=args.namespace))
    device = torch.device(cfg.device)
    set_precision(cfg, device)
    g = graph.to(device)
    model = EPModel(g.num_features, cfg.dims, cfg.pooling, cfg.heads, cfg.init, cfg.seed).to(device)
    sparse_opt = torch.optim.SparseAdam(list(model.features.parameters()), lr=cfg.lr)
    dense = model.dense_parameters()
    dense_opt = torch.optim.Adam(dense, lr=cfg.dense_lr) if dense else None
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    def step():
        nodes = torch.randint(0, g.num_nodes, (cfg.batch_size,), device=device, generator=gen)
        sparse_opt.zero_grad()
        if dense_opt is not None:
            dense_opt.zero_grad()
        loss = batch_loss(model, g, nodes, cfg, gen)
        loss.backward()
        sparse_opt.step()
        if dense_opt is not None:
            dense_opt.step()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with profile(activities=activities) as prof:
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()
    sort = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort, row_limit=args.rows))
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB" if device.type == "cuda" else "")


if __name__ == "__main__":
    main()
