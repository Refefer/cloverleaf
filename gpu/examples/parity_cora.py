"""Checks that GPU EP learns as well as cloverleaf's CPU EP: features-only EP (word embeddings,
every node the mean of its words) on Cora, the same settings on both, scored with the EP paper's
Planetoid protocol from examples/ep_paper/reproduce.py.

    python parity_cora.py --data DIR [--passes 200] [--runs 10] [--skip-cpu]

DIR holds cora/ as reproduce.py expects.  Needs cloverleaf and torch.
"""
import argparse
import importlib.util
import os
import tempfile
import time

import numpy as np

import cloverleaf as cl

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "reproduce", os.path.join(HERE, "..", "..", "examples", "ep_paper", "reproduce.py"))
assert _spec is not None and _spec.loader is not None
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

import sys
sys.path.insert(0, os.path.join(HERE, ".."))
from ep_gpu import data  # noqa: E402
from ep_gpu.train import Config, node_embeddings, train  # noqa: E402


def write_files(ds, tmp):
    edges, feats = os.path.join(tmp, "cora.edges"), os.path.join(tmp, "cora.features")
    with open(edges, "w") as f:
        for a, b in ds.edges:
            f.write(f"n\t{a}\tn\t{b}\t1\n")
    with open(feats, "w") as f:
        for v, words in enumerate(ds.words):
            if words:
                f.write(f"n\t{v}\t{' '.join(f'w{w}' for w in words)}\n")
    return edges, feats


def as_matrix(ds, keys, emb):
    H = np.zeros((ds.num_nodes, emb.shape[1]), np.float32)
    for (_, name), e in zip(keys, emb):
        H[int(name)] = e
    return H


def score(ds, H, runs):
    accs = []
    for run in range(runs):
        tr, va, te = rp.planetoid_split(ds.labels, np.random.default_rng([2017, run]))
        C = max(rp.C_GRID, key=lambda c: rp.accuracy(H[tr], ds.labels[tr], H[va], ds.labels[va], c))
        accs.append(rp.accuracy(H[tr], ds.labels[tr], H[te], ds.labels[te], C))
    return np.mean(accs), np.std(accs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--passes", type=int, default=200)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=20.0)
    p.add_argument("--skip-cpu", action="store_true")
    args = p.parse_args()
    ds = rp.load("cora", args.data)
    with tempfile.TemporaryDirectory() as tmp:
        edges, feats = write_files(ds, tmp)

        graph = data.load(edges, feats, namespace="feat")
        cfg = Config(dims=128, passes=args.passes, batch_size=64, lr=0.001, lr_schedule="constant",
                     init="glorot", loss="margin", gamma=args.gamma, negatives=1)
        start = time.time()
        model = train(graph, cfg, log=lambda m: None)
        gpu_time = time.time() - start
        H = as_matrix(ds, zip(graph.node_types, graph.node_names), node_embeddings(model, graph).cpu().numpy())
        print(f"GPU ({cfg.device}): {gpu_time:.0f}s, accuracy %.2f ± %.2f" % score(ds, H, args.runs), flush=True)

        if not args.skip_cpu:
            g = cl.Graph.load(edges, cl.EdgeType.Undirected)
            fs = cl.FeatureSet.new_from_graph(g)
            fs.load_into(feats, cl.FeatureNamespace.single("feat"))
            ep = cl.EmbeddingPropagator(alpha=0.001, loss=cl.EPLoss.margin(args.gamma, 1), batch_size=64,
                                        dims=128, passes=args.passes, valid_pct=0.0, indicator=False,
                                        lr_schedule="constant", init="glorot", seed=2017)
            start = time.time()
            fe = ep.learn_features(g, fs)
            cpu_time = time.time() - start
            ne = cl.NodeEmbedder(cl.FeatureAggregator.Averaged()).embed_feature_set(fs, fe)
            keys = list(g.vocab())
            H = as_matrix(ds, keys, np.asarray([ne.get_embedding(k) for k in keys], np.float32))
            print(f"CPU cloverleaf: {cpu_time:.0f}s, accuracy %.2f ± %.2f" % score(ds, H, args.runs), flush=True)


if __name__ == "__main__":
    main()
