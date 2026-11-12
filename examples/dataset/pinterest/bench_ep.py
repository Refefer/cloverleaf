"""Benchmark: Embedding Propagation on the Pinterest leave-one-out split.

Same splits, protocol and metrics as bench_rp3b.py (see there): leave-one-out with sampled negatives,
seen items ranked last, ties broken at random.  A board is scored by the cosine similarity of the
user and board node embeddings, optionally after neighborhood alignment.

The dataset has no content features, so --features picks features derived from the graph:

  ids              Each node's only feature is its own id: EP learns one free embedding per node.
  user-boards      A user is the bag of boards they pinned; a board is its own id.  EP learns only
                   board embeddings and builds users from them, like an item-based model.
  user-boards+id   As user-boards, plus the user's own id.

Features come from the edges the model is trained on, so validation users never see their
held-out board.

Each configuration is trained on a validation split carved out of the training edges, and the
alignment strength is picked on validation HR@5.  The configuration is then retrained on the full
training edges and evaluated on test.  Pass --no-tune to skip the validation run and evaluate the
given alignment directly.

Usage (from this directory, after `maturin develop --release` and `pip install numpy scipy`):

    python bench_ep.py                              # the run.sh recipe
    python bench_ep.py --configs margin adamw       # several configurations
    python bench_ep.py --features ids user-boards   # several feature modes
    python bench_ep.py --split ncf --ncf-dir DIR
"""
import argparse
import itertools
import json
import os
import tempfile
import time

import numpy as np

import cloverleaf as cl
from bench_rp3b import (METRICS, PUBLISHED, load_ncf, load_test, load_train, log, metrics,
                        ranks, seen_mask, validation_split, write_graph)

HERE = os.path.dirname(os.path.abspath(__file__))

# Embedding Propagation settings.  "margin" is the recipe in run.sh.
CONFIGS = {
    "margin": dict(loss=("margin", 10.0, 10)),
    "adamw": dict(loss=("margin", 10.0, 10), optimizer=("adamw", 0.01)),
    "starspace": dict(loss=("starspace", 10.0, 10)),
    "hard-negs": dict(loss=("margin", 10.0, 10), hard_negatives=2),
}
BASE = dict(dims=128, passes=200, alpha=0.9, batch_size=1024, max_nodes=10)

ALIGNMENTS = [None, 0.1, 0.3, 0.5, 0.7]

FEATURE_MODES = ["ids", "user-boards", "user-boards+id"]


def make_loss(spec):
    kind, *params = spec
    return getattr(cl.EPLoss, kind)(*params)


def make_optimizer(spec):
    if spec is None:
        return None
    kind, *params = spec
    return getattr(cl.Optimizer, kind)(*params)


def build_features(graph, R, mode):
    """Feature set for the graph; see the module docstring for the modes."""
    features = cl.FeatureSet.new_from_graph(graph)
    if mode == "ids":
        # Nodes without features get their own id when EP fills in missing nodes.
        return features
    for b in np.flatnonzero(np.diff(R.tocsc().indptr)):
        features.set_features(("board", str(b)), [("board", str(b))])
    for u in range(R.shape[0]):
        boards = R.indices[R.indptr[u]:R.indptr[u + 1]]
        if len(boards) == 0:
            continue
        feats = [("board", str(b)) for b in boards]
        if mode == "user-boards+id":
            feats.append(("user", str(u)))
        features.set_features(("user", str(u)), feats)
    return features


def train(R, config, feature_mode, seed, tmp):
    """Trains EP on the edges in R and returns the graph and the node embeddings."""
    path = os.path.join(tmp, "train.edges")
    write_graph(R, path)
    graph = cl.Graph.load(path, cl.EdgeType.Undirected)
    features = build_features(graph, R, feature_mode)
    kwargs = dict(config)
    kwargs["loss"] = make_loss(kwargs["loss"])
    kwargs["optimizer"] = make_optimizer(kwargs.get("optimizer"))
    ep = cl.EmbeddingPropagator(valid_pct=0.0, seed=seed, indicator=False, **kwargs)
    start = time.time()
    feature_embeddings = ep.learn_features(graph, features)
    seconds = time.time() - start
    embedder = cl.NodeEmbedder(cl.FeatureAggregator.Averaged())
    return graph, embedder.embed_feature_set(features, feature_embeddings), seconds


def as_matrices(node_embeddings, shape):
    """User and board embedding matrices, L2-normalised and indexed by id."""
    dims = node_embeddings.dims()
    users = np.zeros((shape[0], dims), np.float32)
    boards = np.zeros((shape[1], dims), np.float32)
    for i in range(len(node_embeddings)):
        (kind, name), emb = node_embeddings[i]
        (users if kind == "user" else boards)[int(name)] = emb
    for m in (users, boards):
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        m /= np.where(norms > 0, norms, 1)
    return users, boards


def evaluate(graph, node_embeddings, alignment, shape, users, cands, seen, seed):
    if alignment is not None:
        node_embeddings = cl.NeighborhoodAligner(alignment).align(node_embeddings, graph)
    U, B = as_matrices(node_embeddings, shape)
    scores = np.einsum("nd,ncd->nc", U[users], B[cands])
    return metrics(ranks(scores, np.random.default_rng(seed), seen))


def fmt_row(name, m):
    return f"{name:<34}" + "".join(f"{m[k]:>9.4f}" for k in METRICS)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--split", choices=["cmn", "ncf"], default="cmn")
    p.add_argument("--train", default=os.path.join(HERE, "pinterest.edges.gz"))
    p.add_argument("--test", default=os.path.join(HERE, "pinterest.test.gz"))
    p.add_argument("--ncf-dir", default=None)
    p.add_argument("--configs", nargs="+", default=["margin"], choices=list(CONFIGS))
    p.add_argument("--features", nargs="+", default=["ids"], choices=FEATURE_MODES)
    p.add_argument("--passes", type=int, default=BASE["passes"])
    p.add_argument("--dims", type=int, default=BASE["dims"])
    p.add_argument("--alignments", type=float, nargs="*", default=None,
                   help="Alignment strengths to try (default: none, 0.1, 0.3, 0.5, 0.7).")
    p.add_argument("--no-tune", action="store_true",
                   help="Skip the validation run; evaluate every alignment on test.")
    p.add_argument("--max-users", type=int, default=None)
    p.add_argument("--include-seen", action="store_true")
    p.add_argument("--seed", type=int, default=2019)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    alignments = ALIGNMENTS if args.alignments is None else [None] + args.alignments
    rng = np.random.default_rng(args.seed)
    log(f"loading the {args.split} split")
    if args.split == "ncf":
        if not args.ncf_dir:
            p.error("--split ncf needs --ncf-dir")
        R, test_users, test_cands = load_ncf(args.ncf_dir, False)
    else:
        R = load_train(args.train, False)
        test_users, test_cands = load_test(args.test)
    test_seen = seen_mask(R, test_users, test_cands)
    if args.max_users and args.max_users < len(test_users):
        keep = np.sort(rng.choice(len(test_users), args.max_users, replace=False))
        test_users, test_cands, test_seen = test_users[keep], test_cands[keep], test_seen[keep]
    if args.include_seen:
        test_seen = None
    R_val, val_users, val_cands = validation_split(
        R, test_users, test_cands, test_cands.shape[1] - 1, rng)
    val_seen = None if args.include_seen else seen_mask(R_val, val_users, val_cands)

    published = PUBLISHED[args.split]["RP3beta"]
    results = {"published RP3beta": {"params": {}, "test": published}}
    for config_name, feature_mode in itertools.product(args.configs, args.features):
        name = config_name if feature_mode == "ids" else f"{config_name}, {feature_mode}"
        config = dict(BASE, passes=args.passes, dims=args.dims, **CONFIGS[config_name])
        with tempfile.TemporaryDirectory() as tmp:
            if args.no_tune:
                candidates = alignments
            else:
                log(f"{name}: training on the validation split")
                graph, ne, secs = train(R_val, config, feature_mode, args.seed, tmp)
                log(f"{name}: trained in {secs:.0f}s")
                val = {}
                for a in alignments:
                    val[a] = evaluate(graph, ne, a, R.shape, val_users, val_cands, val_seen,
                                      args.seed)
                    log(f"  valid alignment={a}: "
                        + " ".join(f"{k}={v:.4f}" for k, v in val[a].items()))
                best = max(val, key=lambda a: (val[a]["HR@5"], val[a]["NDCG@5"]))
                candidates = [best]

            log(f"{name}: training on the full training edges")
            graph, ne, secs = train(R, config, feature_mode, args.seed, tmp)
            log(f"{name}: trained in {secs:.0f}s")
            for a in candidates:
                test = evaluate(graph, ne, a, R.shape, test_users, test_cands, test_seen,
                                args.seed)
                label = f"EP {name}" + (f" (align {a})" if a is not None else "")
                params = {k: v for k, v in config.items()}
                params.update(features=feature_mode, alignment=a, train_seconds=round(secs))
                results[label] = {"params": params, "test": test}
                if not args.no_tune:
                    results[label]["valid"] = val[a]

    print()
    print(f"{args.split + ' split, test':<34}" + "".join(f"{k:>9}" for k in METRICS))
    for name, r in results.items():
        print(fmt_row(name, r["test"]))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"split": args.split, "include_seen": args.include_seen,
                       "results": results}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
