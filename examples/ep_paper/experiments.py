"""Experiments with EP beyond the paper's EP-B, on the paper's datasets and protocol.

Uses reproduce.py's data loading, splits and evaluation (see its docstring), and adds:

  Losses (--loss, --negatives): EP-B's Euclidean margin loss, StarSpace (a cosine margin) or
  RankSpace (StarSpace plus a softmax rank loss over the positive and its negatives).  --gammas
  sets the loss's threshold: the margin for margin and starspace, and for rankspace both
  StarSpace's margin and RankLoss's probability threshold.

  Features only (--features-only; Cora, Citeseer, Pubmed): EP learns word embeddings and no node
  ids, and every node, test nodes included, is embedded as the mean of its own words.  In the
  inductive setting this embeds a test node without using its edges, as for a node that has none
  yet.

  Neighbourhood alignment (--align W or adaptive): after training, each node's embedding becomes
  W * itself + (1 - W) * its neighbours' mean (cloverleaf's NeighborhoodAligner), or uses
  cloverleaf's degree-based weight.  Alignment uses the full graph, which in the inductive setting
  includes the test nodes and their edges, added back by the time they are classified.

  Edge-free alignment (--features-only --setting inductive --align): test nodes have no edges, so
  only the training graph is aligned.  --new-nodes then leaves the test nodes unaligned (raw), or
  estimates where alignment would have moved them from their nearest training nodes:
  cloverleaf's EmbeddingAligner (aligner), which matches distances to those nodes, or their mean
  shift (shift).  A probe of training nodes held out of the anchors checks the estimate: their
  aligned embeddings are known, so the log reports how much of their alignment it recovers.

    python experiments.py --data DIR --datasets cora --setting inductive --features-only
    python experiments.py --data DIR --datasets cora --setting inductive --features-only \\
        --align 0.3 --new-nodes shift
    python experiments.py --data DIR --datasets cora --loss rankspace --negatives 10 --gammas 0.5
"""
import argparse
import importlib.util
import os
import tempfile
import time

import numpy as np

import cloverleaf as cl

# reproduce.py sits next to this file; loaded by path so this works under `python -I`, which
# leaves the script's directory off sys.path.
_spec = importlib.util.spec_from_file_location(
    "reproduce", os.path.join(os.path.dirname(os.path.abspath(__file__)), "reproduce.py"))
assert _spec is not None and _spec.loader is not None
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

log = rp.log

# The paper's bag-of-words baseline (Table 5, "BoW Feat"): logistic regression on the binary word
# vectors.  It uses no graph, so it is the same in both settings, and is the reference for
# --features-only, whose test nodes see no graph either.
PAPER_BOW = {"cora": 58.63, "citeseer": 58.07, "pubmed": 70.49}


def ep_loss(args, gamma):
    make = {"margin": cl.EPLoss.margin, "starspace": cl.EPLoss.starspace,
            "rankspace": cl.EPLoss.rankspace}[args.loss]
    return make(float(gamma), args.negatives)


# Neighbourhood alignment

_FULL_GRAPHS = {}


def align(ds, H, alignment, graph=None, edges=None):
    """Neighbourhood alignment over `graph`, built from `edges`; pass both or neither, for the
    full graph.  Nodes without edges are left as they are."""
    if graph is None or edges is None:
        assert graph is None and edges is None, "pass both graph and edges, or neither"
        if ds.name not in _FULL_GRAPHS:
            with tempfile.TemporaryDirectory() as tmp:
                _FULL_GRAPHS[ds.name] = rp.load_graph(ds, ds.edges, tmp)
        graph, edges = _FULL_GRAPHS[ds.name], ds.edges
    in_graph = np.zeros(ds.num_nodes, bool)
    in_graph[edges.ravel()] = True
    nodes = np.flatnonzero(in_graph)
    embs = cl.NodeEmbeddings(graph, H.shape[1], cl.Distance.Euclidean)
    for v in nodes:
        embs.set_embedding(("n", str(v)), H[v].tolist())
    weight = None if alignment == "adaptive" else alignment
    aligned = cl.NeighborhoodAligner(weight).align(embs, graph)
    out = H.copy()
    for v in nodes:
        out[v] = aligned.get_embedding(("n", str(v)))
    return out


def map_into_aligned(H, aligned, anchors, targets, args):
    """Estimates where `targets`, which have no edges, would land after alignment, from how
    alignment moved their nearest `anchors` (found among the anchors' unaligned embeddings)."""
    def node_embeddings(M):
        return cl.NodeEmbeddings.new_from_list(
            [(("n", str(v)), M[v].tolist()) for v in anchors], cl.Distance.Euclidean)
    pre, post = node_embeddings(H), node_embeddings(aligned)
    ann = cl.EmbAnn(pre, 10, 20, seed=args.seed)
    queries = [cl.Query.embedding(H[v].tolist()) for v in targets]
    if args.new_nodes == "aligner":
        aligner = cl.EmbeddingAligner(args.map_neighbours, 0, args.map_lr, 1e-2, args.map_iters)
        return np.asarray(aligner.bulk_align(pre, ann, post, queries, seed=args.seed), np.float32)
    out = np.empty((len(targets), H.shape[1]), np.float32)
    for i, (v, q) in enumerate(zip(targets, queries)):
        near = [int(name) for (_, name), _ in ann.find(pre, q, args.map_neighbours)]
        out[i] = H[v] + (aligned[near] - H[near]).mean(axis=0)
    return out


def align_without_new_edges(ds, H, removed, graph, edges, args):
    """Aligns the training graph only, then leaves the removed nodes as they are or maps them
    into the aligned space (--new-nodes), checking the mapping on a probe of training nodes."""
    aligned = align(ds, H, args.align, graph, edges)
    if args.new_nodes == "raw":
        return aligned
    in_graph = np.zeros(ds.num_nodes, bool)
    in_graph[edges.ravel()] = True
    candidates = np.flatnonzero(in_graph)
    rng = np.random.default_rng([args.seed, int(np.sum(removed)), 5])
    probe = rng.choice(candidates, min(200, len(candidates) // 10), replace=False)
    anchors = np.setdiff1d(candidates, probe)
    if args.anchors is not None and args.anchors < len(anchors):
        anchors = np.sort(rng.choice(anchors, args.anchors, replace=False))
    targets = np.concatenate([removed, probe])
    mapped = map_into_aligned(H, aligned, anchors, targets, args)
    moved = np.linalg.norm(H[probe] - aligned[probe], axis=1).mean()
    missed = np.linalg.norm(mapped[len(removed):] - aligned[probe], axis=1).mean()
    log(f"    {args.new_nodes} mapping with {len(anchors)} anchors recovers "
        f"{100 * (1 - missed / moved):.0f}% of the alignment on {len(probe)} probe nodes")
    aligned[removed] = mapped[:len(removed)]
    return aligned


# Embedding, with reproduce.embed's signature so reproduce.run can use it


def embed(ds, graph, edges, gamma, seed, args, removed=None, split_tag=None):
    path = None
    if args.cache:
        tag = rp.cache_tag(args)
        if args.features_only:
            tag += "-features_only"
        if (args.loss, args.negatives) != ("margin", 1):
            tag += f"-loss={args.loss}-negatives={args.negatives}"
        if split_tag is not None:
            tag += f"-inductive-{split_tag}"
        path = os.path.join(args.cache, f"{ds.name}-gamma={gamma:g}-{tag}.npy")
    start = time.time()
    if path and os.path.exists(path):
        H = np.load(path)
    else:
        types = ["word"] if args.features_only else ["id"] + (["word"] if ds.words else [])
        # With features only, removed nodes are embedded from their own words like every other
        # node rather than reconstructed from their neighbours.
        reconstruct = None if args.features_only else removed
        H = np.concatenate([rp.train_label_type(ds, graph, edges, t, gamma, seed, args,
                                                reconstruct, ep_loss(args, gamma))
                            for t in types], axis=1)
        if args.features_only and removed is not None:
            unknown = sum(not np.any(H[v]) for v in removed)
            if unknown:
                log(f"    {unknown} of {len(removed)} removed nodes have no trained words; they "
                    f"get a zero embedding")
        if path:
            np.save(path, H)
    seconds = time.time() - start
    # Alignment is applied after the cache, so every alignment reuses the same EP training.
    if args.align is not None:
        if args.features_only and removed is not None:
            H = align_without_new_edges(ds, H, removed, graph, edges, args)
        else:
            H = align(ds, H, args.align)
    return H, seconds


def alignment_arg(value):
    """--align takes a weight in [0, 1] on the node's own embedding, or "adaptive"."""
    if value == "adaptive":
        return value
    weight = float(value)
    if not 0.0 <= weight <= 1.0:
        raise argparse.ArgumentTypeError(f"alignment must be in [0, 1] or 'adaptive', got {value}")
    return weight


def main():
    p = rp.build_parser((__doc__ or "").split("\n")[0])
    p.add_argument("--loss", default="margin", choices=["margin", "starspace", "rankspace"],
                   help="EP's loss; --gammas sets its threshold.")
    p.add_argument("--negatives", type=int, default=1, help="Negatives per node (paper: 1).")
    p.add_argument("--features-only", action="store_true",
                   help="Learn only word embeddings and embed every node as the mean of its own "
                        "words.  Cora, Citeseer and Pubmed only.")
    p.add_argument("--align", type=alignment_arg, default=None,
                   help="Neighbourhood-align the node embeddings before classifying them: the "
                        "weight in [0, 1] on a node's own embedding, or 'adaptive'.")
    p.add_argument("--new-nodes", default="raw", choices=["raw", "aligner", "shift"],
                   help="With --features-only --setting inductive --align: leave the edgeless "
                        "test nodes unaligned, or map them into the aligned space with "
                        "EmbeddingAligner or their nearest training nodes' mean shift.")
    p.add_argument("--map-neighbours", type=int, default=10,
                   help="Nearest training nodes a test node is mapped from.")
    p.add_argument("--map-lr", type=float, default=0.1, help="EmbeddingAligner's learning rate.")
    p.add_argument("--map-iters", type=int, default=10, help="EmbeddingAligner's iterations.")
    p.add_argument("--anchors", type=int, default=None,
                   help="Sample this many training nodes as mapping anchors (default: all).")
    args = p.parse_args()
    if args.features_only:
        featureless = [d for d in args.datasets if d in rp.MULTI_LABEL]
        if featureless:
            p.error(f"--features-only needs node features; {', '.join(featureless)} "
                    f"{'has' if len(featureless) == 1 else 'have'} none")
    if args.new_nodes != "raw" and not (
            args.features_only and args.setting == "inductive" and args.align is not None):
        p.error("--new-nodes maps edgeless test nodes into the aligned space, so it needs "
                "--features-only, --setting inductive and --align")
    rp.run(args, embed_fn=embed)


if __name__ == "__main__":
    main()
