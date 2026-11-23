"""Graph and feature loading, in cloverleaf's file formats, into flat GPU-friendly arrays.

Graph file: one edge per line, `from_type<TAB>from_name<TAB>to_type<TAB>to_name<TAB>weight`.
Feature file: one node per line, `node_type<TAB>name<TAB>f1 f2 ...`.

As in cloverleaf, nodes are numbered in the order they first appear in the graph file, features
are namespaced (by node type, or a single namespace), feature lines for nodes outside the graph
are ignored, and a node without features gets one of its own, ("node", "<node id>").
Optionally every token also gets a feature in one namespace shared across node types.
"""
from dataclasses import dataclass, fields

import numpy as np
import torch


@dataclass
class Graph:
    """Nodes, adjacency in CSR form, and each node's features in CSR form.  All index arrays
    are int64 tensors; `to(device)` moves them."""
    node_types: list           # node id -> type
    node_names: list           # node id -> name
    adj_ptr: torch.Tensor      # (N + 1,) neighbours of v are adj_idx[adj_ptr[v]:adj_ptr[v + 1]]
    adj_idx: torch.Tensor      # (E,)
    adj_w: torch.Tensor        # (E,) edge weights
    feat_ptr: torch.Tensor     # (N + 1,) features of v are feat_idx[feat_ptr[v]:feat_ptr[v + 1]]
    feat_idx: torch.Tensor     # (M,) feature ids
    feature_keys: list         # feature id -> (namespace, name)
    zero_init: torch.Tensor = None  # feature ids whose rows start at zero (offsets), or None

    @property
    def num_nodes(self):
        return len(self.node_names)

    @property
    def num_features(self):
        return len(self.feature_keys)

    def degrees(self):
        return self.adj_ptr[1:] - self.adj_ptr[:-1]

    def to(self, device):
        moved = {}
        for f in fields(self):
            value = getattr(self, f.name)
            moved[f.name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return Graph(**moved)

    def node_id(self, node_type, name):
        if not hasattr(self, "_node_index"):
            self._node_index = {(t, n): i for i, (t, n) in
                                enumerate(zip(self.node_types, self.node_names))}
        return self._node_index[(node_type, name)]

    def feature_index(self):
        """(namespace, name) -> feature id."""
        return {k: i for i, k in enumerate(self.feature_keys)}


def _csr(n, rows, cols, vals):
    order = np.lexsort((cols, rows))
    rows, cols, vals = rows[order], cols[order], vals[order]
    ptr = np.zeros(n + 1, np.int64)
    np.add.at(ptr, rows + 1, 1)
    return np.cumsum(ptr), cols, vals


def load(edges_path, features_path=None, directed=False, namespace="node_type", prune_min_count=0,
         shared_namespace=None):
    """Reads a cloverleaf graph file and, optionally, a feature file.

    namespace: "node_type" puts each feature in its node's type namespace (cloverleaf's
    FeatureNamespace.node_type()); any other string is a single namespace for every feature
    (FeatureNamespace.single(namespace)).
    prune_min_count: drops features that fewer than this many of the graph's nodes have, as
    cloverleaf's FeatureSet.prune_min_count does; a node left with none gets its own feature.
    shared_namespace: with namespace "node_type", also gives every token a feature in this one
    namespace, shared across node types.  The node-type features then start at zero
    (Graph.zero_init), so each is a learned offset from its token's shared feature: a query word
    and the same word in a title start out identical and drift apart only as training needs.
    Pruning counts the two separately, so a token on both sides can keep its shared feature
    and lose an offset.
    """
    if shared_namespace is not None and namespace != "node_type":
        raise ValueError("shared_namespace needs namespace='node_type'")
    if isinstance(prune_min_count, bool) or not isinstance(prune_min_count, int) or prune_min_count < 0:
        raise ValueError(f"prune_min_count must be an int >= 0, got {prune_min_count!r}")
    index, types, names = {}, [], []

    def node(t, n):
        key = (t, n)
        if key not in index:
            index[key] = len(names)
            types.append(t)
            names.append(n)
        return index[key]

    src, dst, wts = [], [], []
    with open(edges_path) as f:
        for i, line in enumerate(f):
            pieces = line.rstrip("\n").split("\t")
            if len(pieces) != 5:
                raise ValueError(f"{edges_path}:{i + 1}: expected 5 tab-separated fields")
            a = node(pieces[0], pieces[1])
            b = node(pieces[2], pieces[3])
            src.append(a)
            dst.append(b)
            wts.append(float(pieces[4]))
    n = len(names)
    src, dst, wts = np.asarray(src, np.int64), np.asarray(dst, np.int64), np.asarray(wts, np.float32)
    if not directed:
        src, dst, wts = (np.concatenate([src, dst]), np.concatenate([dst, src]),
                         np.concatenate([wts, wts]))
    adj_ptr, adj_idx, adj_w = _csr(n, src, dst, wts)

    feature_ids, feature_keys = {}, []
    node_feats = [[] for _ in range(n)]

    def feature(key):
        if key not in feature_ids:
            feature_ids[key] = len(feature_keys)
            feature_keys.append(key)
        return feature_ids[key]

    if features_path is not None:
        with open(features_path) as f:
            for line in f:
                pieces = line.rstrip("\n").split("\t")
                if len(pieces) < 3:
                    continue
                v = index.get((pieces[0], pieces[1]))
                if v is None:
                    continue
                ns = pieces[0] if namespace == "node_type" else namespace
                tokens = pieces[2].split()
                keys = [(ns, tok) for tok in tokens]
                if shared_namespace is not None:
                    keys += [(shared_namespace, tok) for tok in tokens]
                node_feats[v] = sorted({feature(k) for k in keys})
    if prune_min_count > 1 and feature_keys:
        counts = np.bincount(np.fromiter((f for fs in node_feats for f in fs), np.int64),
                             minlength=len(feature_keys))
        kept = np.flatnonzero(counts >= prune_min_count)
        remap = np.full(len(feature_keys), -1, np.int64)
        remap[kept] = np.arange(len(kept))
        node_feats = [[int(remap[f]) for f in fs if remap[f] >= 0] for fs in node_feats]
        feature_keys = [feature_keys[i] for i in kept]
        feature_ids = {k: i for i, k in enumerate(feature_keys)}
    fallbacks = set()
    for v in range(n):
        if not node_feats[v]:
            node_feats[v] = [feature(("node", str(v)))]
            fallbacks.add(node_feats[v][0])
    if shared_namespace is not None and (shared_namespace in set(types) or shared_namespace == "node"):
        raise ValueError(f"shared_namespace {shared_namespace!r} collides with a node type's namespace")
    zero_init = None
    if shared_namespace is not None:
        node_types = set(types)
        zero_init = torch.tensor([i for i, (ns, _) in enumerate(feature_keys)
                                  if ns in node_types and i not in fallbacks], dtype=torch.int64)
    lengths = np.fromiter((len(fs) for fs in node_feats), np.int64, n)
    feat_ptr = np.concatenate([[0], np.cumsum(lengths)])
    feat_idx = np.fromiter((f for fs in node_feats for f in fs), np.int64, int(feat_ptr[-1]))

    # Node and feature ids fit in int32 below 2^31, halving the largest arrays.  Offsets index
    # edges and node-feature pairs, which can pass 2^31 on huge graphs, so they stay int64.
    def ids(a, count):
        return torch.from_numpy(a.astype(np.int32 if count < 2**31 else np.int64))
    t = torch.from_numpy
    return Graph(types, names, t(adj_ptr), ids(adj_idx, n), t(adj_w), t(feat_ptr),
                 ids(feat_idx, len(feature_keys)),
                 feature_keys, zero_init)


def load_cached(cache_path, edges_path, features_path=None, directed=False, namespace="node_type",
                prune_min_count=0, shared_namespace=None):
    """`load`, saving the parsed graph to `cache_path` the first time and reading it back after.
    The cache records its inputs and settings, and is rebuilt if they change."""
    import os
    key = {"edges": os.path.abspath(edges_path), "edges_mtime": os.path.getmtime(edges_path),
           "features": features_path and os.path.abspath(features_path),
           "features_mtime": features_path and os.path.getmtime(features_path),
           "directed": directed, "namespace": namespace, "prune_min_count": prune_min_count,
           "shared_namespace": shared_namespace, "format": 3}
    if os.path.exists(cache_path):
        saved = torch.load(cache_path, weights_only=False)
        if saved["key"] == key:
            return saved["graph"]
    graph = load(edges_path, features_path, directed, namespace, prune_min_count, shared_namespace)
    torch.save({"key": key, "graph": graph}, cache_path)
    return graph


def save_embeddings(path, keys, embeddings):
    """Writes `type<TAB>name<TAB>[e1,e2,...]` lines, which cloverleaf's NodeEmbeddings.load reads.
    Uses cloverleaf's parallel writer when it's installed, and a slow pure-Python one otherwise."""
    embeddings = np.ascontiguousarray(embeddings.detach().float().cpu().numpy())
    try:
        from cloverleaf import write_embeddings
    except ImportError:
        write_embeddings = None
    if write_embeddings is not None:
        write_embeddings(path, [(str(t), str(n)) for t, n in keys], embeddings)
        return
    with open(path, "w") as f:
        for (t, n), e in zip(keys, embeddings):
            f.write(f"{t}\t{n}\t[{','.join(repr(float(x)) for x in e)}]\n")
