"""Graph and feature loading into flat GPU-friendly arrays, done by cloverleaf.

Graph file: one edge per line, `from_type<TAB>from_name<TAB>to_type<TAB>to_name<TAB>weight`.
Feature file: one node per line, `node_type<TAB>name<TAB>f1 f2 ...`.

cloverleaf reads both (`Graph.load`, `FeatureSet.load_into`), so nodes, features and their ids are
exactly cloverleaf's: nodes numbered in the order they first appear in the graph file, features
namespaced by node type (or a single namespace, or both a node type's and a shared one), feature
lines for nodes outside the graph ignored, and a node without features given one of its own,
("node", "<node id>").  Each node's repeated tokens count once.
"""
import dataclasses
from dataclasses import dataclass, fields

import numpy as np
import torch

import cloverleaf as cl


@dataclass
class Graph:
    """Nodes, adjacency in CSR form, and each node's features in CSR form.  `to(device)` moves
    the index arrays."""
    node_types: list           # node id -> type
    node_names: list           # node id -> name
    adj_ptr: torch.Tensor      # (N + 1,) neighbours of v are adj_idx[adj_ptr[v]:adj_ptr[v + 1]]
    adj_idx: torch.Tensor      # (E,)
    adj_w: torch.Tensor        # (E,) edge weights: each node's transition probabilities
    feat_ptr: torch.Tensor     # (N + 1,) features of v are feat_idx[feat_ptr[v]:feat_ptr[v + 1]]
    feat_idx: torch.Tensor     # (M,) feature ids
    feature_keys: list         # feature id -> (namespace, name)
    zero_init: torch.Tensor = None  # feature ids whose rows start at zero (offsets), or None
    # The cloverleaf graph it was loaded from (for the PPR lists), when it was loaded this run.
    cloverleaf: object = dataclasses.field(default=None, repr=False, compare=False)
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


def _namespace(namespace, shared_namespace):
    if shared_namespace is not None:
        if namespace != "node_type":
            raise ValueError("shared_namespace needs namespace='node_type'")
        return cl.FeatureNamespace.node_type_and_shared(shared_namespace)
    return cl.FeatureNamespace.node_type() if namespace == "node_type" else cl.FeatureNamespace.single(namespace)


def _array(buffer, dtype):
    return np.frombuffer(buffer, dtype=dtype)


def load(edges_path, features_path=None, directed=False, namespace="node_type", prune_min_count=0,
         shared_namespace=None):
    """Reads a cloverleaf graph file and, optionally, a feature file, with cloverleaf.

    namespace: "node_type" puts each feature in its node's type namespace (cloverleaf's
    FeatureNamespace.node_type()); any other string is a single namespace for every feature
    (FeatureNamespace.single(namespace)).
    prune_min_count: drops features that fewer than this many of the graph's nodes have, as
    cloverleaf's FeatureSet.prune_min_count does; a node left with none gets its own feature.
    shared_namespace: with namespace "node_type", also gives every token a feature in this one
    namespace, shared across node types (FeatureNamespace.node_type_and_shared).  The node-type
    features then start at zero (Graph.zero_init), so each is a learned offset from its token's
    shared feature: a query word and the same word in a title start out identical and drift
    apart only as training needs.  Pruning counts the two separately, so a token on both sides
    can keep its shared feature and lose an offset.
    """
    if isinstance(prune_min_count, bool) or not isinstance(prune_min_count, int) or prune_min_count < 0:
        raise ValueError(f"prune_min_count must be an int >= 0, got {prune_min_count!r}")
    ns = _namespace(namespace, shared_namespace)
    edge_type = cl.EdgeType.Directed if directed else cl.EdgeType.Undirected
    graph = cl.Graph.load(edges_path, edge_type, chunk_size=100_000)
    features = cl.FeatureSet.new_from_graph(graph)
    if features_path is not None:
        features.load_into(features_path, ns, dedup=True)
    if prune_min_count > 1:
        features = features.prune_min_count(prune_min_count)
    # The nodes left without features get ("node", "<id>") features of their own: never offsets.
    before_ptr = _array(features.csr()[0], np.int64)
    featureless = np.flatnonzero(np.diff(before_ptr) == 0)
    features.fill_missing_nodes()

    n = graph.nodes()
    adj_ptr, adj_idx, adj_w = (_array(b, t) for b, t in zip(graph.csr(), (np.int64, np.int64, np.float32)))
    feat_ptr, feat_idx = (_array(b, np.int64) for b in features.csr())
    feature_keys = features.feature_keys()
    type_table, type_ids = graph.node_type_ids()
    node_types = [type_table[i] for i in _array(type_ids, np.int64).tolist()]
    zero_init = None
    if shared_namespace is not None:
        fallbacks = {("node", str(v)) for v in featureless.tolist()}
        type_set = set(type_table)
        zero_init = torch.tensor([i for i, k in enumerate(feature_keys) if k[0] in type_set and k not in fallbacks],
                                 dtype=torch.int64)

    # Node and feature ids fit in int32 below 2^31, halving the largest arrays.  Offsets index
    # edges and node-feature pairs, which can pass 2^31 on huge graphs, so they stay int64.
    def ids(a, count):
        return torch.from_numpy(a.astype(np.int32 if count < 2**31 else np.int64))
    return Graph(node_types, graph.node_names(), torch.from_numpy(adj_ptr.copy()), ids(adj_idx, n),
                 torch.from_numpy(adj_w.copy()), torch.from_numpy(feat_ptr.copy()),
                 ids(feat_idx, len(feature_keys)), feature_keys, zero_init, graph)


def load_cached(cache_path, edges_path, features_path=None, directed=False, namespace="node_type",
                prune_min_count=0, shared_namespace=None):
    """`load`, saving the parsed graph to `cache_path` the first time and reading it back after.
    The cache records its inputs and settings, and is rebuilt if they change."""
    import os
    key = {"edges": os.path.abspath(edges_path), "edges_mtime": os.path.getmtime(edges_path),
           "features": features_path and os.path.abspath(features_path),
           "features_mtime": features_path and os.path.getmtime(features_path),
           "directed": directed, "namespace": namespace, "prune_min_count": prune_min_count,
           "shared_namespace": shared_namespace, "format": 4}
    if os.path.exists(cache_path):
        saved = torch.load(cache_path, weights_only=False)
        if saved["key"] == key:
            return saved["graph"]
    graph = load(edges_path, features_path, directed, namespace, prune_min_count, shared_namespace)
    torch.save({"key": key, "graph": dataclasses.replace(graph, cloverleaf=None)}, cache_path)
    return graph


def save_aggregator(path, model):
    """Writes the model's pooling as a cloverleaf FeatureAggregator file, which
    `cloverleaf.FeatureAggregator.load` reads: Averaged for mean pooling, or Attention with the
    query/key/value and output projections (PyTorch Linear weights).  A nested reconstruction's
    neighbour pooling is only used in training and isn't written."""
    from .model import AttentionPool
    pool = model.pool
    if not isinstance(pool, AttentionPool):
        cl.FeatureAggregator.Averaged().save(path)
        return
    qkv = pool.qkv.weight.detach().float().cpu()
    out = pool.out.weight.detach().float().cpu()
    cl.FeatureAggregator.Attention(out.shape[0], pool.heads, pool.head_dims,
                                   qkv.flatten().tolist(), out.flatten().tolist()).save(path)


def save_embeddings(path, keys, embeddings):
    """Writes `type<TAB>name<TAB>[e1,e2,...]` lines, which cloverleaf's NodeEmbeddings.load reads,
    with cloverleaf's parallel writer.  keys: (type, name) per row, or a cloverleaf Graph or
    FeatureSet whose ids the rows follow."""
    embeddings = np.ascontiguousarray(embeddings.detach().float().cpu().numpy())
    if isinstance(keys, list):
        keys = [(str(t), str(n)) for t, n in keys]
    cl.write_embeddings(path, keys, embeddings)
