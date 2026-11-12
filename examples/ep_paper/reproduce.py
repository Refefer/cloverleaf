"""Reproduce EP-B's node classification results with cloverleaf.

Paper: García-Durán and Niepert, "Learning Graph Representations with Embedding Propagation",
NIPS 2017 (arXiv 1710.03059).  EP-B is the paper's baseline instance, set up as in section 5.1:

  * one embedding per label (node ids, and words where the dataset has them), 128 dims each,
    Glorot uniform initialisation;
  * h(v) is the mean of v's label embeddings and h~(v) the mean of all labels of v's neighbours;
  * margin loss [gamma + d(h~(v), h(v)) - d(h~(v), h(u))]+ with Euclidean d and one uniformly
    sampled negative u != v;
  * Adam with a constant learning rate of 0.001, mini-batches of 64, 200 epochs, all neighbours;
  * label types are learned independently and the node embedding concatenates them, so each type
    is trained as its own EP run here.

Evaluation follows the paper:

  BlogCatalog, PPI, POS (no attributes; Table 3, transductive): a fraction Tr of nodes trains a
  one-vs-rest L2 logistic regression (liblinear) and the rest are test nodes.  Each test node is
  given its true number of labels, picked by score (the protocol of DeepWalk and node2vec that the
  paper follows).  The regulariser is chosen by 3-fold cross validation from
  {0.01, 0.1, 0.5, 1, 5, 10}, separately for micro and macro F1, and the margin gamma from
  {1, 5, 10, 20} by 3-fold cross validation at Tr = 0.1.

  Cora, Citeseer, Pubmed (ids + words; Table 5, transductive): 20 labelled nodes per class train
  the classifier, 1,000 nodes validate and another 1,000 test.  gamma and the regulariser are
  picked on the validation nodes.

Results are means and standard deviations over --runs random splits.  EP itself is trained once per
dataset and gamma (the paper doesn't say whether its 10 runs retrain EP).

Data (pass the directory holding these with --data):

  POS.mat, Homo_sapiens.mat       https://snap.stanford.edu/node2vec/
  blogcatalog.mat                 https://github.com/phanein/deepwalk/tree/master/example_graphs
  cora/, citeseer/                https://linqs-data.soe.ucsc.edu/public/lbc/{cora,citeseer}.tgz
  Pubmed-Diabetes/                https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz

Needs numpy, scipy and scikit-learn.

    python reproduce.py --data DIR                         # every dataset
    python reproduce.py --data DIR --datasets cora pos     # a subset
"""
import argparse
import json
import os
import tempfile
import time
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier

import cloverleaf as cl

# One-vs-rest warns for every label missing from a training fold, which is routine on PPI and POS.
warnings.filterwarnings("ignore", message="Label not .* is present in all training examples")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Paper results (Table 3 and Table 5, transductive) and the margins it reports choosing.
PAPER = {
    "blogcatalog": {"gamma": 1,
                    "micro": {0.1: 35.05, 0.5: 39.44, 0.9: 40.41},
                    "macro": {0.1: 19.08, 0.5: 25.11, 0.9: 25.97}},
    "pos": {"gamma": 10,
            "micro": {0.1: 46.97, 0.5: 49.52, 0.9: 50.05},
            "macro": {0.1: 8.85, 0.5: 10.45, 0.9: 12.17}},
    "ppi": {"gamma": 5,
            "micro": {0.1: 17.82, 0.5: 23.30, 0.9: 24.74},
            "macro": {0.1: 13.80, 0.5: 18.96, 0.9: 20.36}},
    "cora": {"gamma": 20, "accuracy": 78.05},
    "citeseer": {"gamma": 10, "accuracy": 71.01},
    "pubmed": {"gamma": 1, "accuracy": 79.56},
}
MULTI_LABEL = ["blogcatalog", "ppi", "pos"]
ATTRIBUTED = ["cora", "citeseer", "pubmed"]
GAMMAS = [1, 5, 10, 20]
C_GRID = [0.01, 0.1, 0.5, 1, 5, 10]
TRAIN_FRACTIONS = [0.1, 0.5, 0.9]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class Dataset:
    name: str
    num_nodes: int
    edges: np.ndarray          # (m, 2) undirected, each pair once, no self loops
    labels: np.ndarray         # multi-label: (n, k) bool; single label: (n,) int, -1 if unknown
    words: list | None = None  # per node list of word ids, or None


# Loading


def undirected(pairs, n):
    A = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)
    A.setdiag(0)
    A.eliminate_zeros()
    A = sp.triu(A).tocoo()
    return np.stack([A.row, A.col], axis=1)


def load_mat(name, path):
    d = sio.loadmat(path)
    A = sp.coo_matrix(d["network"])
    Y = sp.csr_matrix(d["group"]).toarray() > 0
    n = A.shape[0]
    return Dataset(name, n, undirected(np.stack([A.row, A.col], axis=1), n), Y)


def load_linqs(name, directory):
    ids, words, labels = {}, [], []
    classes = {}
    with open(os.path.join(directory, f"{name}.content")) as f:
        for line in f:
            row = line.rstrip("\n").split("\t")
            ids[row[0]] = len(ids)
            words.append([i for i, x in enumerate(row[1:-1]) if x == "1"])
            labels.append(classes.setdefault(row[-1], len(classes)))
    pairs = []
    with open(os.path.join(directory, f"{name}.cites")) as f:
        for line in f:
            a, b = line.split()
            for x in (a, b):
                # Some citations name papers missing from the content file: unlabelled nodes
                # with no words.
                if x not in ids:
                    ids[x] = len(ids)
                    words.append([])
                    labels.append(-1)
            pairs.append((ids[a], ids[b]))
    n = len(ids)
    return Dataset(name, n, undirected(np.array(pairs), n), np.array(labels), words)


def load_pubmed(directory):
    base = os.path.join(directory, "data", "Pubmed-Diabetes")
    ids, words, labels, vocab = {}, [], [], {}
    with open(base + ".NODE.paper.tab") as f:
        next(f), next(f)
        for line in f:
            row = line.rstrip("\n").split("\t")
            ids[row[0]] = len(ids)
            labels.append(int(row[1].split("=")[1]) - 1)
            words.append([vocab.setdefault(t.split("=")[0], len(vocab))
                          for t in row[2:] if t.startswith("w-")])
    pairs = []
    with open(base + ".DIRECTED.cites.tab") as f:
        next(f), next(f)
        for line in f:
            row = line.rstrip("\n").split("\t")
            a, b = row[1].split(":")[1], row[3].split(":")[1]
            pairs.append((ids[a], ids[b]))
    n = len(ids)
    return Dataset("pubmed", n, undirected(np.array(pairs), n), np.array(labels), words)


def load(name, data):
    if name == "blogcatalog":
        return load_mat(name, os.path.join(data, "blogcatalog.mat"))
    if name == "ppi":
        return load_mat(name, os.path.join(data, "Homo_sapiens.mat"))
    if name == "pos":
        return load_mat(name, os.path.join(data, "POS.mat"))
    if name == "pubmed":
        return load_pubmed(os.path.join(data, "Pubmed-Diabetes"))
    return load_linqs(name, os.path.join(data, name))


# EP-B


def load_graph(ds, tmp):
    path = os.path.join(tmp, f"{ds.name}.edges")
    with open(path, "w") as f:
        for a, b in ds.edges:
            f.write(f"n\t{a}\tn\t{b}\t1\n")
    return cl.Graph.load(path, cl.EdgeType.Undirected)


def node_labels(ds, label_type, v):
    if label_type == "id":
        return [("id", str(v))]
    labels = [("w", str(w)) for w in ds.words[v]]
    # As in the paper, a node without labels of a type gets a dummy label of its own.
    return labels or [("dummy", str(v))]


def train_label_type(ds, graph, label_type, gamma, seed, args):
    """Runs EP on one label type and returns h(v), the mean label embedding, for every node."""
    features = cl.FeatureSet.new_from_graph(graph)
    in_graph = np.zeros(ds.num_nodes, bool)
    in_graph[ds.edges.ravel()] = True
    for v in np.flatnonzero(in_graph):
        features.set_features(("n", str(v)), node_labels(ds, label_type, v))
    ep = cl.EmbeddingPropagator(
        alpha=args.lr, loss=cl.EPLoss.margin(float(gamma), 1), batch_size=args.batch_size,
        dims=args.dims, passes=args.passes, seed=seed, max_nodes=args.max_nodes,
        valid_pct=0.0, indicator=False, lr_schedule=args.lr_schedule, init=args.init)
    fe = ep.learn_features(graph, features)
    lookup = {}
    for i in range(len(fe)):
        key, emb = fe[i]
        lookup[key] = np.asarray(emb, np.float32)
    H = np.zeros((ds.num_nodes, args.dims), np.float32)
    for v in range(ds.num_nodes):
        vecs = [lookup[l] for l in node_labels(ds, label_type, v) if l in lookup]
        if vecs:
            H[v] = np.mean(vecs, axis=0)
    return H


EP_SETTINGS = ["dims", "passes", "batch_size", "lr", "lr_schedule", "init", "max_nodes", "seed"]


def embed(ds, graph, gamma, seed, args):
    """EP-B embeddings for every node, read from --cache when an identical run is there."""
    path = None
    if args.cache:
        tag = "-".join(f"{k}={getattr(args, k)}" for k in EP_SETTINGS)
        path = os.path.join(args.cache, f"{ds.name}-gamma={gamma:g}-{tag}.npy")
        if os.path.exists(path):
            return np.load(path), 0.0
    types = ["id"] + (["word"] if ds.words is not None else [])
    start = time.time()
    H = np.concatenate([train_label_type(ds, graph, t, gamma, seed, args) for t in types], axis=1)
    if path:
        np.save(path, H)
    return H, time.time() - start


# Multi-label evaluation (BlogCatalog, PPI, POS)


def top_k(scores, k):
    pred = np.zeros(scores.shape, bool)
    order = np.argsort(-scores, axis=1)
    for i, ki in enumerate(k):
        pred[i, order[i, :ki]] = True
    return pred


def multilabel_scores(Xtr, Ytr, Xte, Yte, C):
    clf = OneVsRestClassifier(LogisticRegression(solver="liblinear", C=C), n_jobs=-1)
    clf.fit(Xtr, Ytr)
    # predict_proba, not decision_function: a label missing from the training split gets a
    # constant decision score of 0, which outranks every real label with a negative score.
    pred = top_k(clf.predict_proba(Xte), Yte.sum(axis=1))
    return {"micro": 100 * f1_score(Yte, pred, average="micro", zero_division=0),
            "macro": 100 * f1_score(Yte, pred, average="macro", zero_division=0)}


def cv_scores(X, Y, rng):
    """3-fold cross-validated micro and macro F1 for every regulariser."""
    folds = KFold(3, shuffle=True, random_state=int(rng.integers(2**31))).split(X)
    totals = {C: {"micro": 0.0, "macro": 0.0} for C in C_GRID}
    for tr, va in folds:
        for C in C_GRID:
            for m, v in multilabel_scores(X[tr], Y[tr], X[va], Y[va], C).items():
                totals[C][m] += v / 3
    return totals


def split(n, fraction, rng):
    perm = rng.permutation(n)
    cut = int(round(fraction * n))
    return perm[:cut], perm[cut:]


def evaluate_multilabel(H, Y, fraction, runs, seed):
    out = {"micro": [], "macro": []}
    for run in range(runs):
        rng = np.random.default_rng([seed, run, int(fraction * 100)])
        tr, te = split(len(Y), fraction, rng)
        cv = cv_scores(H[tr], Y[tr], rng)
        for m in ("micro", "macro"):
            C = max(C_GRID, key=lambda c: cv[c][m])
            out[m].append(multilabel_scores(H[tr], Y[tr], H[te], Y[te], C)[m])
    return {m: (float(np.mean(v)), float(np.std(v))) for m, v in out.items()}


def choose_gamma_multilabel(embeddings, Y, seed):
    """3-fold cross validation at Tr = 0.1, once, on micro F1."""
    rng = np.random.default_rng([seed, 999])
    tr, _ = split(len(Y), 0.1, rng)
    scores = {}
    for gamma, H in embeddings.items():
        cv = cv_scores(H[tr], Y[tr], np.random.default_rng([seed, 998]))
        scores[gamma] = max(cv[C]["micro"] for C in C_GRID)
        log(f"  gamma={gamma}: cross-validated micro F1 {scores[gamma]:.2f}")
    return max(scores, key=scores.get), scores


# Single-label evaluation (Cora, Citeseer, Pubmed)


def planetoid_split(labels, rng):
    labelled = np.flatnonzero(labels >= 0)
    train = np.concatenate([rng.choice(np.flatnonzero(labels == c), 20, replace=False)
                            for c in np.unique(labels[labelled])])
    rest = rng.permutation(np.setdiff1d(labelled, train))
    return train, rest[:1000], rest[1000:2000]


def accuracy(Xtr, ytr, Xte, yte, C):
    clf = OneVsRestClassifier(LogisticRegression(solver="liblinear", C=C)).fit(Xtr, ytr)
    return 100 * float((clf.predict(Xte) == yte).mean())


def evaluate_attributed(embeddings, labels, runs, seed):
    """Picks gamma and the regulariser on validation for every run, then scores test."""
    tests, chosen = [], []
    per_gamma = {g: [] for g in embeddings}
    for run in range(runs):
        rng = np.random.default_rng([seed, run])
        tr, va, te = planetoid_split(labels, rng)
        best = None
        for gamma, H in embeddings.items():
            C = max(C_GRID, key=lambda c: accuracy(H[tr], labels[tr], H[va], labels[va], c))
            val = accuracy(H[tr], labels[tr], H[va], labels[va], C)
            test = accuracy(H[tr], labels[tr], H[te], labels[te], C)
            per_gamma[gamma].append(test)
            if best is None or val > best[0]:
                best = (val, test, gamma)
        tests.append(best[1])
        chosen.append(best[2])
    return {"accuracy": (float(np.mean(tests)), float(np.std(tests))),
            "chosen_gammas": chosen,
            "per_gamma": {g: (float(np.mean(v)), float(np.std(v))) for g, v in per_gamma.items()}}


# Main


def fmt(mean_std):
    return f"{mean_std[0]:6.2f} ± {mean_std[1]:4.2f}"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True, help="Directory with the downloaded datasets.")
    p.add_argument("--datasets", nargs="+", default=MULTI_LABEL + ATTRIBUTED,
                   choices=MULTI_LABEL + ATTRIBUTED)
    p.add_argument("--gammas", type=float, nargs="+", default=GAMMAS)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--seed", type=int, default=2017)
    # EP-B settings from the paper; override to compare against cloverleaf's defaults.
    p.add_argument("--dims", type=int, default=128)
    p.add_argument("--passes", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--lr-schedule", default="constant", choices=["constant", "cosine"])
    p.add_argument("--init", default="glorot", choices=["glorot", "unit"])
    p.add_argument("--max-nodes", type=float, default=None,
                   help="Neighbours sampled per reconstruction (default: all).")
    p.add_argument("--out", default=None, help="Write results as JSON to this path.")
    p.add_argument("--cache", default=None,
                   help="Directory to keep trained embeddings in, so reruns skip training.")
    args = p.parse_args()

    results = {"settings": vars(args), "datasets": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for name in args.datasets:
            ds = load(name, args.data)
            log(f"{name}: {ds.num_nodes} nodes, {len(ds.edges)} edges"
                + (f", {sum(map(len, ds.words))} word labels" if ds.words else ""))
            graph = load_graph(ds, tmp)
            embeddings, seconds = {}, {}
            for gamma in args.gammas:
                embeddings[gamma], seconds[gamma] = embed(ds, graph, gamma, args.seed, args)
                log(f"  trained gamma={gamma:g} in {seconds[gamma]:.0f}s")

            paper = PAPER[name]
            r = {"train_seconds": seconds, "paper": paper}
            if name in MULTI_LABEL:
                gamma, cv = choose_gamma_multilabel(embeddings, ds.labels, args.seed)
                r["chosen_gamma"], r["gamma_cv"] = gamma, cv
                r["results"] = {}
                for g in sorted({gamma, paper["gamma"]} & set(embeddings)):
                    r["results"][g] = {
                        fr: evaluate_multilabel(embeddings[g], ds.labels, fr, args.runs,
                                                args.seed)
                        for fr in TRAIN_FRACTIONS}
                    for fr in TRAIN_FRACTIONS:
                        m = r["results"][g][fr]
                        log(f"  gamma={g:g} Tr={fr}: micro {fmt(m['micro'])} "
                            f"(paper {paper['micro'][fr]:.2f}), macro {fmt(m['macro'])} "
                            f"(paper {paper['macro'][fr]:.2f})")
            else:
                r["results"] = evaluate_attributed(embeddings, ds.labels, args.runs, args.seed)
                log(f"  accuracy {fmt(r['results']['accuracy'])} "
                    f"(paper {paper['accuracy']:.2f}); per gamma: "
                    + ", ".join(f"{g:g}: {fmt(v)}" for g, v in r["results"]["per_gamma"].items()))
            results["datasets"][name] = r
            if args.out:
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
