"""Benchmark: reproduce RP3beta on the Pinterest leave-one-out split.

Leave-one-out on Pinterest (55,187 users, 9,916 boards): each test user has one held-out board
and sampled negatives; boards are ranked by score and we report HR@5, NDCG@5, HR@10 and NDCG@10.
Targets are from Dacrema et al., "Are We Really Making Much Progress?", RecSys 2019.  Two splits:

  cmn  The files in this directory: 100 negatives per user.  This is the split of the CMN
       comparison (Table 2); TopPopular reproduces that table's row.  4.7% of test positives also
       appear in the training edges; with seen items excluded (below) those count as misses.
  ncf  The original NCF split (99 negatives, Table 6), from
       https://github.com/hexiangnan/neural_collaborative_filtering/tree/master/Data
       (pinterest-20.train.rating, pinterest-20.test.negative).  Pass the directory to --ncf-dir.

Seen items are excluded by default, as in Dacrema et al.'s evaluator: a candidate the user already
has in train is ranked last.  TopPopular is evaluated too, as a check that the protocol matches.

Three scorers:

  exact       RP3beta computed exactly with sparse matrices, with the alpha / beta / top-k
              parameters of Dacrema et al.'s implementation.  Checks that the protocol reproduces
              the published numbers.
  exact-p3    The same with alpha=1 and no top-k pruning: the quantity cloverleaf's 3-step
              RandomWalker estimates.  The gap between this and the walker is sampling error.
  cloverleaf  RandomWalker(restarts=3, walks=W, beta): visit counts of 3-step walks from the user,
              divided by degree^beta.

Hyper-parameters are tuned on a validation split carved out of the training edges (one held-out
board per user, 100 sampled negatives), never on the test set.  The walker reuses exact-p3's beta,
since it estimates the same scores.

Ties are broken uniformly at random.  Counting ties in the positive's favour inflates sparse
scorers: a walker that never reaches the positive or any negative would score a hit.

Duplicate (user, board) edges are collapsed by default, as implicit feedback is binary.

Usage (from this directory, after `maturin develop --release` and `pip install numpy scipy`):

    python bench_rp3b.py                       # tune, then evaluate everything on all users
    python bench_rp3b.py --split ncf --ncf-dir DIR
    python bench_rp3b.py --walks 1000 10000    # walker sample sizes (default 10k, 100k, 1M)
    python bench_rp3b.py --max-users 5000      # subsample test users for a quick run
"""
import argparse
import gzip
import itertools
import json
import multiprocessing as mp
import os
import tempfile
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))

# Dacrema et al., RecSys 2019: Table 2 (CMN comparison) and Table 6 (NCF comparison).
PUBLISHED = {
    "cmn": {"RP3beta": {"HR@5": 0.7018, "NDCG@5": 0.5041, "HR@10": 0.8644, "NDCG@10": 0.5571},
            "TopPopular": {"HR@5": 0.1668, "NDCG@5": 0.1066, "HR@10": 0.2745, "NDCG@10": 0.1411}},
    "ncf": {"RP3beta": {"HR@5": 0.7105, "NDCG@5": 0.5116, "HR@10": 0.8740, "NDCG@10": 0.5650},
            "TopPopular": {"HR@5": 0.1663, "NDCG@5": 0.1065, "HR@10": 0.2744, "NDCG@10": 0.1412}},
}
METRICS = ["HR@5", "NDCG@5", "HR@10", "NDCG@10"]

ALPHAS = [0.5, 0.75, 1.0, 1.25]
BETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
TOPKS = [50, 100, 200, 500, 1000, None]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def open_text(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


# Data


def load_train(path, keep_duplicates):
    users, boards = [], []
    with open_text(path) as f:
        for line in f:
            _, u, _, b, _ = line.rstrip("\n").split("\t")
            users.append(int(u))
            boards.append(int(b))
    users, boards = np.array(users), np.array(boards)
    shape = (users.max() + 1, boards.max() + 1)
    # The csr constructor sums duplicate entries.
    R = sp.csr_matrix((np.ones(len(users), np.float32), (users, boards)), shape=shape)
    if not keep_duplicates:
        R.data[:] = 1
    return R


def load_test(path):
    users, cands = [], []
    with open_text(path) as f:
        for line in f:
            u, pos, negs = line.rstrip("\n").split("\t")
            users.append(int(u))
            cands.append([int(pos)] + [int(n) for n in negs.split()])
    # Column 0 is the positive.
    return np.array(users), np.array(cands)


def load_ncf(directory, keep_duplicates):
    users, boards = [], []
    with open(os.path.join(directory, "pinterest-20.train.rating")) as f:
        for line in f:
            u, b = line.split("\t")[:2]
            users.append(int(u))
            boards.append(int(b))
    users, boards = np.array(users), np.array(boards)
    R = sp.csr_matrix((np.ones(len(users), np.float32), (users, boards)),
                      shape=(users.max() + 1, boards.max() + 1))
    if not keep_duplicates:
        R.data[:] = 1
    test_users, cands = [], []
    with open(os.path.join(directory, "pinterest-20.test.negative")) as f:
        for line in f:
            head, *negs = line.rstrip("\n").split("\t")
            u, pos = (int(x) for x in head.strip("()").split(","))
            test_users.append(u)
            cands.append([pos] + [int(n) for n in negs])
    return R, np.array(test_users), np.array(cands)


def validation_split(R, test_users, test_cands, num_negs, rng):
    """Holds out one training board per user (with at least two) and samples negatives the user
    has never interacted with, in train or test.  Returns the reduced train matrix and the
    validation users and candidates."""
    R = R.tolil(copy=True)
    test_pos = dict(zip(test_users, test_cands[:, 0]))
    num_items = R.shape[1]
    users, cands = [], []
    for u in range(R.shape[0]):
        items = R.rows[u]
        if len(items) < 2:
            continue
        held = items[rng.integers(len(items))]
        seen = set(items)
        if u in test_pos:
            seen.add(test_pos[u])
        negs = []
        while len(negs) < num_negs:
            c = int(rng.integers(num_items))
            if c not in seen and c not in negs:
                negs.append(c)
        R[u, held] = 0
        users.append(u)
        cands.append([held] + negs)
    R = R.tocsr()
    R.eliminate_zeros()
    return R, np.array(users), np.array(cands)


# Metrics


def seen_mask(R, users, cands):
    """True where a candidate is already among the user's training boards."""
    return np.stack([np.isin(c, R.indices[R.indptr[u]:R.indptr[u + 1]])
                     for u, c in zip(users, cands)])


def ranks(scores, rng, seen=None):
    """Rank of the positive (column 0) among its candidates, breaking ties uniformly at random.
    Seen candidates, if given, are ranked last."""
    if seen is not None:
        scores = np.where(seen, -np.inf, scores)
    pos = scores[:, :1]
    better = (scores[:, 1:] > pos).sum(axis=1)
    ties = (scores[:, 1:] == pos).sum(axis=1)
    return better + rng.integers(0, ties + 1)


def metrics(rank):
    out = {}
    for k in (5, 10):
        hit = rank < k
        out[f"HR@{k}"] = float(hit.mean())
        out[f"NDCG@{k}"] = float(np.where(hit, 1.0 / np.log2(rank + 2.0), 0.0).mean())
    return {m: out[m] for m in METRICS}


# Exact RP3beta


def l1_rows(M):
    s = np.asarray(M.sum(axis=1)).ravel()
    s[s == 0] = 1
    return sp.diags(1.0 / s).astype(np.float32) @ M


def p3_similarity(R, alpha, block=2048):
    """Dense item x item matrix W[i, j] = sum_u P(i -> u)^alpha P(u -> j)^alpha."""
    Pui = l1_rows(R).tocsr()
    Piu = l1_rows(R.T.tocsr()).tocsr()
    if alpha != 1.0:
        Pui = Pui.power(alpha)
        Piu = Piu.power(alpha)
    n = R.shape[1]
    W = np.empty((n, n), np.float32)
    for start in range(0, n, block):
        W[start:start + block] = (Piu[start:start + block] @ Pui).toarray()
    return W


def popularity_discount(R, beta):
    # Board degree in the binarised graph, matching cloverleaf's graph.degree after deduplication.
    deg = np.asarray((R > 0).sum(axis=0)).ravel().astype(np.float32)
    out = np.zeros_like(deg)
    nz = deg > 0
    out[nz] = deg[nz] ** -beta
    return out


def prune_topk(W, k):
    if k is None or k >= W.shape[1]:
        return W
    out = np.zeros_like(W)
    idx = np.argpartition(-W, k - 1, axis=1)[:, :k]
    rows = np.arange(W.shape[0])[:, None]
    out[rows, idx] = W[rows, idx]
    return out


def score_candidates(R, W, users, cands, chunk=4096):
    """scores[n, c] = sum_i R[users[n], i] * W[i, cands[n, c]], touching only the candidates."""
    out = np.empty(cands.shape, np.float32)
    for start in range(0, len(users), chunk):
        sub = R[users[start:start + chunk]]
        row_of_nnz = np.repeat(np.arange(sub.shape[0]), np.diff(sub.indptr))
        contrib = W[sub.indices[:, None], cands[start:start + chunk][row_of_nnz]]
        contrib *= sub.data[:, None]
        sums = np.zeros((sub.shape[0], cands.shape[1]), np.float32)
        # Rows are contiguous in CSR order; reduceat sums each user's rows.  Users with no
        # training edges stay 0.
        nonempty = np.diff(sub.indptr) > 0
        if nonempty.any():
            sums[nonempty] = np.add.reduceat(contrib, sub.indptr[:-1][nonempty], axis=0)
        out[start:start + chunk] = sums
    return out


def evaluate_exact(R, W_alpha, beta, topk, users, cands, rng, seen):
    W = prune_topk(W_alpha * popularity_discount(R, beta)[None, :], topk)
    return metrics(ranks(score_candidates(R, W, users, cands), rng, seen))


def evaluate_toppop(R, users, cands, rng, seen):
    pop = np.asarray((R > 0).sum(axis=0)).ravel().astype(np.float32)
    return metrics(ranks(pop[cands], rng, seen))


def tune_exact(R_val, val_users, val_cands, val_seen, alphas, betas, topks, seed):
    """Grid search on validation HR@5 (the metric Dacrema et al. tuned for)."""
    results = []
    for alpha in alphas:
        W = p3_similarity(R_val, alpha)
        for beta, topk in itertools.product(betas, topks):
            m = evaluate_exact(R_val, W, beta, topk, val_users, val_cands,
                               np.random.default_rng(seed), val_seen)
            results.append(({"alpha": alpha, "beta": beta, "topk": topk}, m))
            log(f"  valid alpha={alpha} beta={beta} topk={topk}: "
                + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
    return max(results, key=lambda r: (r[1]["HR@5"], r[1]["NDCG@5"]))


# Cloverleaf walker

_GRAPH = None
_WALKER = None


def _init_worker(walks, beta):
    global _WALKER
    import cloverleaf as cl
    _WALKER = cl.RandomWalker(3, walks, beta)


def _walk_scores(job):
    user, cands, seed = job
    hits = _WALKER.walk(_GRAPH, ("user", str(user)), seed=seed, filter_type="board")
    found = {int(name): score for (_, name), score in hits}
    return [found.get(int(c), 0.0) for c in cands]


def write_graph(R, path):
    coo = R.tocoo()
    with open(path, "w") as f:
        for u, b, w in zip(coo.row, coo.col, coo.data):
            f.write(f"user\t{u}\tboard\t{b}\t{w:g}\n")


def load_graph(R, directory):
    """Writes the training edges as a cloverleaf edge file and loads it as an undirected graph.
    Workers inherit the graph through fork."""
    global _GRAPH
    import cloverleaf as cl
    path = os.path.join(directory, "train.edges")
    write_graph(R, path)
    _GRAPH = cl.Graph.load(path, cl.EdgeType.Undirected)


def evaluate_walker(walks, beta, users, cands, seed, procs, seen):
    jobs = [(int(u), c.tolist(), seed + int(u)) for u, c in zip(users, cands)]
    with mp.get_context("fork").Pool(procs, _init_worker, (walks, beta)) as pool:
        scores = np.array(pool.map(_walk_scores, jobs, chunksize=64), np.float32)
    return metrics(ranks(scores, np.random.default_rng(seed), seen))


# Main


def fmt_row(name, m):
    return f"{name:<34}" + "".join(f"{m[k]:>9.4f}" for k in METRICS)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--split", choices=["cmn", "ncf"], default="cmn")
    p.add_argument("--train", default=os.path.join(HERE, "pinterest.edges.gz"),
                   help="cmn split: training edges.")
    p.add_argument("--test", default=os.path.join(HERE, "pinterest.test.gz"),
                   help="cmn split: test users with their positive and negatives.")
    p.add_argument("--ncf-dir", default=None,
                   help="ncf split: directory with the NCF Pinterest files.")
    p.add_argument("--walks", type=int, nargs="*", default=[10000, 100000, 1000000],
                   help="Walker sample sizes to evaluate; none skips the walker.")
    p.add_argument("--max-users", type=int, default=None,
                   help="Evaluate on a random subset of test users.")
    p.add_argument("--keep-duplicates", action="store_true",
                   help="Weight repeated (user, board) edges instead of collapsing them.")
    p.add_argument("--include-seen", action="store_true",
                   help="Score candidates already in the user's training boards normally.")
    p.add_argument("--quick-grid", action="store_true",
                   help="Smaller tuning grid: alpha in {0.75, 1}, top-k in {100, 500, none}.")
    p.add_argument("--procs", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=2019)
    p.add_argument("--out", default=None, help="Write results as JSON to this path.")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    log(f"loading the {args.split} split")
    if args.split == "ncf":
        if not args.ncf_dir:
            p.error("--split ncf needs --ncf-dir")
        R, test_users, test_cands = load_ncf(args.ncf_dir, args.keep_duplicates)
    else:
        R = load_train(args.train, args.keep_duplicates)
        test_users, test_cands = load_test(args.test)
    log(f"{R.shape[0]} users, {R.shape[1]} boards, {R.nnz} edges, {len(test_users)} test users, "
        f"{test_cands.shape[1] - 1} negatives each")

    test_seen = seen_mask(R, test_users, test_cands)
    leaked = int(test_seen[:, 0].sum())
    if leaked:
        log(f"{leaked} test positives also appear in train; "
            + ("they score normally" if args.include_seen else "they count as misses"))

    if args.max_users and args.max_users < len(test_users):
        keep = np.sort(rng.choice(len(test_users), args.max_users, replace=False))
        test_users, test_cands, test_seen = test_users[keep], test_cands[keep], test_seen[keep]
        log(f"evaluating on {len(test_users)} sampled test users")
    if args.include_seen:
        test_seen = None

    log("building validation split")
    R_val, val_users, val_cands = validation_split(
        R, test_users, test_cands, test_cands.shape[1] - 1, rng)
    val_seen = None if args.include_seen else seen_mask(R_val, val_users, val_cands)

    alphas = [0.75, 1.0] if args.quick_grid else ALPHAS
    topks = [100, 500, None] if args.quick_grid else TOPKS

    log("tuning exact RP3beta on validation")
    best_exact, best_exact_val = tune_exact(R_val, val_users, val_cands, val_seen, alphas, BETAS,
                                            topks, args.seed)
    log("tuning exact P3 (alpha=1, no top-k) on validation")
    best_p3, best_p3_val = tune_exact(R_val, val_users, val_cands, val_seen, [1.0], BETAS, [None],
                                      args.seed)

    log("evaluating on test")
    published = PUBLISHED[args.split]
    results = {
        "published TopPopular": {"params": {}, "test": published["TopPopular"]},
        "TopPopular": {"params": {}, "test": evaluate_toppop(
            R, test_users, test_cands, np.random.default_rng(args.seed), test_seen)},
        "published RP3beta": {"params": {}, "test": published["RP3beta"]},
    }
    for name, params, val in [("exact RP3beta", best_exact, best_exact_val),
                              ("exact P3 (alpha=1, no top-k)", best_p3, best_p3_val)]:
        W = p3_similarity(R, params["alpha"])
        test = evaluate_exact(R, W, params["beta"], params["topk"], test_users, test_cands,
                              np.random.default_rng(args.seed), test_seen)
        results[name] = {"params": params, "valid": val, "test": test}
        del W

    if args.walks:
        with tempfile.TemporaryDirectory() as tmp:
            load_graph(R, tmp)
        for walks in args.walks:
            log(f"cloverleaf walker, {walks} walks per user")
            start = time.time()
            test = evaluate_walker(walks, best_p3["beta"], test_users, test_cands, args.seed,
                                   args.procs, test_seen)
            results[f"cloverleaf walker ({walks} walks)"] = {
                "params": {"walks": walks, "beta": best_p3["beta"]}, "test": test,
                "seconds": round(time.time() - start, 1)}

    print()
    print(f"{args.split + ' split, test':<34}" + "".join(f"{k:>9}" for k in METRICS))
    for name, r in results.items():
        print(fmt_row(name, r["test"]) + f"   {r['params'] or ''}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"split": args.split, "include_seen": args.include_seen,
                       "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
