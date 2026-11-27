"""Snapshot of the Python API's results, so changes to the Rust core can't silently change what
the bindings return.

`collect` exercises every binding on a small fixed graph and returns a label -> result mapping:
values, reprs, saved files, and the exception type each bad input raises.  The test compares it
with `snapshot.json`, which was taken from the bindings as they were before the Python-only logic
moved into the Rust core.  After an intended change, rewrite it with

    python tests/python/test_snapshot.py --update

and review the diff.  Algorithms whose seeded results vary from run to run (random walks, VPCG,
PPR Embed, Instant Embeddings) are left out; learned embeddings are checked by shape only.
"""
import json
import math
import os
import random
import sys
import tempfile

import pytest

import cloverleaf as cl

SNAPSHOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshot.json")


def _plain(v):
    """Tuples become lists so results compare the same way after a JSON round trip."""
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, float):
        return float(f"{v:.6g}")
    return v


def collect(tmp):
    results = {}

    def path(name):
        return os.path.join(tmp, name)

    def emit(label, value):
        assert label not in results, label
        results[label] = _plain(value)

    def err(label, fn):
        try:
            emit(label, ("ok", fn()))
        except Exception as e:
            emit(label, ("raised", type(e).__name__))


    # --- graph
    random.seed(0)
    with open(path("g.edges"), "w") as f:
        for c in "ab":
            for i in range(30):
                for _ in range(4):
                    j = random.randrange(30)
                    if j != i: f.write(f"{c}\t{i}\t{c}\t{j}\t{random.randint(1, 3)}\n")
        for _ in range(5):
            f.write(f"a\t{random.randrange(30)}\tb\t{random.randrange(30)}\t1\n")
        f.write("lonely\tx\tlonely\ty\t1\n")
    g = cl.Graph.load(path("g.edges"), cl.EdgeType.Undirected)
    emit("graph", (g.nodes(), g.edges(), len(g), repr(g)))
    emit("graph vocab", list(g.vocab())[:5])
    emit("contains", (g.contains_node(("a", "1")), g.contains_node(("a", "zzz"))))
    emit("edges raw", g.get_edges(("a", "1")))
    emit("edges norm", g.get_edges(("a", "1"), normalized=True))
    err("edges unknown", lambda: g.get_edges(("a", "zzz")))
    g.save(path("saved.edges"))
    emit("saved", open(path("saved.edges")).read().splitlines()[:6])
    gt = g.transpose()
    emit("transpose", gt.get_edges(("a", "1"), normalized=True))
    g2 = cl.Graph.load(path("g.edges"), cl.EdgeType.Directed, weighted=False, deduplicate=True)
    emit("graph unweighted dedup", (g2.nodes(), g2.edges(), g2.get_edges(("a", "1"))))
    open(path("bad.edges"), "w").write("a\tb\tc\n")
    err("bad graph", lambda: cl.Graph.load(path("bad.edges"), cl.EdgeType.Directed))
    err("missing graph", lambda: cl.Graph.load(path("nope.edges"), cl.EdgeType.Directed))

    gb = cl.GraphBuilder()
    gb.add_edge(("n", "a"), ("n", "b"), 1.0, cl.EdgeType.Undirected)
    gb.add_edge(("n", "b"), ("n", "c"), 2.0, cl.EdgeType.Directed)
    emit("builder", repr(gb))
    bg = gb.build_graph()
    emit("built", (repr(bg), bg.get_edges(("n", "b")), repr(gb), gb.build_graph()))

    # --- walks
    rw = cl.RandomWalker(0.3, 2000, beta=0.5)
    err("rwr unknown", lambda: rw.walk(g, ("a", "zzz")))
    err("rwr bad restarts", lambda: cl.RandomWalker(0.0, 10))
    ppr = cl.SparsePPR(0.2)
    emit("sparse ppr", ppr.compute(g, ("a", "2"), k=5, filter_type=["a", "b"]))
    rp = cl.RandomPath(seed=3)
    emit("random path", rp.rollout(g, ("a", "3"), 3, 0.3, True))

    # --- embeddings
    names = [n for n in g.vocab()]
    rnd = random.Random(1)
    lst = [(n, [rnd.uniform(-1, 1) for _ in range(4)]) for n in names]
    ne = cl.NodeEmbeddings.new_from_list(lst, cl.Distance.Cosine)
    emit("ne", (repr(ne), ne.dims(), ne.len(), len(ne), ne.get_distance() == cl.Distance.Cosine))
    emit("ne get", ne.get_embedding(("a", "1")))
    err("ne get unknown", lambda: ne.get_embedding(("a", "zzz")))
    ne.set_embedding(("a", "1"), [1.0, 0.0, 0.0, 0.0])
    err("ne set wrong dims", lambda: ne.set_embedding(("a", "1"), [1.0]))
    emit("ne contains", (ne.contains(("a", "1")), ne.contains(("q", "1"))))
    emit("ne nn", ne.nearest_neighbor([1.0, 0, 0, 0], 4))
    emit("ne nn filtered", ne.nearest_neighbor([1.0, 0, 0, 0], 4, filter_type="b"))
    emit("ne dist", ne.compute_distance(cl.Query.node("a", "1"), cl.Query.embedding([0, 1.0, 0, 0])))
    emit("ne dist id", ne.compute_distance(cl.Query.node_id(0), cl.Query.node_id(1)))
    err("ne dist bad id", lambda: ne.compute_distance(cl.Query.node_id(10**6), cl.Query.node_id(1)))
    err("ne dist unknown", lambda: ne.compute_distance(cl.Query.node("a", "zzz"), cl.Query.node_id(1)))
    emit("ne getitem", (ne[0], ne[-1]))
    ne.save(path("ne.emb"))
    emit("ne saved", open(path("ne.emb")).read().splitlines()[:3])
    ne2 = cl.NodeEmbeddings.load(path("ne.emb"), cl.Distance.Euclidean)
    emit("ne loaded", (repr(ne2), ne2.get_embedding(("a", "1"))))
    ne3 = cl.NodeEmbeddings.load(path("ne.emb"), filter_type="b")
    emit("ne loaded filtered", (repr(ne3), list(ne3.vocab())[:3]))
    ne_unit = cl.NodeEmbeddings.new_from_list(lst, cl.Distance.Cosine)
    ne_unit.l2norm()
    emit("l2norm", ne_unit.get_embedding(("a", "2")))
    zeros = cl.NodeEmbeddings(g, 3, cl.Distance.Euclidean)
    emit("zeros", (repr(zeros), zeros.get_embedding(("a", "1"))))
    neb = cl.NodeEmbeddingsBuilder(cl.Distance.Dot)
    neb.add_embedding(("x", "1"), [1.0, 2.0])
    err("neb wrong dims", lambda: neb.add_embedding(("x", "2"), [1.0]))
    emit("neb", (repr(neb), repr(neb.build()), neb.build()))

    # --- features
    with open(path("feats.txt"), "w") as f:
        for n in names:
            if not n[1].isdigit() or (n[0] == "a" and int(n[1]) % 3 == 0): continue
            f.write(f"{n[0]}\t{n[1]}\tcolor:{int(n[1]) % 4} size:{int(n[1]) % 2} plain{int(n[1]) % 5}\n")
    for label, ns in [("default", None), ("node_type", cl.FeatureNamespace.node_type()),
                      ("prefix", cl.FeatureNamespace.prefix(":")), ("single", cl.FeatureNamespace.single("w"))]:
        fs = cl.FeatureSet.new_from_file(path("feats.txt"), ns)
        emit(f"fs {label}", (repr(fs), fs.nodes(), fs.num_features(), fs.get_features(("a", "1")), fs[0], fs[-1]))
    err("fs prefix empty", lambda: cl.FeatureNamespace.prefix(""))
    fs = cl.FeatureSet.new_from_graph(g)
    fs.load_into(path("feats.txt"), cl.FeatureNamespace.prefix(":"))
    emit("fs from graph", (repr(fs), fs.get_features(("a", "1")), fs.get_features(("a", "3"))))
    fs.set_features(("a", "3"), [("color", "9")])
    emit("fs set", fs.get_features(("a", "3")))
    err("fs set unknown", lambda: fs.set_features(("a", "zzz"), [("c", "1")]))
    emit("fs pruned", repr(fs.prune_min_count(5)))
    open(path("bad_feats.txt"), "w").write("a\t1\n")
    err("fs bad file", lambda: cl.FeatureSet.new_from_file(path("bad_feats.txt")))
    err("fs missing", lambda: cl.FeatureSet.new_from_file(path("nope.txt")))

    # --- aggregation over fixed feature embeddings
    fs_full = cl.FeatureSet.new_from_graph(g)
    fs_full.load_into(path("feats.txt"), cl.FeatureNamespace.prefix(":"))
    feat_names = sorted({f for n in names for f in fs_full.get_features(n)})
    frnd = random.Random(2)
    fe = cl.NodeEmbeddings.new_from_list([(f, [frnd.uniform(-1, 1) for _ in range(6)]) for f in feat_names], cl.Distance.Cosine)
    for label, agg in [("avg", cl.FeatureAggregator.Averaged()), ("weighted", cl.FeatureAggregator.Weighted(0.1, fs_full)),
                       ("attention", cl.FeatureAggregator.Attention(
                           6, 2, 8, [math.sin(i * 0.37) / 4 for i in range(48 * 6)],
                           [math.cos(i * 0.53) / 4 for i in range(6 * 16)]))]:
        agg.save(path(f"agg_{label}"))
        loaded = cl.FeatureAggregator.load(path(f"agg_{label}"))
        emit(f"agg {label}", (repr(agg), repr(loaded), open(path(f"agg_{label}")).read().splitlines()[:4]))
        emb = cl.NodeEmbedder(loaded)
        emit(f"embedder {label} repr", repr(emb))
        out = emb.embed_feature_set(fs_full, fe)
        emit(f"embed set {label}", (repr(out), out.get_embedding(("a", "1")), out.get_embedding(("a", "3"))))
        emit(f"embed adhoc {label}", emb.embed_adhoc([("color", "1"), ("size", "0")], fe))
        err(f"embed adhoc strict {label}", lambda: emb.embed_adhoc([("color", "1"), ("zzz", "0")], fe))
        emit(f"embed adhoc lax {label}", emb.embed_adhoc([("color", "1"), ("zzz", "0")], fe, strict=False))
        emit(f"bulk {label}", emb.bulk_embed_adhoc([[("color", "1")], [("size", "1")]], fe))
    open(path("agg_bad"), "w").write("Median\n")
    err("agg bad", lambda: cl.FeatureAggregator.load(path("agg_bad")))

    # --- alignment
    na = cl.NeighborhoodAligner(alpha=0.5)
    al = na.align(ne, g)
    emit("aligned", (repr(al), al.get_embedding(("a", "1"))))
    na.align_to_disk(path("aligned.emb"), ne, g, chunk_size=7)
    emit("aligned disk", open(path("aligned.emb")).read().splitlines()[:3])
    err("align mismatched", lambda: na.align(ne3, g))
    ann = cl.EmbAnn(ne, 3, 5, seed=4)
    emit("ann", ann.find(ne, cl.Query.node("a", "2"), 3))
    ann_b = cl.EmbAnn(ne, 3, 5, filter_type="b", seed=4)
    emit("ann filtered", ann_b.find(ne, cl.Query.node("a", "2"), 3))
    ea = cl.EmbeddingAligner(5, random_nodes=3)
    emit("emb aligner", (repr(ea), ea.align(ne, ann, ne_unit, cl.Query.node("a", "2"), seed=5)))
    emit("emb aligner bulk", ea.bulk_align(ne, ann, ne_unit, [cl.Query.node("a", "2"), cl.Query.embedding([0.1, 0.2, 0.3, 0.4])], seed=5))
    gann = cl.GraphAnn(g, max_steps=50)
    emit("graph ann", gann.find(cl.Query.node("a", "4"), ne, 3, seed=6))
    bw = cl.BiasedRandomWalker(0.3, 500, beta=0.5, blend=0.5)
    emit("biased", bw.walk(g, ne, ("a", "1"), cl.Query.node("b", "1"), k=4, seed=9))
    emit("biased rerank", bw.walk(g, ne, ("a", "1"), cl.Query.node("b", "1"), k=4, seed=9, rerank_context=cl.Query.node("b", "2")))
    emit("biased fixed", cl.BiasedRandomWalker(3, 500).walk(g, ne, ("a", "1"), cl.Query.node("b", "1"), k=4, seed=9))
    err("biased bad", lambda: cl.BiasedRandomWalker(0.0, 5).walk(g, ne, ("a", "1"), cl.Query.node("b", "1")))

    # --- ranking and graph algorithms
    pr = cl.PageRank(20)
    emit("pagerank", (repr(pr), pr.learn(g, indicator=False).get_embedding(("a", "1"))))
    tb = cl.TournamentBuilder()
    tb.add_outcome(("p", "a"), ("p", "b"), 1.0)
    tb.add_ranked_outcomes([("p", "c"), ("p", "a"), ("p", "d")], 2.0)
    emit("tournament builder", repr(tb))
    t = tb.build()
    emit("tournament", (repr(t), repr(tb), tb.build()))
    lsr = cl.LSR(30).learn(t, indicator=False)
    emit("lsr", [(n, lsr.get_embedding(n)) for n in [("p", "a"), ("p", "b"), ("p", "c"), ("p", "d")]])
    cc = cl.ConnectedComponents.learn(g)
    emit("components", (cc.get_embedding(("a", "1")), cc.get_embedding(("lonely", "x"))))
    pruned = cl.ConnectedComponents.prune_largest_components(g, 1)
    emit("pruned", (repr(pruned), pruned.contains_node(("lonely", "x")), pruned.get_edges(("a", "1"), normalized=True)))

    pe_graph = cl.Graph.load(path("g.edges"), cl.EdgeType.Directed)
    pe = cl.PolicyEvaluation(pe_graph)
    emit("pe", repr(pe))
    with open(path("rewards.txt"), "w") as f:
        f.write("a\t1\t1.0\nb\t2\t-0.5\n")
    pe.load_rewards(path("rewards.txt"))
    pe.optimize(pe_graph, indicator=False)
    emit("pe optimized", pe_graph.get_edges(("a", "1"), normalized=True))
    open(path("bad_rewards.txt"), "w").write("a\t1\n")
    err("pe bad rewards", lambda: pe.load_rewards(path("bad_rewards.txt")))
    open(path("unknown_rewards.txt"), "w").write("a\tzzz\t1.0\n")
    err("pe unknown rewards", lambda: pe.load_rewards(path("unknown_rewards.txt")))
    open(path("nan_rewards.txt"), "w").write("a\t1\tabc\n")
    err("pe bad value", lambda: pe.load_rewards(path("nan_rewards.txt")))

    smci = cl.Smci(g)
    smci.add_reward(("a", "1"), ("a", "2"), 1.0)
    sg = smci.optimize(5, 20, 0.5, 0.9, 0.0, 0.1, 0.2, seed=3)
    emit("smci", (repr(sg), sg.get_edges(("a", "1"), normalized=True)))
    sg2 = smci.optimize(5, 20, 0.5, 0.9, 0.0, 0.1, 0.2, embeddings=ne, seed=3)
    emit("smci emb", sg2.get_edges(("a", "1"), normalized=True))


    emit("dist emb", cl.DistanceEmbedder(3).learn(g).get_embedding(("a", "1")))
    emit("lpa emb", cl.ClusterLPAEmbedder(3, 5, seed=1).learn(g).get_embedding(("a", "1")))
    emit("slpa emb", cl.SLPAEmbedder(5, 2, seed=1).learn(g).get_embedding(("a", "1")))
    err("ppr emb bad steps", lambda: cl.PPREmbedder(8, 5, 0.0).learn(g, cl.FeatureSet.new_from_graph(g)))
    err("instant bad", lambda: cl.InstantEmbeddings.random_walk(8, 2, 5, -1.0))

    # FastRP is deterministic, so its values are pinned.
    frp = cl.FastRPEmbedder(8, normalization_strength=-0.5, indicator=False)
    emit("fastrp", (repr(frp), frp.learn(g).get_embedding(("a", "1"))))
    fs_f = cl.FeatureSet.new_from_graph(g)
    fs_f.load_into(path("feats.txt"), cl.FeatureNamespace.prefix(":"))
    emit("fastrp features", frp.learn(g, fs_f, feature_dims=4).get_embedding(("a", "3")))
    err("fastrp bad weights", lambda: cl.FastRPEmbedder(8, weights=[0.0]))
    err("fastrp feature_dims without features", lambda: frp.learn(g, feature_dims=2))

    fp = cl.FeaturePropagator(2)
    fs_p = cl.FeatureSet.new_from_graph(g)
    fs_p.load_into(path("feats.txt"), cl.FeatureNamespace.prefix(":"))
    fp.propagate(g, fs_p)
    emit("propagated", fs_p.get_features(("a", "3")))

    # --- learning: shapes and warm-start handling (training is hogwild, so values vary)
    ep = cl.EmbeddingPropagator(dims=8, passes=2, indicator=False, valid_pct=0.0, seed=1)
    fs_e = cl.FeatureSet.new_from_graph(g)
    fe1 = ep.learn_features(g, fs_e)
    emit("ep", (repr(fe1).split(",")[0], fe1.dims(), fs_e.num_features()))
    fe2 = ep.learn_features(g, fs_e, fe1)
    emit("ep warm", (fe2.dims(), fe1.dims(), fe1.len()))
    err("ep mismatched features", lambda: ep.learn_features(g, cl.FeatureSet.new_from_file(path("feats.txt"))))
    bad_warm = cl.NodeEmbeddings.new_from_list([(("x", "1"), [1.0] * 8)], cl.Distance.Cosine)
    err("ep mismatched warm", lambda: ep.learn_features(g, fs_e, bad_warm))
    emit("ep mismatched warm untouched", bad_warm.dims())
    pr_l = cl.PprRankLearner(0.01, 32, 8, 1, 0.3, 5, 5, 2, valid_pct=0.0)
    fs_r = cl.FeatureSet.new_from_graph(g)
    fr = pr_l.learn_features(g, fs_r, indicator=False, seed=1)
    emit("pprrank", (fr.dims(), fr.len()))

    return results


def close(expected, actual):
    if isinstance(expected, float) or isinstance(actual, float):
        if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
            return False
        return math.isclose(expected, actual, rel_tol=1e-3, abs_tol=1e-4)
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(expected) == len(actual)
                and all(close(e, a) for e, a in zip(expected, actual)))
    return expected == actual


def _load_snapshot():
    """The recorded results, or nothing before the first `--update`."""
    if not os.path.exists(SNAPSHOT):
        return {}
    with open(SNAPSHOT) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    return collect(str(tmp_path_factory.mktemp("snapshot")))


def test_snapshot_covers_every_case(results):
    assert sorted(results) == sorted(_load_snapshot())


@pytest.mark.parametrize("label", list(_load_snapshot()))
def test_snapshot(results, label):
    expected = _load_snapshot()[label]
    assert close(expected, results.get(label)), f"{label}: expected {expected!r}, got {results.get(label)!r}"


if __name__ == "__main__":
    if sys.argv[1:] != ["--update"]:
        sys.exit("usage: test_snapshot.py --update")
    with tempfile.TemporaryDirectory() as tmp:
        snapshot = collect(tmp)
    with open(SNAPSHOT, "w") as f:
        json.dump(snapshot, f, indent=1, sort_keys=False)
        f.write("\n")
    print(f"wrote {len(snapshot)} cases to {SNAPSHOT}")
