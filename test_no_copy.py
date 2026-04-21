"""
Validates PolicyEvaluation's new in-place optimize(graph, no_copy=...) contract:

  1. Happy path: no other references to the graph -> default (no_copy=True) succeeds.
  2. Shared inner Arc (via Smci holding a clone) -> default no_copy=True raises
     RuntimeError with a Python-flavored message, and the graph is left intact.
  3. Same sharing, but no_copy=False -> falls back to a clone and succeeds.
"""
import cloverleaf


def build_graph():
    # Use Undirected so every node has at least one outbound edge; the
    # policy-evaluation CDF update path panics on dead-end nodes (pre-existing
    # bug in src/graph.rs:684, not part of this refactor).
    gb = cloverleaf.GraphBuilder()
    gb.add_edge(("user", "alice"), ("prod", "widget"), 3.0, cloverleaf.EdgeType.Undirected)
    gb.add_edge(("user", "alice"), ("prod", "gadget"), 1.0, cloverleaf.EdgeType.Undirected)
    gb.add_edge(("user", "bob"), ("prod", "gadget"), 5.0, cloverleaf.EdgeType.Undirected)
    gb.add_edge(("user", "bob"), ("prod", "doohickey"), 2.0, cloverleaf.EdgeType.Undirected)
    return gb.build_graph()


def make_pe(graph):
    pe = cloverleaf.PolicyEvaluation(
        graph, gamma=0.9, iterations=50, eps=1e-6, temperature=0.5
    )
    pe.set_reward(("prod", "widget"), 10.0)
    pe.set_reward(("prod", "gadget"), 5.0)
    pe.set_reward(("prod", "doohickey"), 2.0)
    return pe


def snapshot(graph):
    snap = {}
    for nt, nn in graph.vocab():
        nodes, weights = graph.get_edges((nt, nn), normalized=True)
        if nodes:
            snap[(nt, nn)] = list(weights)
    return snap


def weights_equal(a, b):
    if set(a) != set(b):
        return False
    for k in a:
        if len(a[k]) != len(b[k]):
            return False
        for wa, wb in zip(a[k], b[k]):
            if abs(wa - wb) > 1e-6:
                return False
    return True


def test_happy_path():
    graph = build_graph()
    pe = make_pe(graph)
    before = snapshot(graph)
    pe.optimize(graph, indicator=False)  # no_copy defaults to True
    after = snapshot(graph)
    assert not weights_equal(before, after), "graph weights should have changed"
    print("PASS: no_copy=True default succeeds when graph has no other references")


def test_shared_arc_raises_by_default():
    graph = build_graph()
    pe = make_pe(graph)
    # Smci's constructor clones graph.graph (Arc<CumCSR>) into its struct.
    # That bumps the inner CSR's refcount to 2 -> optimize must refuse.
    _smci = cloverleaf.Smci(graph)
    before = snapshot(graph)
    try:
        pe.optimize(graph, indicator=False)
    except RuntimeError as e:
        msg = str(e)
        assert "other live references" in msg, f"unexpected message: {msg}"
        assert "no_copy=False" in msg, f"message must mention the escape hatch: {msg}"
        after = snapshot(graph)
        assert weights_equal(before, after), "graph should be untouched after the error"
        print("PASS: no_copy=True default raises RuntimeError on shared graph")
        print(f"       message: {msg!r}")
        return
    raise AssertionError("optimize should have raised but did not")


def test_shared_arc_opt_in_copy():
    graph = build_graph()
    pe = make_pe(graph)
    _smci = cloverleaf.Smci(graph)
    before = snapshot(graph)
    pe.optimize(graph, no_copy=False, indicator=False)
    after = snapshot(graph)
    assert not weights_equal(before, after), "graph weights should have changed"
    print("PASS: no_copy=False falls back to clone and succeeds on shared graph")


def test_vocab_mismatch():
    g1 = build_graph()
    g2 = build_graph()  # independently built => independent vocab Arc
    pe = make_pe(g1)
    try:
        pe.optimize(g2, indicator=False)
    except ValueError as e:
        assert "different graph" in str(e).lower() or "vocabulary" in str(e).lower(), str(e)
        print("PASS: vocab mismatch raises ValueError")
        return
    raise AssertionError("optimize on different-vocab graph should have raised")


if __name__ == "__main__":
    test_happy_path()
    test_shared_arc_raises_by_default()
    test_shared_arc_opt_in_copy()
    test_vocab_mismatch()
    print("\nAll no_copy contract tests passed.")
