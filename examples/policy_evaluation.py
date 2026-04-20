import cloverleaf

gb = cloverleaf.GraphBuilder()

gb.add_edge(("user", "alice"), ("prod", "widget"), 3.0, cloverleaf.EdgeType.Directed)
gb.add_edge(("user", "alice"), ("prod", "gadget"), 1.0, cloverleaf.EdgeType.Directed)
gb.add_edge(("user", "bob"), ("prod", "gadget"), 5.0, cloverleaf.EdgeType.Directed)
gb.add_edge(("user", "bob"), ("prod", "doohickey"), 2.0, cloverleaf.EdgeType.Directed)
gb.add_edge(("shop", "mart"), ("prod", "widget"), 4.0, cloverleaf.EdgeType.Directed)
gb.add_edge(("shop", "mart"), ("prod", "doohickey"), 1.0, cloverleaf.EdgeType.Directed)

graph = gb.build_graph()
print(f"Original graph: {graph}\n")

pe = cloverleaf.PolicyEvaluation(
    graph,
    gamma=0.9,
    iterations=100,
    eps=1e-6,
    temperature=0.5,
    indicator=True,
)

# Products get positive reward -- we want trajectories biased toward them
pe.set_reward(("prod", "widget"), 10.0)
pe.set_reward(("prod", "gadget"), 5.0)
pe.set_reward(("prod", "doohickey"), 2.0)

new_graph = pe.optimize()

print("Edge weight changes under optimal policy:\n")
for node_type, node_name in graph.vocab():
    nodes_old, weights_old = graph.get_edges((node_type, node_name), normalized=True)
    nodes_new, weights_new = new_graph.get_edges((node_type, node_name), normalized=True)
    if not nodes_old:
        continue
    print(f"  {node_type}:{node_name}")
    for (_, wo), (nn, wn) in zip(zip(nodes_old, weights_old), zip(nodes_new, weights_new)):
        changed = " <--" if abs(wo - wn) > 0.001 else ""
        print(f"    -> {nn[0]}:{nn[1]:10s}  {wo:.4f} -> {wn:.4f}{changed}")
    print()

new_graph.save("/tmp/pe_graph")
print("Saved reweighted graph to /tmp/pe_graph")
