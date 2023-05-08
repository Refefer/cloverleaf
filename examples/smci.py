import cloverleaf

gb = cloverleaf.GraphBuilder()

# 0 -- 1 -- 2 -- 3
#      |    |
# 4 -- 5 -- 6 -- 7
for i in range(0, 3):
    gb.add_edge(('node', str(i)), ('node', str(i+1)), 1, cloverleaf.EdgeType.Undirected)

for i in range(4, 7):
    gb.add_edge(('node', str(i)), ('node', str(i+1)), 1, cloverleaf.EdgeType.Undirected)

gb.add_edge(('node', '1'), ('node', '5'), 1, cloverleaf.EdgeType.Undirected)
gb.add_edge(('node', '2'), ('node', '6'), 1, cloverleaf.EdgeType.Undirected)

graph = gb.build_graph()

smci = cloverleaf.Smci(graph)

smci.add_reward(('node', '0'), ('node', '3'), 2)
smci.add_reward(('node', '0'), ('node', '4'), -2)
smci.add_reward(('node', '7'), ('node', '4'), 2)
smci.add_reward(('node', '7'), ('node', '3'), -2)

new_graph = smci.optimize(
    iterations=5,
    num_walks=100,
    alpha=0.1,
    discount=0.1,
    step_penalty=-0.1,
    explore_pct=0.0,
    compression=10.0,
    restart_prob=1/5,
    seed=123123)

new_graph.save("/tmp/here")
print("done")

