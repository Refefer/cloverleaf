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

embedder = cloverleaf.RSVDEmbedder(k=2, n_samples=None, n_subspace_iters=None)
es = embedder.learn(graph)

print(es.get_embedding(('node', '1')))
print(es.get_embedding(('node', '2')))
print(es.get_embedding(('node', '5')))
print(es.get_embedding(('node', '6')))
