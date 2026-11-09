import cloverleaf

edges = [
    (0, 1, 0.2),
    (0, 2, 0.3),
    (1, 2, 1),
]

def printg(g):
    for i in range(3):
        print(i, g.get_edges(t(i)))

gb = cloverleaf.GraphBuilder()

t = lambda n: ('n', str(n))
for fn, tn, w in edges:
    gb.add_edge(t(fn), t(tn), w, cloverleaf.EdgeType.Directed)

graph = gb.build_graph()
printg(graph)
tgraph = graph.transpose();
printg(tgraph)

(edges, weights) = tgraph.get_edges(t(0))
assert len(edges) == 0

(edges, weights) = tgraph.get_edges(t(1))
print(edges)
assert edges == [t(0)]
assert weights == [1.0]
