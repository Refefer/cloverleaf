import cloverleaf as cl

graph = cl.Graph.load('pinterest.edges', cl.EdgeType.Undirected)
N = 6
weights = [1.0/N]*N
sp_embedder = cl.FastRandomProjectionEmbedder(
    dims=128,
    weights = weights,
    norm_powers = True,
)

res = sp_embedder.learn(graph)
res.save("fast-cloverleaf.embeddings")
