import cloverleaf as cl

graph = cl.Graph.load('pinterest.edges', cl.EdgeType.Undirected)
embedder = cl.ProneEmbedder(
    dims=128,
    n_subspace_iters=1,
    n_samples=None,
    order=5,
    mu=0.2,
    theta=0.5,
)

res = embedder.learn(graph)
res.save("prone-cloverleaf.embeddings")
