import sys
import cloverleaf
graph = cloverleaf.Graph.load(sys.argv[1], cloverleaf.EdgeType.Undirected)
slpa = cloverleaf.SLPAEmbedder(t=256, threshold=2, memory_size=64, rule=cloverleaf.ListenerRule.Probabilistic)
embs = slpa.learn(graph)
embs.save('/tmp/embs.test')
