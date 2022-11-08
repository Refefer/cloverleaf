import time
import sys
import json
import cloverleaf
import numpy as np

PASSES = 100
BATCH_SIZE = 128
ALPHA = 9e-1
DIMS = 50
MAX_FEATURES=None
MAX_NODES=10
WD = 0
#loss = cloverleaf.EPLoss.contrastive(1, 5)
    #loss = cloverleaf.EPLoss.starspace(0.5, 5)
LOSS = cloverleaf.EPLoss.margin(10, 5)


def main(g_name, f_name):
    
    print("Loading graph...")
    graph = cloverleaf.RwrGraph.load(g_name, cloverleaf.EdgeType.Undirected)
    #graph = load_graph(g_name)
    print("Nodes={},Edges={}".format(graph.nodes(), graph.edges()), file=sys.stderr)
    print('Loading features...')
    features = cloverleaf.FeatureSet(graph)
    features.load_features(f_name, error_on_missing=False)
    print("Unique Features found: {}".format(features.num_features()))
    sTime = time.time()
    ep = cloverleaf.EmbeddingPropagator(
        alpha=ALPHA, gamma=0.9, loss=LOSS, batch_size=BATCH_SIZE, dims=DIMS, 
        passes=PASSES, wd=WD, max_nodes=MAX_NODES, max_features=MAX_FEATURES)

    feature_embeddings = ep.learn_features(graph, features)
    eTime = time.time() - sTime

    print("Time to learn:{}, Nodes/sec:{}".format(eTime, (graph.nodes() * 50) / eTime, file=sys.stderr))
    feature_embeddings.save(g_name+'.feature-embeddings')

    print("Constructing nodes...")
    embedder = cloverleaf.FeatureEmbeddingAggregator(features)
    node_embeddings = embedder.embed_graph(graph, features, feature_embeddings, alpha=1e-3)
    #node_embeddings = embedder.embed_graph(graph, features, feature_embeddings)
    embedder.save(g_name+'.embedder')

    node_embeddings.save(g_name+'.node-embeddings')
    #feature_embeddings.save(g_name+'.feature-embeddings')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
