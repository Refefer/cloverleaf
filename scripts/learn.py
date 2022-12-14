import argparse
import time
import sys
import json
import cloverleaf
import numpy as np

PASSES = 200
BATCH_SIZE = 128
ALPHA = 9e-1
DIMS = 50
MAX_FEATURES=None
MAX_NODES=10
WD = 0
#loss = cloverleaf.EPLoss.contrastive(1, 5)
    #loss = cloverleaf.EPLoss.starspace(0.5, 5)
LOSS = cloverleaf.EPLoss.margin(10, 5)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Plots embeddings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("edges",
        help="Path to edges.")

    parser.add_argument("features",
        help="Path to node features.")

    parser.add_argument("output",
        help="Output namespace")

    parser.add_argument("--dims",
        dest="dims",
        type=int,
        default=100,
        help="Number of dimensions for each embedding.")

    parser.add_argument("--passes",
        type=int,
        default=200,
        help="Number of optimization passes")

    parser.add_argument("--lr",
        type=float,
        default=9e-1,
        help="Learning Rate.")

    parser.add_argument("--momentum",
        type=float,
        default=1e-9,
        help="Nestrov Momentum rate.")

    parser.add_argument("--batch-size",
        type=int,
        default=128,
        help="Batch size.")

    parser.add_argument("--max-features",
        dest="max_features",
        type=int,
        default=None,
        help="If provided, samples a max of MAX_FEATURES for each node embedding construction.")

    parser.add_argument("--max-neighbors",
        dest="max_neighbors",
        type=int,
        default=10,
        help="Samples MAX_NEIGHBORS nodes for node reconstruction.")

    parser.add_argument("--weight-decay",
        dest="wd",
        type=float,
        default=0,
        help="If provided, adds weight decay to the embeddings.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--embedding-propagation",
        dest="ep",
        nargs=2,
        metavar=('MARGIN', 'NEGATIVES'),
        help="Uses margin loss for optimization.")

    group.add_argument("--starspace",
        dest="starspace",
        nargs=2,
        metavar=('MARGIN', 'NEGATIVES'),
        help="Optimizes using Starspace embedding learning.")

    group.add_argument("--contrastive",
        dest="contrastive",
        nargs=2,
        metavar=('TEMPERATURE', 'NEGATIVES'),
        help="Optimizes using contrastive loss.")

    return parser


def main(args):

    g_name = args.edges
    f_name = args.features
    
    print("Loading graph...")
    graph = cloverleaf.RwrGraph.load(g_name, cloverleaf.EdgeType.Undirected)
    print("Nodes={},Edges={}".format(graph.nodes(), graph.edges()), file=sys.stderr)
    print('Loading features...')
    features = cloverleaf.FeatureSet(graph)
    features.load_features(f_name, error_on_missing=False)
    print("Unique Features found: {}".format(features.num_features()))
    sTime = time.time()

    if args.ep is not None:
        margin, negatives = args.ep
        loss = cloverleaf.EPLoss.margin(float(margin), int(negatives))
    elif args.starspace is not None:
        margin, negatives = args.starspace
        loss = cloverleaf.EPLoss.starspace(float(margin), int(negatives))
    else:
        temp, negs = args.contrastive
        loss = cloverleaf.EPLoss.contrastive(float(temp), int(negs))

    ep = cloverleaf.EmbeddingPropagator(
        alpha=args.lr, gamma=args.momentum, loss=loss, batch_size=args.batch_size, dims=args.dims, 
        passes=args.passes, wd=args.wd, max_nodes=args.max_neighbors, max_features=args.max_features)

    feature_embeddings = ep.learn_features(graph, features)
    eTime = time.time() - sTime

    print("Time to learn:{}, Nodes/sec:{}".format(eTime, (graph.nodes() * 50) / eTime, file=sys.stderr))
    feature_embeddings.save(args.output + '.feature-embeddings')

    print("Constructing nodes...")
    embedder = cloverleaf.FeatureEmbeddingAggregator(features)
    node_embeddings = embedder.embed_graph(graph, features, feature_embeddings, alpha=1e-3)
    embedder.save(args.output + '.embedder')
    node_embeddings.save(args.output + '.node-embeddings')

if __name__ == '__main__':
    main(build_arg_parser().parse_args())
