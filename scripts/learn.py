import argparse
import time
import sys
import json
import cloverleaf
import numpy as np

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Learn cloverleaf embeddings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("edges",
        help="Path to edges.")

    parser.add_argument("features",
        help="Path to node features.")

    parser.add_argument("output",
        help="Output namespace")

    parser.add_argument("--warm-start",
        dest="warm_start",
        required=False,
        default=None,
        help="If provided, loads feature embeddings from a previous run.")

    parser.add_argument("--propagate-features",
        dest="feat_prop",
        type=int,
        default=None,
        required=False,
        help="Propagates feature instead of using anonymous features")

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

    parser.add_argument("--gradient-noise",
        dest="gradient_noise",
        type=float,
        default=0.0,
        help="If provided, adds gradient noise scaled by GRADIENT_NOISE")

    parser.add_argument("--valid-pct",
        dest='valid_pct',
        type=float,
        default=0.1,
        help="Percentage of nodes to use for validation.")

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

    parser.add_argument("--hard-negatives",
        dest="hard_negatives",
        type=int,
        default=None,
        help="If provided, samples hard negatives from the graph")

    parser.add_argument("--min-feature-count",
        dest="min_feature_count",
        type=int,
        default=1,
        help="If set, filters out features which have fewer than min_count.")

    parser.add_argument("--alpha",
        dest="alpha",
        type=float,
        default=None,
        help="If provided, uses weighted embeddings.")

    parser.add_argument("--attention",
        dest="attention",
        type=int,
        default=None,
        help="If provided, uses self attention with D dims.")

    parser.add_argument("--attention-heads",
        dest="attention_heads",
        type=int,
        default=1,
        help="If attention is used, number of heads.  Default is 1")

    parser.add_argument("--context-window",
        dest="context_window",
        type=int,
        default=None,
        help="If provided, uses self attention with local window size of CONTEXT_WINDOW * 2.")

    parser.add_argument("--neighborhood-alignment",
        dest="neighborhood_alignment",
        type=float,
        default=None,
        help="If provided, applies neighborhood alignment to the embeddings.")

    parser.add_argument("--full-features",
        dest="full_features",
        default=None,
        help="If provided, embeds features from the given file instead of training set.")

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

    group.add_argument("--rank",
        dest="rank",
        nargs=2,
        metavar=('MARGIN', 'NEGATIVES'),
        help="Performs rank loss on the items.")

    group.add_argument("--ppr",
        dest="ppr",
        nargs=3,
        metavar=('MARGIN', 'EXAMPLES', 'RESTART_PROB'),
        help="Optimizes uses Personalized Page Rank Sampling")

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
    graph = cloverleaf.Graph.load(g_name, cloverleaf.EdgeType.Undirected)
    print("Nodes={},Edges={}".format(graph.nodes(), graph.edges()), file=sys.stderr)
    print('Loading features...')
    features = cloverleaf.FeatureSet.new_from_graph(graph)
    if f_name != 'none':
        features.load_into(f_name)
        if args.min_feature_count > 1:
            print("Pruning features: Original {}".format(features.num_features()))
            features = features.prune_min_count(args.min_feature_count)

        if args.feat_prop is not None:
            print("Propagating features")
            fp = cloverleaf.FeaturePropagator(args.feat_prop)
            fp.propagate(graph, features)

    print("Unique Features found: {}".format(features.num_features()))
    sTime = time.time()

    if args.ep is not None:
        margin, negatives = args.ep
        loss = cloverleaf.EPLoss.margin(float(margin), int(negatives))
    elif args.starspace is not None:
        margin, negatives = args.starspace
        loss = cloverleaf.EPLoss.starspace(float(margin), int(negatives))
    elif args.ppr is not None:
        temp, negs, p = args.ppr
        loss = cloverleaf.EPLoss.ppr(float(temp), int(negs), float(p))
    elif args.rank is not None:
        tau, negs = args.rank
        loss = cloverleaf.EPLoss.rank(float(tau), int(negs))
    else:
        temp, negs = args.contrastive
        loss = cloverleaf.EPLoss.contrastive(float(temp), int(negs))

    ep = cloverleaf.EmbeddingPropagator(
        alpha=args.lr, loss=loss, batch_size=args.batch_size, dims=args.dims, 
        passes=args.passes, max_nodes=args.max_neighbors, 
        max_features=args.max_features, attention=args.attention, 
        attention_heads=args.attention_heads, context_window=args.context_window, 
        noise=args.gradient_noise, hard_negatives=args.hard_negatives, valid_pct=args.valid_pct)

    if args.warm_start is not None:
        feature_embeddings = cloverleaf.NodeEmbeddings.load(args.warm_start, cloverleaf.Distance.Cosine)
    else:
        feature_embeddings = None

    if args.passes > 0 or feature_embeddings is None:
        feature_embeddings = ep.learn_features(graph, features, feature_embeddings)

    eTime = time.time() - sTime

    print("Time to learn:{}, Nodes/sec:{}".format(eTime, (graph.nodes() * 50) / eTime, file=sys.stderr))
    feature_embeddings.save(args.output + '.feature-embeddings')

    print("Constructing nodes...")
    if args.attention is not None:
        aggregator = cloverleaf.FeatureAggregator.Attention(args.attention_heads, args.attention, args.context_window)
    elif args.alpha is not None:
        aggregator = cloverleaf.FeatureAggregator.Weighted(args.alpha, features)
    else:
        aggregator = cloverleaf.FeatureAggregator.Averaged()

    embedder = cloverleaf.NodeEmbedder(aggregator)
    if args.full_features:
        features = cloverleaf.FeatureSet.new_from_file(args.full_features)

    node_embeddings = embedder.embed_feature_set(features, feature_embeddings)

    if args.neighborhood_alignment is None:
        node_embeddings.save(args.output + '.node-embeddings')
    else:
        node_embeddings.save(args.output + '.node-embeddings.orig')
        aligner = cloverleaf.NeighborhoodAligner(args.neighborhood_alignment)
        aligner.align_to_disk(args.output + '.node-embeddings', node_embeddings, graph)

    aggregator.save(args.output + '.embedder')

if __name__ == '__main__':
    main(build_arg_parser().parse_args())
