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

    parser.add_argument("--valid-pct",
        dest='valid_pct',
        type=float,
        default=0.1,
        help="Percentage of nodes to use for validation.")

    parser.add_argument("--batch-size",
        type=int,
        default=128,
        help="Batch size.")

    parser.add_argument("--min-feature-count",
        dest="min_feature_count",
        type=int,
        default=1,
        help="If set, filters out features which have fewer than min_count.")

    parser.add_argument("--neighborhood-alignment",
        dest="neighborhood_alignment",
        type=float,
        default=None,
        help="If provided, applies neighborhood alignment to the embeddings.")

    parser.add_argument("--full-features",
        dest="full_features",
        default=None,
        help="If provided, embeds features from the given file instead of training set.")

    parser.add_argument("--negatives",
        dest="negatives",
        default=5,
        type=int,
        help="Number of random negatives to append to the list")

    parser.add_argument("--k",
        dest="k",
        default=20,
        type=int,
        help="Top items to gather")

    parser.add_argument("--steps",
        dest="steps",
        default=1/3,
        type=float,
        help="Restart probability for Random Walk")

    parser.add_argument("--walks",
        dest="walks",
        default=10_000,
        type=int,
        help="Number of random walks for each node to estimae PPR")

    parser.add_argument("--compression",
        dest="compression",
        default=1,
        type=float,
        help="How to compressing the L1 norm")

    parser.add_argument("--num-features",
        dest="num_features",
        default=None,
        type=int,
        help="Number of features to randomly select")

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

    ppr_learner = cloverleaf.PprRankLearner(
        alpha=args.lr,  batch_size=args.batch_size, dims=args.dims, 
        passes=args.passes, 
        negatives=args.negatives,
        walks=args.walks,
        steps=args.steps,
        k=args.k,
        compression=args.compression,
        num_features=args.num_features,
        valid_pct=args.valid_pct)

    if args.warm_start is not None:
        feature_embeddings = cloverleaf.NodeEmbeddings.load(args.warm_start, cloverleaf.Distance.Cosine)
    else:
        feature_embeddings = None

    if args.passes > 0 or feature_embeddings is None:
        feature_embeddings = ppr_learner.learn_features(graph, features, feature_embeddings)

    eTime = time.time() - sTime

    feature_embeddings.save(args.output + '.feature-embeddings')

    print("Constructing nodes...")
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
