import traceback
import sys
import numpy as np
import json
import argparse

import tabulate
import cloverleaf
from sklearn.utils import murmurhash3_32

def build_grams(query):
    pieces = query.split()
    bow = []
    for g in (2,3):
        for i in range(len(pieces) - g + 1):
            bow.append('_'.join(pieces[i:i+g]))

    bow.extend(pieces)

    print("Terms:", bow)
    return bow

def construct_adhoc_embedding(query, embeddings, aggregator, alpha=None):
    tokens = []
    for token in build_grams(query):
        tokens.append(('feat', token))

    e = aggregator.embed_adhoc(tokens, embeddings, alpha=alpha, strict=False)
    if sum(e) == 0:
        return None

    return e

def build_queries(queries):
    tokens = queries.strip().split('\t')
    qs = []
    for i in range(0, len(tokens), 2):
        qs.append((tokens[i], tokens[i+1]))

    return qs

def build_embedding(queries, embeddings):
    print(queries)
    e = []
    for node_type, name in queries:
        e.append(embeddings.get_embedding((node_type, name)))

    print(e)
    if len(e) == 1:
        return e[0]

    return np.mean(np.array(e), axis=0)

def load(args):
    ne_fname = args.model + '.node-embeddings'
    fe_fname = args.model + '.feature-embeddings'
    agg_fname = args.model + '.embedder'

    print("Loading Node Embeddings...")
    ne_embeddings = cloverleaf.NodeEmbeddings.load(ne_fname, cloverleaf.Distance.Cosine, args.filter_type)
    print("Loading Feature Embeddings...")
    fe_embeddings = cloverleaf.NodeEmbeddings.load(fe_fname, cloverleaf.Distance.Cosine)
    print("Loading aggregator")
    aggregator = cloverleaf.FeatureEmbeddingAggregator.load(agg_fname)
    
    return ne_embeddings, fe_embeddings, aggregator

def main(args):
    ne_embeddings, fe_embeddings, aggregator = load(args)

    headers = ('type', 'Name', 'Score')
    while True:
        query = input(">")
        try:
            if query.startswith('*'):
                tokens = [('feat', t) for t in query[1:].strip().split()]
                emb = aggregator.embed_adhoc(tokens, fe_embeddings, alpha=args.alpha, strict=True)
            elif '\t' not in query:
                emb = construct_adhoc_embedding(query, fe_embeddings, aggregator, alpha=args.alpha)
                if emb is None:
                    print("Tokens not found in feature embeddings!")
                    continue

            else:
                queries = build_queries(query)
                emb = build_embedding(queries, ne_embeddings)

            print()
            print(emb)

            top_k = ne_embeddings.nearest_neighbor(emb, args.k)

            rows = []
            for (node_type, node), score in reversed(top_k):
                rows.append((node_type, node, score))

            print(tabulate.tabulate(rows, headers=headers))

        except Exception as e:
            print('Unable to run!')
            raise

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Query Cloverleaf embeddings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("model",
        help="Path to model prefix.")

    parser.add_argument("--k",
        dest="k",
        type=int,
        default=50,
        help="Max number of results to return")

    parser.add_argument("--alpha",
        dest="alpha",
        type=float,
        default=None,
        help="Alpha to use for weighted embeddings.")

    parser.add_argument("--filter-type",
        dest="filter_type",
        default=None,
        help="If provided, only searches the provided node type.")

    return parser

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    main(args)
