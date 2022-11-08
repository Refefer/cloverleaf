import traceback
import sys
import numpy as np
import json

import tabulate
import cloverleaf
from sklearn.utils import murmurhash3_32

K = 50
CHAR_GRAMS = False
ALPHA = 1e-3 

def char_grams(token, grams=(3,6), hash_space=50001):
    t = '^{}$'.format(token)
    gs = set([t])
    for cg in grams:
        for window in range(len(t) - cg + 1):
            idx = murmurhash3_32(t[window:window+cg], positive=True)
            gs.add('cg:{}'.format(idx % hash_space))

    return list(gs)

def build_grams(query):
    pieces = query.split()
    bow = []
    for g in (2,3):
        for i in range(len(pieces) - g + 1):
            bow.append('_'.join(pieces[i:i+g]))

    if CHAR_GRAMS:
        for token in pieces:
            bow.extend(char_grams(token))
    else:
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

def main(ne_fname, fe_fname, agg_fname):
    print("Loading Node Embeddings...")
    ne_embeddings = cloverleaf.NodeEmbeddings.load(ne_fname, cloverleaf.Distance.Cosine)
    print("Loading Feature Embeddings...")
    fe_embeddings = cloverleaf.NodeEmbeddings.load(fe_fname, cloverleaf.Distance.Cosine)
    print("Loading aggregator")
    aggregator = cloverleaf.FeatureEmbeddingAggregator.load(agg_fname)

    headers = ('type', 'Name', 'Score')
    while True:
        query = input(">")
        try:
            if query.startswith('*'):
                tokens = [('feat', t) for t in query[1:].strip().split()]
                emb = aggregator.embed_adhoc(tokens, fe_embeddings, alpha=ALPHA, strict=True)
            elif '\t' not in query:
                if fe_fname is not None:
                    emb = construct_adhoc_embedding(query, fe_embeddings, aggregator, alpha=ALPHA)
                    if emb is None:
                        print("Tokens not found in feature embeddings!")
                        continue
                else:
                    print("feature embeddings not available!")
                    continue

            else:
                queries = build_queries(query)
                emb = build_embedding(queries, ne_embeddings)

            print()
            print(emb)

            top_k = ne_embeddings.nearest_neighbor(emb, K)

            rows = []
            for (node_type, node), score in reversed(top_k):
                rows.append((node_type, node, score))

            print(tabulate.tabulate(rows, headers=headers))

        except Exception as e:
            traceback.print_exception(e)
            continue

if __name__ == '__main__':
    node_embeddings = sys.argv[1]
    feature_embeddings = sys.argv[2]
    embedder = sys.argv[3]

    main(node_embeddings, feature_embeddings, embedder)
