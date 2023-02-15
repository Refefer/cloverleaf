import sys
import requests
import argparse

from flask import Flask, request, jsonify

import cloverleaf

app = Flask(__name__)
app.config.from_object(__name__)

@app.route('/query', methods=['GET', 'POST'])
def query():
    query = request.args["query"]
    k     = int(request.args.get("k", 100))
    alpha = request.args.get("alpha")
    if alpha is not None:
        alpha = float(alpha)

    filter_type = request.args.get("filter_node_type", None)

    emb = construct_adhoc_embedding(query, FE_EMBEDDINGS, AGGREGATOR, alpha=alpha)
    top_k = NE_EMBEDDINGS.nearest_neighbor(emb, k, filter_type)
    return jsonify({"results": top_k})

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

    e = AGGREGATOR.embed_adhoc(tokens, embeddings, alpha=alpha, strict=False)
    if sum(e) == 0:
        return None

    print(e)

    return e

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Wires up cloverleaf to a simple API.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("node-embeddings",
            help="Path to node embeddings file")

    parser.add_argument("feature-embeddings",
            help="Path to feature embeddings file")

    parser.add_argument("embedder",
            help="Path to embedder file")

    parser.add_argument("--host",
            dest="host",
            default="127.0.0.1",
            help="Host IP Address.  Default is 127.0.0.1")

    parser.add_argument("--port",
            dest="port",
            default=5000,
            type=int,
            help="Host Port Address.  Default is 5000")

    return parser

def load(ne_fname, fe_fname, agg_fname):
    print("Loading Node Embeddings...")
    ne_embeddings = cloverleaf.NodeEmbeddings.load(ne_fname, cloverleaf.Distance.Euclidean)
    print("Loading Feature Embeddings...")
    fe_embeddings = cloverleaf.NodeEmbeddings.load(fe_fname, cloverleaf.Distance.Euclidean)
    print("Loading aggregator")
    aggregator = cloverleaf.FeatureEmbeddingAggregator.load(agg_fname)
    
    return ne_embeddings, fe_embeddings, aggregator

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    NE_EMBEDDINGS, FE_EMBEDDINGS, AGGREGATOR = load(args.__dict__['node-embeddings'], 
                                                    args.__dict__['feature-embeddings'], 
                                                    args.embedder)
    app.run(host=args.host, port=args.port)
