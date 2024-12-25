import os
import sys
import requests
import argparse
from flask import Flask, request, jsonify
import cloverleaf
from pydgraph import DgraphClient, DgraphClientStub

app = Flask(__name__)
app.config.from_object(__name__)

# Global Dgraph client
client = None

# Initialize Dgraph client
def init_dgraph_client():
    global client
    assert "DGRAPH_GRPC" in os.environ, "DGRAPH_GRPC must be defined"
    dgraph_grpc = os.environ["DGRAPH_GRPC"]
    
    if "cloud.dgraph" in dgraph_grpc:
        assert "DGRAPH_ADMIN_KEY" in os.environ, "DGRAPH_ADMIN_KEY must be defined"
        APIAdminKey = os.environ["DGRAPH_ADMIN_KEY"]
    else:
        APIAdminKey = None

    if APIAdminKey is None:
        print("Using no API key")
    else:
        print("Using cloud API key")

    print(dgraph_grpc)

    # Initialize the Dgraph client
    if APIAdminKey:
        stub = DgraphClientStub(f"{dgraph_grpc}:9080", api_key=APIAdminKey)
    else:
        stub = DgraphClientStub(f"{dgraph_grpc}:9080")
    
    client = DgraphClient(stub)

# Unified function to load embeddings (from file or Dgraph)
def load_embeddings(ne_fname=None, fe_fname=None, agg_fname=None, from_dgraph=False):
    if from_dgraph:
        # Query Dgraph to load embeddings
        txn = client.txn(read_only=True)
        try:
            dgraph_query = """
            {
                all(func: has(embedding)) {
                    uid
                    embedding
                    // Add other fields you want to retrieve
                }
            }
            """
            response = txn.query(dgraph_query)
            results = response.json.get('all', [])
            embeddings = {}
            for result in results:
                uid = result['uid']
                embedding = result.get('embedding', [])
                embeddings[uid] = embedding
            return embeddings
        finally:
            txn.discard()
    
    # If not from Dgraph, load embeddings from files
    print("Loading Node Embeddings from files...")
    ne_embeddings = cloverleaf.NodeEmbeddings.load(ne_fname, cloverleaf.Distance.Cosine)
    
    print("Loading Feature Embeddings from files...")
    fe_embeddings = cloverleaf.NodeEmbeddings.load(fe_fname, cloverleaf.Distance.Cosine, 'feat')
    
    print("Loading aggregator from files...")
    aggregator = cloverleaf.FeatureAggregator.load(agg_fname)

    return ne_embeddings, fe_embeddings, aggregator

@app.route('/query', methods=['GET', 'POST'])
def query():
    query = request.args["query"]
    k = int(request.args.get("k", 100))
    alpha = request.args.get("alpha")
    if alpha is not None:
        alpha = float(alpha)

    filter_type = request.args.get("filter_node_type", None)

    # Query Dgraph for relevant data
    dgraph_results = query_dgraph(query)

    # Check if Dgraph returned any results
    if not dgraph_results:
        return jsonify({"error": "No results found in Dgraph for the given query."}), 404

    # Use Dgraph results to construct embeddings
    emb = construct_adhoc_embedding(dgraph_results, FE_EMBEDDINGS, AGGREGATOR, alpha=alpha)

    # Check if the embedding is valid
    if emb is None or sum(emb) == 0:
        return jsonify({"error": "Failed to construct a valid embedding."}), 400

    top_k = NE_EMBEDDINGS.nearest_neighbor(emb, k, filter_type)
    return jsonify({"results": top_k})

def query_dgraph(query):
    # Create a new transaction
    txn = client.txn(read_only=True)
    try:
        # Construct a Dgraph query
        dgraph_query = f"""
        {{
            all(func: has(name)) @filter(eq(name , "{query}")) {{
                uid
                name
                // Add other fields you want to retrieve
            }}
        }}
        """
        response = txn.query(dgraph_query)
        results = response.json.get('all', [])
        return results
    finally:
        txn.discard()

def build_grams(query):
    pieces = query.split()
    bow = []
    for g in (2, 3):
        for i in range(len(pieces) - g + 1):
            bow.append('_'.join(pieces[i:i + g]))

    bow.extend(pieces)

    print("Terms:", bow)
    return bow

def construct_adhoc_embedding(dgraph_results, embeddings, aggregator, alpha=None):
    tokens = []
    for result in dgraph_results:
        uid = result['uid']
        if uid in embeddings:
            tokens.append(embeddings[uid])
    
    if not tokens:
        return None

    # Aggregate the embeddings
    combined_embedding = aggregator.aggregate(tokens)
    
    if alpha is not None:
        combined_embedding = [x * alpha for x in combined_embedding]

    return combined_embedding

if __name__ == '__main__':
    init_dgraph_client()

    # Decide whether to load embeddings from Dgraph or from files
    use_dgraph = os.getenv("USE_DGRAPH", "false").lower() == "true"  # Use an environment variable to toggle this
    ne_fname = "path/to/node_embeddings"  # Specify the path to node embeddings
    fe_fname = "path/to/feature_embeddings"  # Specify the path to feature embeddings
    agg_fname = "path/to/aggregator"  # Specify the path to the aggregator

    NE_EMBEDDINGS, FE_EMBEDDINGS, AGGREGATOR = load_embeddings(
        ne_fname if not use_dgraph else None, 
        fe_fname if not use_dgraph else None, 
        agg_fname if not use_dgraph else None, 
        from_dgraph=use_dgraph
    )

    app.run(host='0.0.0.0', port=5000)

