"""End to end: train EP on the GPU, then embed and retrieve for a query the graph has never seen,
with cloverleaf doing everything after training.

The data is synthetic and built here: products with short titles and queries of two words, both
drawn from per-topic vocabularies, and click edges from each query to products of its topic.  A
held-out query is embedded from its words alone (inductively), then aligned and used to retrieve
products, the way a search system would treat new traffic.

    python quickstart.py [--passes 20] [--out DIR]

Runs on a GPU when there is one, otherwise on the CPU in about a minute.  Needs torch, numpy and
cloverleaf (see the repository README to build it).
"""
import argparse
import os
import random
import sys
import tempfile

import numpy as np

import cloverleaf as cl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ep_gpu import data  # noqa: E402
from ep_gpu.train import Config, export, train  # noqa: E402

TOPICS, WORDS_PER_TOPIC, PRODUCTS, QUERIES = 20, 15, 1500, 3000


def make_data(directory, seed=0):
    """Writes the graph and feature files; returns each product's topic and a held-out query."""
    rng = random.Random(seed)
    vocab = [[f"t{t}w{i}" for i in range(WORDS_PER_TOPIC)] for t in range(TOPICS)]
    product_topic = {f"p{i}": rng.randrange(TOPICS) for i in range(PRODUCTS)}
    by_topic = [[p for p, t in product_topic.items() if t == topic] for topic in range(TOPICS)]
    edges, features = [], []
    for p, t in product_topic.items():
        title = rng.sample(vocab[t], 4) + [f"brand{rng.randrange(50)}"]
        features.append(f"product\t{p}\t{' '.join(title)}\n")
    for i in range(QUERIES):
        t = rng.randrange(TOPICS)
        words = rng.sample(vocab[t], 2)
        features.append(f"query\tq{i}\t{' '.join(words)}\n")
        for p in rng.sample(by_topic[t], 3):
            edges.append(f"query\tq{i}\tproduct\t{p}\t{rng.randint(1, 5)}\n")
    edges_path, features_path = os.path.join(directory, "graph.edges"), os.path.join(directory, "graph.features")
    with open(edges_path, "w") as f:
        f.writelines(edges)
    with open(features_path, "w") as f:
        f.writelines(features)
    new_topic = 7
    new_query = [("query", w) for w in rng.sample(vocab[new_topic], 2)]
    return edges_path, features_path, product_topic, new_query, new_topic


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--passes", type=int, default=20)
    parser.add_argument("--out", default=None, help="Directory for the files (default: a temporary one).")
    args = parser.parse_args()
    directory = args.out or tempfile.mkdtemp(prefix="ep-gpu-quickstart-")
    os.makedirs(directory, exist_ok=True)

    edges, features, product_topic, new_query, new_topic = make_data(directory)
    graph = data.load(edges, features)
    print(f"{graph.num_nodes} nodes, {graph.num_features} features, in {directory}")

    # The recommended configuration: attention pooling with mean-combined neighbours, StarSpace,
    # row-wise Adagrad with a bfloat16 table, and the sawtooth learning rate.
    cfg = Config(dims=32, passes=args.passes, batch_size=256, lr=0.3, loss="starspace", gamma=0.1,
                 negatives=12, max_nodes=12, max_features=0.5, weighted_averaging=True,
                 pooling="attention", heads=4, neighbour_pooling="mean",
                 optimizer="rowwise_adagrad", table_dtype="bfloat16",
                 lr_schedule="sawtooth", lr_cycles=4, valid_pct=0.05)
    model = train(graph, cfg, log=lambda m: print(" ", m) if m.startswith("pass") and
                  int(m.split()[1].split("/")[0]) % 5 == 0 else None)
    prefix = os.path.join(directory, "ep")
    export(model, graph, cfg, prefix)

    # Everything from here is cloverleaf: the exported feature table and pooling embed the new
    # query exactly as the trained model would.
    feature_embeddings = cl.NodeEmbeddings.load(prefix + ".feature-embeddings", cl.Distance.Cosine)
    nodes = cl.NodeEmbeddings.load(prefix + ".node-embeddings", cl.Distance.Cosine)
    embedder = cl.NodeEmbedder(cl.FeatureAggregator.load(prefix + ".aggregator"))
    query = embedder.embed_adhoc(new_query, feature_embeddings, strict=False)
    print(f"new query {' '.join(w for _, w in new_query)!r} (topic {new_topic}), embedded by {embedder}")

    # Neighbourhood alignment pulls each node towards its neighbours.  The new query has no edges,
    # so it moves as its nearest training queries did (EmbeddingAligner's mean shift).
    cl_graph = cl.Graph.load(edges, cl.EdgeType.Undirected)
    aligned = cl.NeighborhoodAligner(0.5).align(nodes, cl_graph)
    queries = cl.NodeEmbeddings.new_from_list(
        [(n, nodes.get_embedding(n)) for n in nodes.vocab() if n[0] == "query"], cl.Distance.Cosine)
    ann = cl.EmbAnn(queries, 10, 20, seed=1)
    shifted = cl.EmbeddingAligner(10, method="mean_shift").align(
        queries, ann, aligned, cl.Query.embedding(query), seed=1)

    products = [n for n in nodes.vocab() if n[0] == "product"]
    for label, q, space in (("unaligned", query, nodes), ("aligned", shifted, aligned)):
        matrix = np.array([space.get_embedding(p) for p in products], np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        scores = matrix @ (np.asarray(q, np.float32) / np.linalg.norm(q))
        top = [products[i][1] for i in np.argsort(-scores)[:10]]
        on_topic = sum(product_topic[p] == new_topic for p in top)
        print(f"  {label}: {on_topic}/10 of the top products are on the query's topic")


if __name__ == "__main__":
    main()
