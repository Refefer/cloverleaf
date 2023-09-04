Cloverleaf
====

Cloverleaf is a high performance graph library which provides a number of different optimizers for a variety of different usecases.  The library allows practioners to mine graphs for useful information through a variety of techniques, scalable to hundreds of millions of nodes and billions of edges (given sufficent hardware).  A key goal of the library is to avoid GPU computing while scaling to large scale graphs, enabling less well resourced teams to learn effective representations on industry scale graphs.  While it offers a suite of established tools, it also serves as a testing ground for new approaches in graph learning aligned with those needs.

The library is broken up into a few different methods:

1. Graph Clustering
    - Label Propagation Algorithm (with parallel extensions), which fits each graph node into a single cluster.
    - Speaker-Listener Propagation Algorithm (Allows for multiple clusters).
2. Graph Embedding
    - Walk Distance Embeddings
    - Vector Propagation on Click Graphs (VPCG)
    - Embedding Propagation with several novel extensions
3. Random Walks
    - Random Walks with Restarts
    - Guided Random Walks with Restarts
4. Utilities
    - Approximate Nearest Neighbors (ANN) via Random Projections
    - Neighborhood Alignment

##Data Format


Graphs can be defined either adhoc or loaded from files.  Notably, Cloverleaf does _not_ allow for live graph updating currently; this allows it to make a number of optimizations to maximize the memory and compute efficiency of the graph.

###Adhoc Graph Definition
===

Cloverleaf provides a GraphBuilder constructor for adhoc building of graphs.  The function `add_edge` adds an edge between two nodes.  Nodes are defined by two attributes: the node_type, and the node_name, which together represent a unique node within the Cloverleaf graph.

#### Parameters
    1. `from_node` - Fully qualified node tuple - (node_type, node_name)
    2. `to_node` - Fully qualified node tuple - (node_type, node_name)
    3. `edge_weight` - Edge weight; used in a variety of algorithms.
    4. `edge_type` - Directed or Undirected.  Cloverleaf internally stores edges as directed; setting the edge_type as Undirected will create two directed edges between the two nodes.

```python3
>>> import cloverleaf
>>> gb = cloverleaf.GraphBuilder()
>>> gb.add_edge(('query', 'red sox'), ('sku', '15715'), 1, cloverleaf.EdgeType.Undirected)
>>> graph = gb.build_graph()
```

###Load Graph From Disk

This is the prefered method of loading graphs into Cloverleaf.  The file format is simple, using TSVs for specifying the graphs:

Edges file:

```
From_Node_Type<TAB>From_Node_Name<TAB>To_Node_Type<TAB>To_Node_Name<TAB>Edge_Weight\n
```

To load a graph from disk, the `load()` method is invoked:

```python3
>>> import cloverleaf
>>> graph = cloverleaf.Graph.load("karate.edges", cloverleaf.EdgeType.Undirected)
Read 34 nodes, 156 edges...
```

###Loading Features

Some algorithms use node features as part of the optimization step.  FeatureSets can be instantiated in two ways:

1. Loaded from a file.
2. Instantiated from a defined graph.

It is preferable to instantiate a FeatureSet from a graph for memory efficiency:

```python3
>>> fs = cloverleaf.FeatureSet.new_from_graph(graph)
>>> fs.get_features(('node', '1'))
[]
>>> fs.set_features(('node', '1'), ['hello', 'world'])
>>> fs.get_features(('node', '1'))
['hello', 'world']
```

After instantiating, the `load_into()` can be called to add features to all nodes:

Features file:

```
Node_Type<TAB>Node_Name<TAB>feat1 feat2 ...\n
```

Invoked with:

```python3
>>> fs.load_into("path")
```

Algorithms In Detail
---

### Label Propagation Algorithm

This is a classic method for computing node cluster based on graph topology.  It's best used for cases where homophily is of predominant interest; that is, walk distance has bearing on the node similarity.

Cloverleaf implements the algorithm with a few additional features.  First, it allows running the algorithm multiple times with different seeds to create a cluster embedding (with Hamming Distance as the metric), which is controlled by `k`.  It also allows setting of max passes and setting a random seed.

The embedder returns a NodeEmbeddings object.

#### Parameters
    1. `k` - Number of times to rerun LPA
    2. `passes` - Number of iterations to run each LPA.
    3. `seed` - Random seed to use.

#### Example

```python3
>>> import cloverleaf
>>> graph = cloverleaf.Graph.load("karate.edges", cloverleaf.EdgeType.Undirected)
>>> lpa_embedder = cloverleaf.ClusterLPAEmbedder(k=5, passes=20)
>>> embs = lpa_embedder.learn(graph)
k=5,passes=20,seed=20222022
Finished 1/5
Finished 2/5
Finished 3/5
Finished 4/5
Finished 5/5
>>> embs.get_embedding(('node', '12'))
[1.0, 1.0, 5.0, 1.0, 15.0]
```

### Speaker-Listener Propagation Algorithm

Unlike LPA, which assigns a single cluster id to each node in the graph, SLPA allows for a node to occupy different numbers of clusters.  Similarity is given as Jaccard.  A cluster ID of -1 is used as a sentinel value indicating that no cluster id occupies that slot (which will be influenced by the threshold parameter)

#### Parameters
1. `k` - Number of overlapping clusters a node is allowed to occupy
2. `threshold` - Filters out communities from a node with less than K.
3. `seed` - Random seed to use.


#### Example

```python3
>>> import cloverleaf
>>> graph = cloverleaf.Graph.load("karate.edges", cloverleaf.EdgeType.Undirected)
>>> slpa = cloverleaf.SLPAEmbedder(k=5, threshold=0.2)
>>> embs = slpa.learn(graph)
>>> embs.get_embedding(('node', '12'))
[1.0, 11.0, 13.0, -1.0, -1.0, -1.0]
```

### Walk Distance Embeddings

Walk Distance Embeddings create embeddings by learning a walk distance, that is the minimum number of edges between two nodes, and a set of landmark nodes.  Importantly, this requires a fully connected graph - disconnected components will not work with this algorithm.  It is fast, deterministic, and produces reasonably good embeddings for the compute.  Returns NodeEmbeddings with the distance metric set to ALT (triangular inequality).

#### Parameters
1. `n_landmarks` - Number of to use for computing distance
2. `seed` - Random seed to use.  If provided, randomly samples landmark nodes; otherwise selects landmarks by highest degree count


#### Example

```python3
>>> dist_embedder = cloverleaf.DistanceEmbedder(n_landmarks=5)
>>> embs = dist_embedder.learn(graph)
>>> embs.get_embedding(('node', '12'))
[2.0, 1.0, 2.0, 3.0, 3.0]
```

### Vector Propagation on Click Graphs (VPCG)

VPCG is a node feature based embedding learning, applicable to bipartite graphs.  Starting with discrete features from one side of the graph, it propagates the features to the other side in an iterative fashion, diffusing common terms across related nodes, learning a set of terms and associated weights with each.  After the optimization step is completed, it computes a hash embedding (in the vein of Vowpal Wabbit), collapsing the sparse embedding into a dense embedding. 

This is a great baseline for both Search and Recommendation applications, with User <> Product and Query <> Product graphs respectably.

#### Parameters
1. `max_terms` - Number of discrete features, typically words or tokens, to track per node.  The more tokens tracked, the higher fidelity the embedding at the expense of memory.
2. `passes` - Number of times to propagate features.  Too many passes will overfit the dataset for transductive usecases, with passes=10 or 20 being reasonable starting points.
3. `dims` - Size of the dense embedding to hash the sparse VPCG terms to.

#### Example

```python3
>>> graph = cloverleaf.Graph.load("graph.edges", cloverleaf.EdgeType.Undirected)
>>> node_features = cloverleaf.FeatureSet.new_from_graph(graph)
>>> node_features.load_into("graph.features")
>>> vpcg_embedder = cloverleaf.VpcgEmbedder(max_terms=100, passes=10, dims=256)
>>> embs = vpcg_embedder.learn(graph, node_features, "query")
```

### Embedding Propagation

Embedding Propagation is a node embedding method which uses both the graph structure and node features to learn node embeddings.  As with the original paper, EP-B is implemented as a baseline.  In addition, Cloverleaf offers a variety of useful extensions:

1. Multiple losses - EP, Starspace, Rankspace, Contrastive, PPR, and more
2. Average or Attention-based node feature aggregation with Full, Random, and sliding window approaches (useful for textual node features).
3. Neighborhood Alignment - Uses graph topology to influence node feature-based embeddings.

To maximize cpu utilization, Hogwild [8] is used to update weights lock free.  This has the side effect of non-determinism in runs as well; similarly, if collisions between features are likely, the fidelity of the embeddings will be impacted.  In practice it's a non-issue but worth paying attention to, especially if reproducability is critical.

The result of the optimization phase is feature embeddings, which can be used to construct node embeddings dynamically or in batch.

#### Parameters
1. `alpha` - Learning rate for the optimization step.
2. `loss` - one of cloverleaf.EPLoss - margin, contrastive, starspace, rank, rankspace, ppr.
3. `batch_size` - Number of examples to use per update step.
4. `dims` - Dimension size of each feature.
5. `passes` - Number of epochs to train the feature embeddings.
6. `max_nodes` - Maximum number of neighbor nodes, randomly sampled, used to reconstruct the node embedding.  Higher numbers of nodes potentially increase fidelity at the expense of compute time.
7. `max_features` - Maximum number of features per node, randomly sampled, to use for aggregation.  Higher numbers of features increase fidelity at the cost of run time.
8. `attention` - Number of dimensions to use for the attention vector for each feature.  The higher the dimension, the more memory is needed.  Warning: attention is extremely computational expensive and reason needs to be used on the size of a graph to optimize for.
9. `attention_heads` - Number of different heads to use for attention.  The higher the heads, the greater the fidelity.
10. `context_window` - Number of features to consider on each side of the pivot; useful for reducing the complexity of attention computation while preserving locality.
11. `noise` - Adds gradient noise to the updates; likely to be removed in the future.
12. `hard_negatives` - Computes hard negatives through a fixed random walk from the anchor node, providing a similar but non-neighbor to the node.
13. `valid_pct` - Percentage of nodes in the graph to use for validation.

#### Example

Due to the extensive number of options, the reader is encouraged to read `scripts/learn.py` which provides a convenient entry into the Embedding Propagation methods.


### Random Walk with Restarts
Random Walks with Restarts is an algorithm which estimates the stationary distribution from a given node, returning the top K most highest weighted nodes with respect to starting context.  

#### Parameters
1. `walks` - Number of random walks to perform.  The larger the number, the more accurate the estimation with the expense of latency.
2. `restarts` - Float defining the termination criteria.  When `restarts` is between (0,1), walk termination is probabilistic.  When `restarts` >=1, it is interpreted as a fixed number of steps (e.g. restarts=3 would indicate walks should be terminated after three steps). 
3. `beta` - Damping parameter controlling how much to bias toward popular vs rare items. See [6] for more details.

#### Example

```python3
>>> rwr = cloverleaf.RandomWalker(walks=100_000, restarts=1/3, beta=0.8)
>>> rwr.walk(graph, ('node', '1'), k=5)
[(('node', '12'), 0.026669999584555626), (('node', '13'), 0.019355567172169685), (('node', '11'), 0.018968328833580017), (('node', '18'), 0.018936293199658394), (('node', '5'), 0.018457578495144844)]
```

Install
---

1. Setup a new python virtualenv
2. pip install maturin numpy
3. Ensure you have the latest Rust compiler
4. RUSTFLAGS="-C target-cpu=native" maturin develop --release
5. Profit!

TODO
---

1. Lots of documentation still needed
2. Lots of tests still needed :)
3. Build examples for each of the methods currently available
4. Algos: Power Iteration based PageRank optimizer

References
---

1. ZhuЃ, Xiaojin, and Zoubin GhahramaniЃн. "Learning from labeled and unlabeled data with label propagation." ProQuest Number: INFORMATION TO ALL USERS (2002).
2. Xie, Jierui, Boleslaw K. Szymanski, and Xiaoming Liu. "Slpa: Uncovering overlapping communities in social networks via a speaker-listener interaction dynamic process." 2011 ieee 11th international conference on data mining workshops. IEEE, 2011.
3. Goldberg, Andrew V., and Chris Harrelson. "Computing the shortest path: A search meets graph theory." SODA. Vol. 5. 2005.
4. Jiang, Shan, et al. "Learning query and document relevance from a web-scale click graph." Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval. 2016.
5. Garcia Duran, Alberto, and Mathias Niepert. "Learning graph representations with embedding propagation." Advances in neural information processing systems 30 (2017).
6. Christoffel, Fabian, et al. "Blockbusters and wallflowers: Accurate, diverse, and scalable recommendations with random walks." Proceedings of the 9th ACM Conference on Recommender Systems. 2015.
7. Eksombatchai, Chantat, et al. "Pixie: A system for recommending 3+ billion items to 200+ million users in real-time." Proceedings of the 2018 world wide web conference. 2018.
8. Recht, Benjamin, et al. "Hogwild!: A lock-free approach to parallelizing stochastic gradient descent." Advances in neural information processing systems 24 (2011).
