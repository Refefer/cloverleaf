# Cloverleaf: High-Performance Graph Analytics Library - Comprehensive Technical Summary

## 1. Executive Overview

**Cloverleaf** is a high-performance, CPU-optimized graph analytics library written in **Rust** with **Python bindings** (via PyO3). It is designed to process massive-scale graphs containing hundreds of millions of nodes and billions of edges without requiring GPU hardware. The library focuses on graph clustering, embedding generation, random walks, and ranking algorithms, making it suitable for applications in recommendation systems, knowledge graphs, social network analysis, and click-through rate modeling.

### Key Design Principles
- **CPU-First Architecture**: Optimized for CPU execution to ensure accessibility for teams without GPU resources.
- **Static Graph Assumption**: Graphs are immutable after construction, enabling aggressive memory and compute optimizations.
- **Lock-Free Concurrency**: Uses the "Hogwild" pattern for parallel updates without locks.
- **Memory Efficiency**: Employs Compressed Sparse Row (CSR) formats and custom data structures to minimize memory footprint.
- **Modular Algorithm Design**: Algorithms are decoupled from graph representations via Rust traits.

---

## 2. Architecture & Core Components

### 2.1 Graph Representation (`src/graph.rs`)

Cloverleaf uses the **Compressed Sparse Row (CSR)** format as its primary internal representation. This choice is driven by memory efficiency and cache locality for read-heavy operations typical in graph algorithms.

#### Data Structures
```rust
pub struct CSR {
    rows: Vec<NodeID>,      // Row pointers (offsets)
    columns: Vec<NodeID>,   // Column indices (neighbors)
    weights: Vec<f32>       // Edge weights
}
```

#### Key Traits
- **`Graph`**: Defines read-only access to graph topology.
  - `len()`: Number of nodes.
  - `edges()`: Total number of edges.
  - `degree(idx)`: Out-degree of a node.
  - `get_edges(idx)`: Returns neighbor IDs and weights for a node.
  
- **`ModifiableGraph`**: Allows in-place edge updates (used during construction or specific algorithms).
  - `modify_edges(idx)`: Mutable access to edges/weights.

- **`ParallelModifiableGraph`**: Enables thread-safe concurrent modification via `rayon`.
  - `par_iter_mut()`: Parallel iterator yielding mutable slices per node.

- **`Transpose`**: Computes the transpose of the graph (reverses edge directions).

#### Construction & Optimization
- **Deduplication**: Parallel sorting and merging of duplicate edges with weight summation.
- **Cumulative Distributions**: Optional `CDFtoP` wrapper for efficient sampling from weighted edges.
- **Static Nature**: Once built, the graph is immutable, allowing algorithms to assume stable topology.

### 2.2 Embedding Store (`src/embeddings.rs`)

The `EmbeddingStore` manages vector representations for nodes or features. It supports multiple distance metrics and thread-safe concurrent writes.

#### Structure
```rust
pub struct EmbeddingStore {
    dims: usize,                  // Dimensionality of embeddings
    embeddings: Hogwild<Vec<f32>>, // Flattened vector of all embeddings
    bitfield: BitSet,             // Tracks initialized embeddings
    distance: Distance,           // Cosine, Hamming, Euclidean, etc.
    nodes: usize                  // Number of entities
}
```

#### Features
- **Distance Metrics**: Configurable at creation (Cosine similarity is default for many algorithms).
- **Hogwild Access**: `get_embedding_mut_hogwild()` allows lock-free concurrent writes from multiple threads.
- **Entity Abstraction**: Supports querying by `NodeID` or ad-hoc embedding vectors (`Entity::Embedding`).

### 2.3 Hogwild Concurrency (`src/hogwild.rs`)

A custom lock-free synchronization primitive enabling high-throughput parallel updates.

#### Implementation Details
```rust
pub struct Hogwild<T>(Arc<UnsafeCell<T>>);
```
- Uses `Arc` for shared ownership and `UnsafeCell` for interior mutability.
- **Safety Guarantee**: Safe only when concurrent writes do not target the same memory location (e.g., distinct nodes in an embedding update).
- **Use Cases**: SGD updates in embedding propagation, parallel cluster assignments in LPA.

### 2.4 Vocabulary & Mapping (`src/vocab.rs`)

Manages the mapping between human-readable node identifiers and internal integer IDs.

#### Functionality
- **Node ID Assignment**: Maps `(node_type, node_name)` tuples to unique `NodeID` (usize).
- **Reverse Lookup**: Retrieves original names from IDs for result formatting.
- **Efficiency**: Uses hash maps (`lasso` crate) for O(1) lookups.

### 2.5 Feature Store (`src/feature_store.rs`)

Stores sparse feature vectors associated with nodes, used in algorithms like VPCG and Embedding Propagation.

#### Structure
- Maps `NodeID` to a list of feature indices (sparse representation).
- Supports feature counting for IDF weighting.

### 2.6 Distance Metrics (`src/distance.rs`)

Defines distance/similarity functions used by embedding stores and ANN algorithms.
- **Cosine**: Dot product normalized by magnitude.
- **Hamming**: Count of differing bits (used for clustering embeddings).
- **Euclidean**: L2 distance.

### 2.7 Approximate Nearest Neighbors (`src/algos/ann.rs`, `src/algos/graph_ann.rs`)

Implements ANN via **Random Projections** (Locality Sensitive Hashing - LSH).
- **Construction**: Projects embeddings into lower-dimensional random subspaces.
- **Query**: Efficiently retrieves top-K similar nodes without exhaustive search.
- **Use Case**: Scaling similarity search to billions of nodes.

### 2.8 I/O & Serialization (`src/io.rs`)

Handles efficient loading and saving of graphs and embeddings.
- **Graph Loading**: Parses TSV files with format: `from_type\tfrom_name\tto_type\tto_name\tweight`.
- **Embedding Writing**: Exports embeddings to text or binary formats for downstream use.

---

## 3. Algorithm Catalog

### 3.1 Graph Clustering

#### Label Propagation Algorithm (LPA) (`src/algos/lpa.rs`)
- **Goal**: Assign each node to a single cluster based on neighbor majority.
- **Mechanism**: Iteratively updates node labels to the most frequent label among neighbors.
- **Parallelization**: Runs multiple independent LPA instances with different seeds; results are concatenated into a multi-dimensional embedding (Hamming distance).
- **Parameters**:
  - `passes`: Number of iterations (risk of overfitting if too high).
  - `k`: Number of parallel runs (embedding dimensionality).
- **Output**: `EmbeddingStore` where each dimension represents a cluster ID from one run.

#### Speaker-Listener Label Propagation (SLPA) (`src/algos/slpa.rs`)
- **Goal**: Detect overlapping communities (nodes can belong to multiple clusters).
- **Mechanism**: Nodes act as "speakers" broadcasting their memory of labels and "listeners" updating their own memory based on received labels.
- **Memory Threshold**: Prunes low-frequency labels from node memory.
- **Output**: Overlapping cluster assignments, useful for nodes with multiple roles (e.g., users interested in multiple topics).

### 3.2 Graph Embedding

#### Vector Propagation on Click Graphs (VPCG) (`src/algos/vpcg.rs`)
- **Goal**: Learn dense embeddings for bipartite graphs (e.g., User-Item click graphs) by propagating sparse features.
- **Mechanism**:
  1. Initialize node features (Uniform or IDF weighted).
  2. Iteratively propagate features across edges: `new_features = alpha * neighbor_features + (1-alpha) * current_features`.
  3. Filter and normalize features at each step.
- **Key Parameters**:
  - `alpha`: Propagation strength (0.0 to 1.0).
  - `max_terms`: Max features retained per node.
  - `err`: Threshold for pruning low-weight features.
  - `iterations`: Number of propagation passes.
- **Output**: Sparse embeddings converted to dense vectors via hashing or direct storage.

#### Embedding Propagation (EP) (`src/algos/ep/`)
- **Goal**: Learn node embeddings by minimizing a loss function over graph neighborhoods using SGD.
- **Architecture**:
  - **Model**: Defines how to aggregate neighbor embeddings (e.g., Averaging, Attention).
  - **Loss**: Contrastive losses (e.g., Margin Ranking, Triplet) comparing positive vs. negative pairs.
  - **Optimizer**: Adam optimizer with learning rate scheduling (warmup + cosine decay).
- **Process**:
  1. Sample batches of nodes.
  2. Generate hard negatives via random walks.
  3. Compute loss and gradients.
  4. Update feature embeddings in parallel using Hogwild.
- **Advanced Features**:
  - **Attention Mechanisms**: Multi-headed attention to weigh neighbor importance (`src/algos/ep/attention.rs`).
  - **Loss Weighting**: Degree-based reweighting (Log or Exponential) to handle long-tail distributions.
  - **Gradient Noise**: Injects Gaussian noise during training to prevent overfitting.
- **Output**: Dense `EmbeddingStore` with learned representations.

#### Instant Embeddings (`src/algos/instantembedding.rs`)
- **Goal**: Generate embeddings for new nodes without retraining the entire model.
- **Mechanism**: Uses a pre-trained estimator (e.g., linear regression or neural network) to map node features directly to embedding space.
- **Use Case**: Real-time inference for dynamic graphs where new nodes appear frequently.

#### PPR Embeddings (`src/algos/pprembed.rs`)
- **Goal**: Learn embeddings based on Personalized PageRank (PPR) proximity.
- **Mechanism**: Computes PPR scores between nodes and uses them to construct embeddings that capture local graph structure.

### 3.3 Random Walks

#### Random Walk with Restarts (RWR) (`src/algos/rwr.rs`)
- **Goal**: Compute proximity scores from a source node to all others.
- **Mechanism**: Simulates a random walker that, at each step, either moves to a neighbor or restarts at the source with probability `restart_prob`.
- **Output**: Steady-state probability distribution (PPR vector).

#### Guided RWR (`src/algos/grwr.rs`)
- **Goal**: Bias random walks towards specific behaviors or constraints.
- **Mechanism**: Uses a "guide" function to modify transition probabilities dynamically during the walk.
- **Applications**: Content-aware recommendations, constrained exploration.

### 3.4 Ranking & Tournaments

#### Luce Spectral Ranking (LSR) (`src/algos/lsr.rs`)
- **Goal**: Rank items based on pairwise comparisons (tournament data).
- **Mechanism**: Solves the eigenvector problem of the Laplacian matrix derived from comparison counts.
- **Use Case**: Sports rankings, product preference ordering.

### 3.5 Utilities

#### Neighborhood Alignment (`src/algos/alignment.rs`)
- **Goal**: Align neighborhoods of two nodes to measure structural similarity beyond direct connections.
- **Mechanism**: Computes optimal matching between neighbor sets.

#### Approximate Nearest Neighbors (ANN) (`src/algos/ann.rs`)
- **Goal**: Fast retrieval of similar nodes in high-dimensional spaces.
- **Mechanism**: Random projection trees or LSH.

#### Connected Components (`src/algos/connected.rs`)
- **Goal**: Identify disjoint subgraphs.
- **Mechanism**: Union-Find or BFS-based traversal.
- **Use Case**: Pruning isolated nodes, analyzing graph connectivity.

---

## 4. Performance Engineering

### 4.1 Parallelism Strategy
- **Rayon**: Used extensively for data-parallel operations (e.g., iterating over all nodes).
- **Chunking**: Large iterations are split into chunks to balance load and reduce overhead.
- **Thread Safety**: Relies on `Hogwild` for write-heavy sections; read-only sections use standard shared references.

### 4.2 Memory Optimization
- **CSR Format**: Reduces memory usage by ~50% compared to adjacency lists for sparse graphs.
- **In-Place Updates**: Algorithms like VPCG and EP update embeddings in-place to avoid allocation churn.
- **BitSets**: Compact representation for tracking initialized embeddings or visited nodes.

### 4.3 CPU-Specific Optimizations
- **Target CPU Native**: Compilation flag `-C target-cpu=native` enables AVX2/AVX-512 instructions.
- **Cache Locality**: CSR layout ensures sequential memory access during neighbor traversal.
- **SIMD**: Implicitly leveraged via `rayon` and optimized math libraries (e.g., `candle-core`).

### 4.4 GPU Support (Experimental)
- **Backend**: Uses `candle-core` for tensor operations.
- **Status**: CPU path is currently faster for typical workloads; CUDA support is provided for experimentation.
- **Build**: Requires `CUDA_COMPUTE_CAP` and `--features cuda`.

---

## 5. Python Integration

### 5.1 Build Process
```bash
pip install maturin numpy
RUSTFLAGS="-C target-cpu=native" maturin develop --release
```

### 5.2 Core Python API

#### Graph Construction
```python
import cloverleaf

# Adhoc construction
gb = cloverleaf.GraphBuilder()
gb.add_edge(('user', 'alice'), ('item', 'book1'), 1.0, cloverleaf.EdgeType.Undirected)
graph = gb.build_graph()

# Load from disk (TSV format)
graph = cloverleaf.Graph.load("edges.tsv", cloverleaf.EdgeType.Undirected)
```

#### Running Algorithms
```python
# Label Propagation
clusters = cloverleaf.lpa(graph, passes=10, seed=42)

# Embedding Propagation
propagator = cloverleaf.EmbeddingPropagation(
    alpha=0.01, loss=cloverleaf.Loss.Margin, batch_size=1024, 
    d_model=128, passes=50, hard_negs=5
)
embeddings = propagator.learn(graph, features, model)

# Random Walk with Restarts
rwr = cloverleaf.RWR(restart_prob=0.15)
scores = rwr.run(graph, source_node_id)
```

#### Query Interface
```python
query = cloverleaf.Query.node("user", "alice")
results = graph.search(query, top_k=10)
# Returns: [(('item', 'book1'), 0.95), ...]
```

### 5.3 Data Formats
- **Input**: TSV files with tab-separated columns: `from_type`, `from_name`, `to_type`, `to_name`, `weight`.
- **Output**: Embeddings exported as text (node_id, dim1, dim2, ...) or binary formats.

---

## 6. Use Cases & Applications

1. **Recommendation Systems**
   - VPCG and EP for learning user/item embeddings from click graphs.
   - ANN for real-time "similar items" retrieval.

2. **Community Detection**
   - LPA and SLPA for identifying overlapping communities in social networks.

3. **Knowledge Graph Completion**
   - Embedding propagation to infer missing links based on structural proximity.

4. **Ranking & Leaderboards**
   - LSR for ranking entities based on pairwise comparisons (e.g., sports, esports).

5. **Graph Visualization**
   - Dimensionality reduction of embeddings for 2D/3D visualization of large graphs.

---

## 7. Limitations & Future Directions

### Current Limitations
- **Static Graphs**: No support for dynamic updates (edge/node insertion/deletion) without rebuilding.
- **CPU Bound**: While optimized, extremely large graphs (>10B edges) may still require distributed processing.
- **GPU Maturity**: CUDA backend is experimental and not yet production-ready.

### Future Directions
- **Dynamic Graph Support**: Incremental updates to embeddings without full retraining.
- **Distributed Training**: Scaling EP and VPCG across multiple machines.
- **Deep Learning Integration**: Deeper integration with PyTorch/TensorFlow for hybrid models.
- **Advanced Loss Functions**: Incorporation of newer contrastive losses (e.g., InfoNCE).

---

## 8. Conclusion

Cloverleaf represents a robust, production-ready solution for large-scale graph analytics on CPU hardware. Its combination of efficient data structures (CSR), lock-free concurrency (Hogwild), and a rich suite of algorithms (LPA, VPCG, EP) makes it a powerful tool for practitioners working with massive graphs who cannot rely on GPU infrastructure. The library's modular design and Python bindings ensure ease of use while maintaining the performance benefits of Rust.

For further details, refer to the source code in `src/`, benchmarks in `benches/`, and examples in `examples/`.
