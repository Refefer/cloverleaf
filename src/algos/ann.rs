use std::cmp::{Ordering,Eq};
use std::collections::BinaryHeap;

use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;
use float_ord::FloatOrd;

use crate::graph::NodeID;
use crate::embeddings::{EmbeddingStore,Entity};
use crate::algos::graph_ann::{NodeDistance,TopK};

struct Hyperplane {
    coef: Vec<f32>,
    bias: f32
}

impl Hyperplane {
    fn new(coef: Vec<f32>, bias: f32) -> Self {
        Hyperplane { coef, bias }
    }

    fn point_is_above(&self, emb: &[f32]) -> bool {
        self.distance(emb) >= 0.
    }

    fn distance(&self, emb: &[f32]) -> f32 {
        self.coef.iter().zip(emb.iter())
            .map(|(ci, ei)| ci * ei)
            .sum::<f32>() + self.bias
    }

}

type TreeIndex = usize;
type TreeTable = Vec<Tree>;

enum Tree {
    Leaf { indices: Vec<NodeID> },

    Split {
        hp: Hyperplane,
        above: TreeIndex,
        below: TreeIndex
    }
}

#[derive(Debug)]
struct HpDistance(f32, usize);

impl HpDistance {
    fn new(tree_idx: usize, score: f32) -> Self {
        HpDistance(score, tree_idx)
    }

}

impl Ord for HpDistance {
    fn cmp(&self, other: &Self) -> Ordering {
        FloatOrd(other.0).cmp(&FloatOrd(self.0)).then_with(|| other.1.cmp(&self.1))
    }
}

impl PartialOrd for HpDistance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for HpDistance {
    fn eq(&self, other: &Self) -> bool {
        let s_cmp = (self.0, self.1);
        let o_cmp = (other.0, other.1);
        s_cmp == o_cmp
    }
}

impl Eq for HpDistance {}

fn tree_predict(
    tree_table: &TreeTable,
    es: &EmbeddingStore, 
    emb: &[f32],
    k: usize,
    mut min_search_nodes: usize
) -> Vec<(NodeID, f32)> {

    // Must explore at least K
    min_search_nodes = min_search_nodes.max(k);

    // The set of nodes to return
    let mut return_set = TopK::new(k);
    
    // heap to store tree splits
    let mut heap = BinaryHeap::with_capacity(k * 2);

    // Root node is last in the table.  Start exploring the root tree
    let tree_idx = tree_table.len() - 1;
    heap.push( HpDistance::new(tree_idx, 0.) );

    let mut visited = 0usize;
    while let Some(HpDistance(_, tree_idx)) = heap.pop() {
        match &tree_table[tree_idx] {
            Tree::Leaf { ref indices } => {
                let qemb = Entity::Embedding(emb);
                indices.iter().for_each(|&node_id| {
                    let dist = es.compute_distance(&Entity::Node(node_id), &qemb);
                    return_set.push(node_id, dist);
                    visited += 1;
                });
            },
            Tree::Split { ref hp, ref above, ref below } => {
                let dist = hp.distance(emb);
                let above_dist = if dist >= 0.0 { 0.0 } else { dist.abs() };
                let below_dist = if dist < 0.0 { 0.0 } else { dist.abs() };
                heap.push(HpDistance::new(*above, above_dist));
                heap.push(HpDistance::new(*below, below_dist));
            }
        }
        if visited >= min_search_nodes { break }
    }

    return_set.into_sorted().into_iter().map(|nd| {
        nd.to_tup_cloned()
    }).collect()
}


fn tree_leaf_index(
    tree_table: &TreeTable,
    emb: &[f32]
) -> usize {
    let mut node = tree_table.len() - 1;
    loop {
        match &tree_table[node] {
            Tree::Leaf { indices: _ } => { return node },
            Tree::Split { ref hp, ref above, ref below } => {
                node = if hp.point_is_above(emb) { *above } else { *below };
            }
        }
    }
}

/**
 * Produces the path an embedding took through the ANN and returns it as
 * a path
 */
fn tree_leaf_path(
    tree_table: &TreeTable,
    emb: &[f32]
) -> Vec<usize> {
    let mut path = Vec::new();
    let mut node = tree_table.len() - 1;
    loop {
        match &tree_table[node] {
            Tree::Leaf { indices: _ } => { break },
            Tree::Split { ref hp, ref above, ref below } => {
                node = if hp.point_is_above(emb) { *above } else { *below };
            }
        }
        path.push(node);
    }
    path
}

/**
 * Computes the max depth of a tree
 */
fn tree_depth(
    tree_table: &TreeTable,
    node: TreeIndex
) -> usize {
    match &tree_table[node] {
        Tree::Leaf { indices: _ } =>  1,
        Tree::Split { hp: _, above, below } => {
            let above_depth = tree_depth(tree_table, *above);
            let below_depth = tree_depth(tree_table, *below);
            above_depth.max(below_depth) + 1
        }
    }
}

pub struct AnnBuildConfig {
    max_nodes_per_leaf: usize,
    test_hp_per_split: usize,
    num_sampled_nodes_split_test: usize
}

/** Implements an ANN based on random hyperplanes.  It offers the advantage of also
 * producing leaf index transforms, which can be suitable for indexing in traditional 
 * inverted indexs
 */
pub struct Ann {
    trees: Vec<TreeTable>
}

impl Ann {
    pub fn new() -> Self {
        Ann { trees: Vec::new() }
    }

    pub fn fit(
        &mut self,
        es: &EmbeddingStore,
        n_trees: usize,
        max_nodes_per_leaf: usize,
        test_hp_per_split: Option<usize>,
        num_sampled_nodes_split_test: Option<usize>,
        seed: u64
    ) {
        let config = AnnBuildConfig {
            max_nodes_per_leaf: max_nodes_per_leaf,
            test_hp_per_split: test_hp_per_split.unwrap_or(5),
            num_sampled_nodes_split_test: num_sampled_nodes_split_test.unwrap_or(30)
        };

        // Setup the number of trees necessary to build
        self.trees.clear();
        let mut trees = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            trees.push(Vec::new());
        }

        // Learn each tree, using separate random seeds
        trees.par_iter_mut().enumerate().for_each(|(idx, tree) | {
            let mut indices = (0..es.len()).map(|idx| (idx, false)).collect::<Vec<_>>();
            let mut rng = XorShiftRng::seed_from_u64(seed + idx as u64);
            self.fit_group_(&config, tree, 1, es, indices.as_mut_slice(), &mut rng);
        });

        self.trees = trees;

    }

    pub fn depth(&self) -> Vec<usize> {
        self.trees.par_iter().map(|t| tree_depth(t, t.len() - 1)).collect()
    }

    fn fit_group_(
        &self, 
        config: &AnnBuildConfig,
        tree_table: &mut TreeTable,
        depth: usize,
        es: &EmbeddingStore,
        indices: &mut [(NodeID, bool)],
        rng: &mut impl Rng
    ) -> TreeIndex {
        if indices.len() < config.max_nodes_per_leaf {
            let node_ids = indices.iter().map(|(node_id, _)| *node_id).collect();
            tree_table.push(Tree::Leaf { indices: node_ids });
            return tree_table.len() - 1
        }

        // Pick two point to create the hyperplane
        let mut best = (0usize, None);
        // Try several different candidates and select the hyperplane that divides them the best
        for _ in 0..config.test_hp_per_split {
            
            // Select two points to create the hyperplane
            let idx_1 = indices.choose(rng).unwrap().0;
            let mut idx_2 = idx_1;
            while idx_1 == idx_2 {
                idx_2 = indices.choose(rng).unwrap().0;
            }

            let pa = es.get_embedding(idx_1); 
            let pb = es.get_embedding(idx_2); 

            // Compute the hyperplane
            let diff: Vec<_> = pa.iter().zip(pb.iter()).map(|(pai, pbi)| pai - pbi).collect();
            
            // Figure out the vector bias
            let bias: f32 = diff.iter().zip(pa.iter().zip(pb.iter()))
                .map(|(d, (pai, pbi))| d * (pai + pbi) / 2.)
                .sum();

            // Count the number of instances on each side of the hyperplane from random points
            let hp = Hyperplane::new(diff, bias);
            let mut s = 0usize;
            for _ in 0..config.num_sampled_nodes_split_test {
                let idx = indices.choose(rng).unwrap().0;
                let emb = es.get_embedding(idx);
                if hp.point_is_above(emb) { s += 1; } 
            }
             
            let delta = config.num_sampled_nodes_split_test - s;
            let score = s.max(delta) - s.min(delta);
            if score < best.0 || best.1.is_none() {
                best = (score, Some(hp));
            }
        }

        // Score the nodes and get the number below the hyperplane
        let hp = best.1.unwrap();
        let split_idx: usize = indices.par_iter_mut().map(|v| {
            v.1 = hp.point_is_above(es.get_embedding(v.0));
            if v.1 { 0 } else { 1 }
        }).sum();

        // Fast sort - we do this to save memory allocations
        indices.par_sort_by_key(|s| s.1);

        let (below, above) = indices.split_at_mut(split_idx);

        if above.len() > 0 && below.len() > 0 {
            let above_idx = self.fit_group_(config, tree_table, depth + 1, es, above, rng);
            let below_idx = self.fit_group_(config, tree_table, depth + 1, es, below, rng);

            tree_table.push(Tree::Split { hp: hp, above: above_idx, below: below_idx })

        } else {
            let node_ids = indices.iter().map(|(node_id, _)| *node_id).collect();
            tree_table.push(Tree::Leaf { indices: node_ids })
        }

        tree_table.len() - 1
    }

    pub fn predict(
        &self, 
        es: &EmbeddingStore, 
        emb: &[f32],
        k: usize,
        min_search_nodes: Option<usize>
    ) -> Vec<NodeDistance> {
        // Get the scores
        let min_search = min_search_nodes.unwrap_or(self.trees.len() * k);
        let scores = self.trees.par_iter().map(|tree| {
            tree_predict(tree, es, emb, k, min_search)
        }).collect::<Vec<_>>();

        // Fold them into a single vec
        let n = scores.iter().map(|x| x.len()).sum::<usize>();
        let mut all_scores = Vec::with_capacity(n);
        scores.into_iter().for_each(|subset| {
            subset.into_iter().for_each(|(node_id, s)| {
                all_scores.push(NodeDistance::new(s, node_id));
            });
        });

        // Sort by Score and Node ID
        all_scores.par_sort();

        // Deduplicate nodes which show up in multiple trees
        let mut cur_pointer = 1;
        let mut cur_node_id = all_scores[0].1;
        for i in 1..n {
            let next_id = all_scores[i].1;
            if next_id != cur_node_id {
                all_scores[cur_pointer] = all_scores[i];
                cur_node_id = next_id;
                cur_pointer += 1;
            }
        }

        all_scores.truncate(cur_pointer);
        all_scores.reverse();
        all_scores.truncate(k);
        all_scores
    }

    pub fn predict_leaf_indices(
        &self,
        emb: &[f32]
    ) -> Vec<usize> {
        self.trees.par_iter().map(|tree| {
            tree_leaf_index(tree, emb)
        }).collect()
    }

    pub fn predict_leaf_paths(
        &self,
        emb: &[f32]
    ) -> Vec<Vec<usize>> {
        self.trees.par_iter().map(|tree| {
            tree_leaf_path(tree, emb)
        }).collect()
    }

    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

}
