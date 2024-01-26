use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;

use crate::graph::NodeID;
use crate::embeddings::{EmbeddingStore,Entity};
use crate::algos::graph_ann::NodeDistance;

struct Hyperplane {
    coef: Vec<f32>,
    bias: f32
}

impl Hyperplane {
    fn new(coef: Vec<f32>, bias: f32) -> Self {
        Hyperplane { coef, bias }
    }

    fn point_is_above(&self, emb: &[f32]) -> bool {
        self.coef.iter().zip(emb.iter())
            .map(|(ci, ei)| ci * ei)
            .sum::<f32>() + self.bias >= 0.
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


fn tree_predict(
    tree_table: &TreeTable,
    es: &EmbeddingStore, 
    emb: &[f32]
) -> Vec<(NodeID, f32)> {
    let mut node = tree_table.len() - 1;
    loop {
        match &tree_table[node] {
            Tree::Leaf { ref indices } => {
                let qemb = Entity::Embedding(emb);
                return indices.par_iter().map(|idx| {
                    let d = es.compute_distance(&Entity::Node(*idx), &qemb);
                    (*idx, d)
                }).collect()
            },
            Tree::Split { ref hp, ref above, ref below } => {
                node = if hp.point_is_above(emb) { *above } else { *below };
            }
        }
    }
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
        seed: u64
    ) {
        self.trees.clear();
        let mut trees = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            trees.push(Vec::new());
        }

        trees.par_iter_mut().enumerate().for_each(|(idx, tree) | {
            let indices = (0..es.len()).collect::<Vec<_>>();
            let mut rng = XorShiftRng::seed_from_u64(seed + idx as u64);
            self.fit_group_(tree, 1, es, indices, max_nodes_per_leaf, &mut rng);
        });

        self.trees = trees;

    }

    pub fn depth(&self) -> Vec<usize> {
        self.trees.par_iter().map(|t| tree_depth(t, t.len() - 1)).collect()
    }

    fn fit_group_(
        &self, 
        tree_table: &mut TreeTable,
        depth: usize,
        es: &EmbeddingStore,
        indices: Vec<NodeID>,
        max_nodes_per_leaf: usize,
        rng: &mut impl Rng
    ) -> TreeIndex {
        if indices.len() < max_nodes_per_leaf {
            tree_table.push(Tree::Leaf { indices });
            return tree_table.len() - 1
        }

        // Pick two point
        let mut best = (0i8, None);
        for _ in 0..5 {
            let idx_1 = indices.choose(rng).unwrap();
            let mut idx_2 = indices.choose(rng).unwrap();
            while idx_1 == idx_2 {
                idx_2 = indices.choose(rng).unwrap();
            }

            let pa = es.get_embedding(*idx_1); 
            let pb = es.get_embedding(*idx_2); 

            let diff: Vec<_> = pa.iter().zip(pb.iter()).map(|(pai, pbi)| pai - pbi).collect();
            let bias: f32 = diff.iter().zip(pa.iter().zip(pb.iter()))
                .map(|(d, (pai, pbi))| d * (pai + pbi) / 2.)
                .sum();

            let hp = Hyperplane::new(diff, bias);
            let mut s = 0i8;
            for _ in 0..30 {
                let idx = indices.choose(rng).unwrap();
                let emb = es.get_embedding(*idx);
                if hp.point_is_above(emb) { s += 1; } 
            }
            let score = (s - 15).abs();
            if best.0 > score || best.1.is_none() {
                best = (score, Some(hp));
            }
        }

        let hp = best.1.unwrap();
        let scores = indices.par_iter().map(|idx| {
            hp.point_is_above(es.get_embedding(*idx))
        }).collect::<Vec<_>>();

        let mut above = Vec::new();
        let mut below = Vec::new();

        scores.into_iter().zip(indices.into_iter()).for_each(|(is_above, idx)| {
            if is_above {
                above.push(idx);
            } else {
                below.push(idx);
            }
        });

        if above.len() > 0 && below.len() > 0 {
            let above_idx = self.fit_group_(tree_table, depth+1, es, above, max_nodes_per_leaf, rng);
            let below_idx = self.fit_group_(tree_table, depth+1, es, below, max_nodes_per_leaf, rng);

            tree_table.push(Tree::Split { hp: hp, above: above_idx, below: below_idx })
        } else {
            let idxs = if above.len() == 0 { below } else { above };
            tree_table.push(Tree::Leaf { indices: idxs })
        }
        tree_table.len() - 1
    }

    pub fn predict(
        &self, 
        es: &EmbeddingStore, 
        emb: &[f32]
    ) -> Vec<NodeDistance> {
        let scores = self.trees.par_iter().map(|tree| {
            tree_predict(tree, es, emb)
        }).collect::<Vec<_>>();

        let n = scores.iter().map(|x| x.len()).sum::<usize>();
        let mut all_scores = Vec::with_capacity(n);
        scores.into_iter().for_each(|subset| {
            subset.into_iter().for_each(|(node_id, s)| {
                all_scores.push(NodeDistance(s, node_id));
            });
        });

        all_scores.par_sort();

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
