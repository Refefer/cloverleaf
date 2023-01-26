use rand::prelude::*;
use crate::NodeID;
use crate::vocab::Vocab;

pub struct Counter<'a> {
    slice: &'a [usize],
    idx: usize
}

impl <'a> Counter<'a> {
    pub fn new(slice: &'a [usize]) -> Self {
        Counter {
            slice,
            idx: 0
        }
    }
}

impl <'a> Iterator for Counter<'a> {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        let start = self.idx;
        let mut count = 0;
        for _ in self.idx..self.slice.len() {
            if self.slice[self.idx] != self.slice[start] { 
                return Some((self.slice[start], count)) 
            }
            self.idx += 1;
            count += 1;
        }
        if count > 0 {
            return Some((self.slice[start], count)) 
        } 
        None
    }
}

pub fn get_best_count<R: Rng>(counts: &[usize], rng: &mut R) -> usize{
    let mut best_count = 0;
    let mut ties = Vec::new();
    for (cluster, count) in Counter::new(counts) {
        if count > best_count {
            best_count = count;
            ties.clear();
            ties.push(cluster)
        } else if count == best_count {
            ties.push(cluster)
        }
    }

    // We tie break by randomly choosing an item
    if ties.len() > 1 {
        *ties.as_slice()
            .choose(rng)
            .expect("If a node has no edges, code bug")
    } else {
        ties[0]
    }

}

#[derive(Debug)]
pub struct FeatureStore {
    features: Vec<Vec<usize>>,
    namespace: String,
    feature_vocab: Vocab,
}

impl FeatureStore {

    pub fn new(size: usize, namespace: String) -> Self {
        FeatureStore {
            features: vec![Vec::with_capacity(0); size],
            namespace: namespace,
            feature_vocab: Vocab::new(),
        }
    }

    pub fn get_ns(&self) -> &String {
        &self.namespace
    }

    pub fn set_features(&mut self, node: NodeID, node_features: Vec<String>) {
        self.features[node] = node_features.into_iter()
            .map(|f| self.feature_vocab.get_or_insert(self.namespace.clone(), f))
            .collect()
    }

    pub fn get_features(&self, node: NodeID) -> &[usize] {
        &self.features[node]
    }

    pub fn get_pretty_features(&self, node: NodeID) -> Vec<String> {
        self.features[node].iter().map(|v_id| {
            let (_nt, name) = self.feature_vocab.get_name(*v_id).unwrap();
            (*name).clone()
        }).collect()
    }

    pub fn num_features(&self) -> usize {
        self.feature_vocab.len()
    }

    pub fn num_nodes(&self) -> usize {
        self.features.len()
    }

    pub fn fill_missing_nodes(&mut self) {
        let mut idxs = self.feature_vocab.len();
        for i in 0..self.features.len() {
            if self.features[i].len() == 0 {
                self.set_features(i, vec![format!("_node_{}", i)]);
            }
        }
    }

    pub fn get_vocab(self) -> Vocab {
        self.feature_vocab
    }

    pub fn clone_vocab(&self) -> Vocab {
        self.feature_vocab.clone()
    }

    pub fn iter(&self) -> impl Iterator<Item=&Vec<usize>> {
        self.features.iter()
    }
}

#[cfg(test)]
mod utils_tests {
    use super::*;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_choose_best() {
        let counts = vec![0, 0, 1, 1, 1, 2];
        let mut rng = XorShiftRng::seed_from_u64(1);
        let best_count = get_best_count(&counts, &mut rng);
        assert_eq!(best_count, 1);
    }

    #[test]
    fn test_choose_one() {
        let counts = vec![0, 0, 0];
        let mut rng = XorShiftRng::seed_from_u64(1);
        let best_count = get_best_count(&counts, &mut rng);
        assert_eq!(best_count, 0);
    }

    #[test]
    fn test_choose_between() {
        let counts = vec![0, 1];
        let mut rng = XorShiftRng::seed_from_u64(1231232132);
        let best_count = get_best_count(&counts, &mut rng);
        assert_eq!(best_count, 0);
        let best_count = get_best_count(&counts, &mut rng);
        assert_eq!(best_count, 1);
        let best_count = get_best_count(&counts, &mut rng);
        assert_eq!(best_count, 1);
        let best_count = get_best_count(&counts, &mut rng);
        assert_eq!(best_count, 0);
    }

    #[test]
    fn test_choose_last() {
        let counts = vec![0, 1, 1];
        let mut rng = XorShiftRng::seed_from_u64(1231232132);
        let best_count = get_best_count(&counts, &mut rng);
        assert_eq!(best_count, 1);
    }

    #[test]
    fn test_choose_only() {
        let counts = vec![0, 0, 0];
        let mut rng = XorShiftRng::seed_from_u64(1231232132);
        let best_count = get_best_count(&counts, &mut rng);
        assert_eq!(best_count, 0);
    }

    #[test]
    fn test_counter() {
        let counts = [0, 0, 0, 1, 2, 2, 3];
        let mut counter = Counter::new(&counts);
        assert_eq!(counter.next(), Some((0, 3)));
        assert_eq!(counter.next(), Some((1, 1)));
        assert_eq!(counter.next(), Some((2, 2)));
        assert_eq!(counter.next(), Some((3, 1)));
        assert_eq!(counter.next(), None);

        let counts = [];
        let mut counter = Counter::new(&counts);
        assert_eq!(counter.next(), None);

        let counts = [0];
        let mut counter = Counter::new(&counts);
        assert_eq!(counter.next(), Some((0, 1)));
        assert_eq!(counter.next(), None);

    }

}

