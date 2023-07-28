/// Random utility functions
use std::hash::{Hash,Hasher};

use rand::prelude::*;
use ahash::AHasher;

/// Counts a set of items by id.  See the test for examples.
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

pub fn get_best_count<R: Rng>(counts: &[usize], rng: &mut R) -> usize {
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

pub struct FeatureHasher {
    dims: usize
}

impl FeatureHasher {

    pub fn new(dims: usize) -> Self {
        FeatureHasher { dims }
    }

    #[inline]
    pub fn hash(
        &self,
        feature: usize, 
        hash_num: usize
    ) -> (i8, usize) {
        self.compute_sign_idx(feature, hash_num)
    }

    #[inline(always)]
    fn calculate_hash<T: Hash>(t: T) -> u64 {
        let mut s = AHasher::default();
        t.hash(&mut s);
        s.finish()
    }

    #[inline]
    fn compute_sign_idx(&self, feat: usize, hash_num: usize) -> (i8, usize) {
        let hash = FeatureHasher::calculate_hash((feat, hash_num)) as usize;
        let sign = (hash & 1) as i8;
        let idx = (hash >> 1) % self.dims as usize;
        (2 * sign - 1, idx)
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

