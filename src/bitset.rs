//! BitSet class; we use it in a few places to know when things have been set, such as embeddings.
use std::sync::atomic::{AtomicU32, Ordering};

/// Simple BitSet class, using 32-bit unsized ints for track flags
#[derive(Clone)]
pub struct BitSet {
    bitfield: Vec<u32>
}

impl BitSet {
    pub fn new(size: usize) -> Self {
        Self { bitfield: vec![0; (size / 32) + 1] }
    }

    fn get_bit_idx(&self, idx: &usize) -> (usize, u32) {
        let field_offset = idx / 32;
        let bit_offset = idx % 32;
        (field_offset, 1u32 << bit_offset)
    }

    pub fn is_set(&self, idx: usize) -> bool {
        let (fo, bm) = self.get_bit_idx(&idx);
        (self.bitfield[fo] & bm) > 0
    }

    pub fn set_bit(&mut self, idx: usize) {
        let (fo, bm) = self.get_bit_idx(&idx);
        self.bitfield[fo] |= bm;
    }

}

/// BitSet which can be set through a shared reference.  Used by the EmbeddingStore, whose hogwild
/// writers only hold `&self`.
pub struct AtomicBitSet {
    bitfield: Vec<AtomicU32>
}

impl AtomicBitSet {
    pub fn new(size: usize) -> Self {
        Self { bitfield: (0..(size / 32) + 1).map(|_| AtomicU32::new(0)).collect() }
    }

    fn get_bit_idx(&self, idx: &usize) -> (usize, u32) {
        let field_offset = idx / 32;
        let bit_offset = idx % 32;
        (field_offset, 1u32 << bit_offset)
    }

    pub fn is_set(&self, idx: usize) -> bool {
        let (fo, bm) = self.get_bit_idx(&idx);
        (self.bitfield[fo].load(Ordering::Relaxed) & bm) > 0
    }

    pub fn set_bit(&self, idx: usize) {
        let (fo, bm) = self.get_bit_idx(&idx);
        let field = &self.bitfield[fo];
        // Check first so repeated writes to the same embedding stay read-only on this word.
        if field.load(Ordering::Relaxed) & bm == 0 {
            field.fetch_or(bm, Ordering::Relaxed);
        }
    }
}

impl Clone for AtomicBitSet {
    fn clone(&self) -> Self {
        let bitfield = self.bitfield.iter()
            .map(|f| AtomicU32::new(f.load(Ordering::Relaxed)))
            .collect();
        AtomicBitSet { bitfield }
    }
}

#[cfg(test)]
mod bitset_tests {
    use super::*;

    #[test]
    fn test_setting() {
        let mut bitset = BitSet::new(12);
        bitset.set_bit(1);
        bitset.set_bit(3);
        bitset.set_bit(5);
        bitset.set_bit(7);
        bitset.set_bit(9);

        assert_eq!(bitset.is_set(1), true);
        assert_eq!(bitset.is_set(3), true);
        assert_eq!(bitset.is_set(5), true);
        assert_eq!(bitset.is_set(7), true);

        assert_eq!(bitset.is_set(2), false);
        assert_eq!(bitset.is_set(4), false);
        assert_eq!(bitset.is_set(6), false);
        assert_eq!(bitset.is_set(8), false);
        
        for i in 0..10 {
            let truthy = (i % 2) == 1;
            assert_eq!(bitset.is_set(i), truthy);
        }

    }

    #[test]
    fn test_size_of() {
        let mut bitset = BitSet::new(1201);
        assert_eq!(bitset.bitfield.len(), 38);
    }


}
