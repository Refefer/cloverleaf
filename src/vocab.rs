use lasso::{Rodeo,Spur};
use hashbrown::HashMap;
use crate::graph::NodeID;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize,Ordering};

static VOCAB_ID: AtomicUsize = AtomicUsize::new(0);

pub type TranslationTable = Vec<Option<NodeID>>;

#[derive(Clone,Debug)]
pub struct Vocab {
    interner: Rodeo,
    vocab_id: usize,
    vocab_to_idx: HashMap<(usize, Spur), NodeID>,
    node_id_to_node: Vec<(usize,Spur)>,
    node_type_to_id: HashMap<Arc<String>, usize>,
    id_to_node_type: Vec<Arc<String>>,
}

impl Vocab {
    pub fn new() -> Self {
        let vocab_id = VOCAB_ID.fetch_add(1, Ordering::SeqCst);
        Vocab { 
            interner: Rodeo::default(),
            vocab_id: vocab_id,
            node_type_to_id: HashMap::new(),
            id_to_node_type: Vec::new(),
            vocab_to_idx: HashMap::new(),
            node_id_to_node: Vec::new()
        }
    }

    pub fn is_identical(&self, other: &Vocab) -> bool {
        self.vocab_id == other.vocab_id
    }

    pub fn get_node_id<A: Into<String>, B: AsRef<str>>(&self, node_type: A, name: B) -> Option<NodeID> {
        self.get_node_id_int(&Arc::new(node_type.into()), name.as_ref())
    }

    fn get_node_id_int(&self, node_type: &Arc<String>, name: &str) -> Option<NodeID> {
        self.node_type_to_id.get(node_type).and_then(|nt_id| {
            self.interner.get(name).and_then(|key| {
                self.vocab_to_idx.get(&(*nt_id, key)).map(|n| n.clone())
            })
        })
    }

    pub fn get_node_type(&self, node: NodeID) -> Option<&Arc<String>> {
        self.node_id_to_node.get(node).map(|(nt_id, _name)| {
            &self.id_to_node_type[*nt_id]
        })
    }

    fn get_or_insert_node_type(&mut self, node_type: Arc<String>) -> usize {
        if let Some(nt_id) = self.node_type_to_id.get(&node_type) {
            *nt_id
        } else {
            let new_idx = self.id_to_node_type.len();
            self.node_type_to_id.insert(node_type.clone(), new_idx.clone());
            self.id_to_node_type.push(node_type);
            new_idx
        }
    }

    pub fn get_or_insert<A: Into<String>, B: AsRef<str>>(&mut self, node_type: A, name: B) -> NodeID {
        self.get_or_insert_shared(Arc::new(node_type.into()), name.as_ref())
    }

    fn get_or_insert_shared(&mut self, node_type: Arc<String>, name: &str) -> NodeID {
        let nt_id = self.get_or_insert_node_type(node_type);
        let name_key = self.interner.get_or_intern(name);
        let t = (nt_id, name_key);
        if let Some(node_id) = self.vocab_to_idx.get(&t) {
            node_id.clone()
        } else {
            let new_idx = self.node_id_to_node.len();
            self.vocab_to_idx.insert((t.0.clone(), t.1.clone()), new_idx.clone());
            self.node_id_to_node.push((t.0, t.1));
            new_idx
        }
 
    }

    pub fn get_name(&self, node: NodeID) -> Option<(Arc<String>, &str)> {
        self.node_id_to_node.get(node).map(|(nt_id, name)| {
            let nn = self.interner.resolve(name);
            (self.id_to_node_type[*nt_id].clone(), nn)
        })
    }

    pub fn len(&self) -> usize {
        self.node_id_to_node.len()
    }

    pub fn translate_node(&self, other: &Vocab, other_node_id: NodeID) -> Option<NodeID> {
        if self.is_identical(other) {
            Some(other_node_id)
        } else {
            other.get_name(other_node_id).and_then(|(node_type, node_name)| {
                self.get_node_id_int(&node_type, node_name)
            })
        }
    }

    pub fn create_translation_table(&self, to_vocab: &Vocab) -> TranslationTable {
        if self.is_identical(to_vocab) {
            (0..self.node_id_to_node.len()).map(|idx| Some(idx)).collect()
        } else {
            self.node_id_to_node.iter().map(|(node_type_id, node_name)| {
                let node_type = &self.id_to_node_type[*node_type_id];
                let name = self.interner.resolve(node_name);
                to_vocab.get_node_id_int(node_type, name)
            }).collect()
        }
    }

}

#[cfg(test)]
mod vocab_tests {
    use super::*;

    #[test]
    fn test_get_set() {
        let mut vocab = Vocab::new();

        let nodes = vec![
            ("feat".to_string(), "abc".to_string()),
            ("feat".to_string(), "efg".to_string()),
            ("title".to_string(), "foobarbaz".to_string())
        ];

        let node_ids: Vec<_> = nodes.iter().map(|(nt, n)| {
            vocab.get_or_insert(nt.clone(), n.clone())
        }).collect();

        nodes.iter().zip(node_ids.iter()).for_each(|((nt, n), node_id)| {
            // Assert lookups get the correct node_ids
            let expected = Some(node_id.clone());
            let ret = vocab.get_node_id(nt.clone(), n.clone());
            assert_eq!(expected, ret);

            // Assert node ids retrieve the correct node type/name
            let expected = Some((Arc::new((*nt).clone()), n.as_str()));
            let ret = vocab.get_name(*node_id);
            assert_eq!(expected, ret);
        });
    }

}
