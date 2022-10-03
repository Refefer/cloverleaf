use hashbrown::HashMap;
use crate::graph::NodeID;
use std::sync::Arc;

#[derive(Debug)]
pub struct Vocab {
    vocab_to_idx: HashMap<Arc<String>, NodeID>,
    idx_to_vocab: Vec<Arc<String>>
}

impl Vocab {
    pub fn new() -> Self {
        Vocab { 
            vocab_to_idx: HashMap::new(),
            idx_to_vocab: Vec::new()
        }
    }

    pub fn get_node_id(&self, name: String) -> Option<NodeID> {
        let node = Arc::new(name);
        self.vocab_to_idx.get(&node).map(|n| n.clone())
    }

    pub fn get_or_insert(&mut self, name: String) -> NodeID {
        let node = Arc::new(name);
        if let Some(node_id) = self.vocab_to_idx.get(&node) {
            node_id.clone()
        } else {
            let new_idx = self.idx_to_vocab.len();
            self.vocab_to_idx.insert(node.clone(), new_idx.clone());
            self.idx_to_vocab.push(node);
            new_idx
        }
    }

    pub fn get_name(&self, node: NodeID) -> Option<Arc<String>> {
        self.idx_to_vocab.get(node).map(|v| v.clone())
    }

    pub fn len(&self) -> usize {
        self.idx_to_vocab.len()
    }
}
