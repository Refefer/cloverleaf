use hashbrown::HashMap;
use crate::graph::NodeID;
use std::sync::Arc;

#[derive(Debug)]
pub struct Vocab {
    vocab_to_idx: HashMap<(usize, Arc<String>), NodeID>,
    node_id_to_node: Vec<(usize,Arc<String>)>,
    node_type_to_id: HashMap<Arc<String>, usize>,
    id_to_node_type: Vec<Arc<String>>,
}

impl Vocab {
    pub fn new() -> Self {
        Vocab { 
            node_type_to_id: HashMap::new(),
            id_to_node_type: Vec::new(),
            vocab_to_idx: HashMap::new(),
            node_id_to_node: Vec::new()
        }
    }

    pub fn get_node_id(&self, node_type: String, name: String) -> Option<NodeID> {
        let node_type = Arc::new(node_type);
        self.node_type_to_id.get(&node_type).and_then(|nt_id| {
            let node = Arc::new(name);
            self.vocab_to_idx.get(&(*nt_id, node)).map(|n| n.clone())
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

    pub fn get_or_insert(&mut self, node_type: String, name: String) -> NodeID {
        let node = Arc::new(name);
        let nt_id = self.get_or_insert_node_type(Arc::new(node_type));
        let t = (nt_id, node);
        if let Some(node_id) = self.vocab_to_idx.get(&t) {
            node_id.clone()
        } else {
            let new_idx = self.node_id_to_node.len();
            self.vocab_to_idx.insert((t.0.clone(), t.1.clone()), new_idx.clone());
            self.node_id_to_node.push((t.0, t.1));
            new_idx
        }
    }

    pub fn get_name(&self, node: NodeID) -> Option<(Arc<String>, Arc<String>)> {
        self.node_id_to_node.get(node).map(|(nt_id, name)| {
            (self.id_to_node_type[*nt_id].clone(), name.clone())
        })
    }

    pub fn len(&self) -> usize {
        self.node_id_to_node.len()
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
            let expected = Some((Arc::new((*nt).clone()), Arc::new((*n).clone())));
            let ret = vocab.get_name(*node_id);
            assert_eq!(expected, ret);
        });
    }

}
