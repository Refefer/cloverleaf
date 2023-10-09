//! Defines the FeatureStore class which is used to define discrete features for each node
use std::sync::Arc;
use crate::NodeID;
use crate::vocab::Vocab;

/// Main FeatureStore struct.  We use a vector of vectors to allow for dynamic numbers of features.
/// This can and should be updated to a more memory friendly approach since vectors have surprising
/// overhead and the number of discrete features tends to be relatively small.
#[derive(Debug)]
pub struct FeatureStore {
    /// Raw storage for features, indexed by node id
    features: Vec<Vec<usize>>,

    /// Since we often convert features to embeddings, which need namespaces, we have a feature
    /// namespace.
    namespace: String,

    /// Maps a raw feature to a feature_id
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

    fn set_nt_features(&mut self, node: NodeID, namespace: String, node_features: Vec<String>) {
        let ns = Arc::new(namespace);
        let fs = node_features.iter().map(|f| f.as_str());
        self.set_nt_features_shared(node, ns, fs);
    }

    fn set_nt_features_shared<'a>(&mut self, 
        node: NodeID, 
        namespace: Arc<String>, 
        node_features: impl Iterator<Item=&'a str>
    ) {
        self.features[node] = node_features
            .map(|f| self.feature_vocab.get_or_insert_shared(namespace.clone(), f))
            .collect()
    }

    pub fn set_features(&mut self, node: NodeID, node_features: Vec<String>) {
        self.set_nt_features(node, self.namespace.clone(), node_features);
    }

    pub fn set_features_raw(&mut self, node: NodeID, node_features: impl Iterator<Item=usize>) {
        self.features[node].extend(node_features);
    }

    pub fn get_features(&self, node: NodeID) -> &[usize] {
        &self.features[node]
    }

    fn get_pretty_feature(&self, feat_id: usize) -> String {
        let (_nt, name) = self.feature_vocab.get_name(feat_id).unwrap();
        name.to_string()
    }

    pub fn get_pretty_features(&self, node: NodeID) -> Vec<String> {
        self.features[node].iter().map(|v_id| {
            self.get_pretty_feature(*v_id)
        }).collect()
    }

    pub fn num_features(&self) -> usize {
        self.feature_vocab.len()
    }

    pub fn num_nodes(&self) -> usize {
        self.features.len()
    }

    /// This method assigns an unique, anonymous feature to all nodes which lack any features.  This is
    /// necessary for all graph embedding algorithms which map {feature} -> Embedding.
    pub fn fill_missing_nodes(&mut self) {
        for i in 0..self.features.len() {
            if self.features[i].len() == 0 {
                self.set_nt_features(i, "node".into(), vec![format!("{}", i)]);
            }
        }
    }

    pub fn get_vocab(&self) -> &Vocab {
        &self.feature_vocab
    }

    pub fn clone_vocab(&self) -> Vocab {
        self.feature_vocab.clone()
    }

    pub fn iter(&self) -> impl Iterator<Item=&Vec<usize>> {
        self.features.iter()
    }

    /// Count the number of occurrences of each feature in the feature set.  This is helpful when
    /// pruning to a minimum count.
    pub fn count_features(&self) -> Vec<usize> {
        let mut counts = vec![0usize; self.feature_vocab.len()];
        for feats in self.features.iter() {
            for f_i in feats.iter() {
                counts[*f_i] += 1;
            }
        }
        counts
    }

    /// Removes features which don't meet the provided `count`.  This is helpful to prevent one-off
    /// occurences of words acting as node biasesand otherwise harming the quality of the
    /// embeddings.
    pub fn prune_min_count(&self, count: usize) -> FeatureStore {
        let counts = self.count_features();

        let mut new_fs = FeatureStore::new(self.features.len(), self.namespace.clone());
        
        // Filter out features that don't meet the min_count
        let ns = Arc::new(self.namespace.clone());
        self.features.iter().enumerate().for_each(|(node_id, feats)| {
            let new_feats = feats.iter()
                .filter(|f_i| counts[**f_i] >= count)
                .map(|f_i| {
                    let (_nt, nn) = self.feature_vocab.get_name(*f_i)
                        .expect("Should never be unavailable!");
                    nn
                });

            new_fs.set_nt_features_shared(node_id, ns.clone(), new_feats);
        });
        new_fs
    }

}

