//! Defines the FeatureStore class which is used to define discrete features for each node
use std::sync::Arc;
use crate::NodeID;
use crate::vocab::Vocab;

struct ArcWrap(Arc<String>);
impl AsRef<str> for ArcWrap {
    fn as_ref(&self) -> &str {
        self.0.as_ref()
    }
}

/// Main FeatureStore struct.  We use a vector of vectors to allow for dynamic numbers of features.
/// This can and should be updated to a more memory friendly approach since vectors have surprising
/// overhead and the number of discrete features tends to be relatively small.
#[derive(Debug)]
pub struct FeatureStore {
    /// Raw storage for features, indexed by node id
    features: Vec<Vec<usize>>,

    /// Maps a raw feature to a feature_id
    feature_vocab: Vocab,
}

impl FeatureStore {

    pub fn new(size: usize) -> Self {
        FeatureStore {
            features: vec![Vec::with_capacity(0); size],
            feature_vocab: Vocab::new(),
        }
    }

    pub fn set_features<A,B>(
        &mut self, 
        node: NodeID, 
        node_features: impl Iterator<Item=(A, B)>
    )
        where
            A: AsRef<str>,
            B: AsRef<str>
    {
        self.features[node] = node_features
            .map(|(ft, fname)| self.feature_vocab.get_or_insert(ft.as_ref(), fname.as_ref()))
            .collect()
    }

    pub fn set_features_raw(
        &mut self, 
        node: NodeID, 
        node_features: impl Iterator<Item=usize>
    ) {
        self.features[node].extend(node_features);
    }

    pub fn get_features(&self, node: NodeID) -> &[usize] {
        &self.features[node]
    }

    fn get_pretty_feature(&self, feat_id: usize) -> (String, String) {
        let (nt, name) = self.feature_vocab.get_name(feat_id).unwrap();
        (nt.to_string(), name.to_string())
    }

    pub fn get_pretty_features(&self, node: NodeID) -> Vec<(String, String)> {
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
                self.set_features(i, [("node", i.to_string())].into_iter());
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

        let mut new_fs = FeatureStore::new(self.features.len());
        
        // Filter out features that don't meet the min_count
        self.features.iter().enumerate().for_each(|(node_id, feats)| {
            let new_feats = feats.iter()
                .filter(|f_i| counts[**f_i] >= count)
                .map(|f_i| {
                    let (nt, nn) = self.feature_vocab.get_name(*f_i)
                        .expect("Should never be unavailable!");
                    (ArcWrap(nt), nn)
                });

            new_fs.set_features(node_id, new_feats);
        });
        new_fs
    }

}

