//! Defines the FeatureStore class which is used to define discrete features for each node
use std::sync::Arc;
use std::io::BufRead;

use crate::NodeID;
use crate::embeddings::NodeEmbeddings;
use crate::error::{Error, ensure_aligned};
use crate::io::{FeatureFileError, open_file_for_reading, read_feature_vocab};
use crate::named_graph::NamedGraph;
use crate::vocab::Vocab;

/// Makes it compatible for with feature setting
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

    /// [`set_features`](Self::set_features), keeping only the first occurrence of each feature.
    pub fn set_features_unique<A,B>(
        &mut self,
        node: NodeID,
        node_features: impl Iterator<Item=(A, B)>
    )
        where
            A: AsRef<str>,
            B: AsRef<str>
    {
        let mut ids: Vec<usize> = Vec::new();
        for (ft, fname) in node_features {
            let id = self.feature_vocab.get_or_insert(ft.as_ref(), fname.as_ref());
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        self.features[node] = ids;
    }

    /// Every node's features as flat arrays: offsets (`num_nodes() + 1` of them; node v's
    /// features are `ids[offsets[v]..offsets[v + 1]]`) and feature ids, in stored order.
    pub fn csr(&self) -> (Vec<usize>, Vec<usize>) {
        let mut offsets = Vec::with_capacity(self.features.len() + 1);
        offsets.push(0);
        let mut ids = Vec::with_capacity(self.features.iter().map(|f| f.len()).sum());
        for fs in self.features.iter() {
            ids.extend_from_slice(fs);
            offsets.push(ids.len());
        }
        (offsets, ids)
    }

    pub fn set_features_raw(
        &mut self, 
        node: NodeID, 
        node_features: impl Iterator<Item=usize>
    ) {
        self.features[node] = node_features.collect();
    }

    pub fn get_features(&self, node: NodeID) -> &[usize] {
        &self.features[node]
    }

    pub fn get_pretty_feature(&self, feat_id: usize) -> (String, String) {
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


/// How tokens in a feature file get their feature type.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct FeatureNamespace(Namespace);

#[derive(Clone,Debug,PartialEq,Eq)]
enum Namespace {
    Static(String),
    NodeType,
    Prefix(String),
    NodeTypeAndShared(String)
}

impl FeatureNamespace {
    /// Every feature gets the type `ns`.
    pub fn single(ns: impl Into<String>) -> Self {
        FeatureNamespace(Namespace::Static(ns.into()))
    }

    /// Features take the type of the node they belong to.
    pub fn node_type() -> Self {
        FeatureNamespace(Namespace::NodeType)
    }

    /// Each token is `type<delim>name`.  Tokens without the delimiter get the type "feat".
    pub fn prefix(delim: impl Into<String>) -> Result<Self, Error> {
        let delim = delim.into();
        if delim.is_empty() {
            return Err(Error::InvalidArgument("Prefix delimiter must not be empty".into()))
        }
        Ok(FeatureNamespace(Namespace::Prefix(delim)))
    }

    /// Each token becomes two features: one of its node's type, as with
    /// [`node_type`](Self::node_type), and one of type `shared`, the same for every node type.
    /// So a word in a query and the same word in a product title share a feature as well as
    /// having their own.  `shared` can't be "node" (the type of the features
    /// [`FeatureSet::fill_missing_nodes`] adds) or empty, and loading rejects it if it's also
    /// a node type.
    pub fn node_type_and_shared(shared: impl Into<String>) -> Result<Self, Error> {
        let shared = shared.into();
        if shared.is_empty() || shared == "node" {
            return Err(Error::InvalidArgument(format!(
                "the shared namespace can't be empty or \"node\", got {:?}", shared)))
        }
        Ok(FeatureNamespace(Namespace::NodeTypeAndShared(shared)))
    }

    /// Splits a token into (feature type, feature name), or None when a prefix is missing.  With
    /// [`node_type_and_shared`](Self::node_type_and_shared) this is the node type's feature;
    /// [`keys`](Self::keys) gives both.
    pub fn parse<'a>(&'a self, node_type: &'a str, token: &'a str) -> Option<(&'a str, &'a str)> {
        match &self.0 {
            Namespace::Static(s) => Some((s, token)),
            Namespace::NodeType | Namespace::NodeTypeAndShared(_) => Some((node_type, token)),
            Namespace::Prefix(delim) => token.find(delim.as_str())
                .map(|idx| (&token[..idx], &token[(idx + delim.len())..]))
        }
    }

    /// Every feature a token becomes: one, or two with
    /// [`node_type_and_shared`](Self::node_type_and_shared).  A token missing its prefix gets
    /// the type "feat".
    pub fn keys<'a>(&'a self, node_type: &'a str, token: &'a str) -> impl Iterator<Item=(&'a str, &'a str)> {
        let first = self.parse(node_type, token).unwrap_or(("feat", token));
        let second = match &self.0 {
            Namespace::NodeTypeAndShared(shared) => Some((shared.as_str(), token)),
            _ => None
        };
        std::iter::once(first).chain(second)
    }

    /// The shared namespace's name, for [`node_type_and_shared`](Self::node_type_and_shared).
    pub fn shared(&self) -> Option<&str> {
        match &self.0 {
            Namespace::NodeTypeAndShared(shared) => Some(shared),
            _ => None
        }
    }
}

/// How [`FeatureSet::load_into_with`] reads a feature file.
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct LoadOptions {
    /// Keep only the first occurrence of each feature on a node, so a word repeated on a line
    /// counts once (in means and in [`FeatureSet::prune_min_count`]'s counts).  Off by default:
    /// repeats count every time.
    pub dedup: bool
}

/// Features type "feat".
impl Default for FeatureNamespace {
    fn default() -> Self {
        FeatureNamespace::single("feat")
    }
}

/// Discrete features for named nodes: a [`FeatureStore`] indexed by the ids of a node [`Vocab`].
pub struct FeatureSet {
    pub(crate) vocab: Arc<Vocab>,
    pub(crate) features: FeatureStore
}

impl FeatureSet {
    /// An empty feature set for every node in `vocab`.
    pub fn new(vocab: Arc<Vocab>) -> Self {
        let features = FeatureStore::new(vocab.len());
        FeatureSet { vocab, features }
    }

    /// An empty feature set for the graph's nodes.
    pub fn from_graph(graph: &NamedGraph) -> Self {
        FeatureSet::new(graph.vocab().clone())
    }

    /// Reads a `node_type<TAB>name<TAB>f1 f2 ...` file; its nodes become the vocab.
    pub fn from_file(path: &str, namespace: &FeatureNamespace) -> Result<Self, FeatureFileError> {
        let mut fs = FeatureSet::new(Arc::new(read_feature_vocab(path)?));
        fs.load_into(path, namespace)?;
        Ok(fs)
    }

    /// Sets features from a `node_type<TAB>name<TAB>f1 f2 ...` file.  Nodes not in the vocab are
    /// skipped.
    pub fn load_into(&mut self, path: &str, namespace: &FeatureNamespace) -> Result<(), FeatureFileError> {
        self.load_into_with(path, namespace, &LoadOptions::default())
    }

    /// [`load_into`](Self::load_into) with options.
    pub fn load_into_with(
        &mut self,
        path: &str,
        namespace: &FeatureNamespace,
        options: &LoadOptions
    ) -> Result<(), FeatureFileError> {
        if let Some(shared) = namespace.shared() {
            if self.vocab.has_node_type_named(shared) {
                return Err(FeatureFileError::SharedNamespaceClash(shared.to_string()))
            }
        }
        let reader = open_file_for_reading(path).map_err(FeatureFileError::Io)?;
        for (i, line) in reader.lines().enumerate() {
            let line = line.map_err(FeatureFileError::Io)?;
            let pieces: Vec<_> = line.split('\t').collect();
            if pieces.len() != 3 {
                return Err(FeatureFileError::Malformed(i + 1))
            }
            if let Some(node_id) = self.vocab.get_node_id(pieces[0], pieces[1]) {
                let bow = pieces[2].split_whitespace().flat_map(|s| namespace.keys(pieces[0], s));
                if options.dedup {
                    self.features.set_features_unique(node_id, bow);
                } else {
                    self.features.set_features(node_id, bow);
                }
            }
        }
        Ok(())
    }

    /// Replaces a node's features with `(feature type, feature name)` pairs.
    pub fn set_features<A: AsRef<str>, B: AsRef<str>>(
        &mut self,
        node_type: &str,
        name: &str,
        features: impl Iterator<Item=(A, B)>
    ) -> Result<(), Error> {
        let node_id = self.vocab.node_id(node_type, name)?;
        self.features.set_features(node_id, features);
        Ok(())
    }

    /// A node's features as `(feature type, feature name)` pairs.
    pub fn get_features(&self, node_type: &str, name: &str) -> Result<Vec<(String, String)>, Error> {
        Ok(self.features.get_pretty_features(self.vocab.node_id(node_type, name)?))
    }

    /// A copy without the features that occur fewer than `count` times.
    pub fn prune_min_count(&self, count: usize) -> Self {
        FeatureSet { vocab: self.vocab.clone(), features: self.features.prune_min_count(count) }
    }

    /// Gives every node without features a feature of its own, as embedding requires.
    pub fn fill_missing_nodes(&mut self) {
        self.features.fill_missing_nodes();
    }

    /// Readies the features for learning feature embeddings over `graph`: checks they're indexed
    /// by the graph's nodes, gives featureless nodes a feature of their own, then checks that
    /// any warm-start feature embeddings are indexed by the resulting features.
    pub fn prepare_for_learning(
        &mut self,
        graph: &NamedGraph,
        warm_start: Option<&NodeEmbeddings>
    ) -> Result<(), Error> {
        ensure_aligned(&graph.vocab, &self.vocab, "Graph and FeatureSet")?;
        self.features.fill_missing_nodes();
        if let Some(fes) = warm_start {
            ensure_aligned(self.features.get_vocab(), &fes.vocab,
                           "FeatureSet features and warm-start feature embeddings")?;
        }
        Ok(())
    }

    pub fn vocab(&self) -> &Arc<Vocab> {
        &self.vocab
    }

    pub fn store(&self) -> &FeatureStore {
        &self.features
    }

    pub fn nodes(&self) -> usize {
        self.vocab.len()
    }

    pub fn num_features(&self) -> usize {
        self.features.num_features()
    }
}

#[cfg(test)]
mod feature_set_tests {
    use super::*;

    /// The name starts after the whole delimiter, however many bytes it is.
    #[test]
    fn test_prefix_namespace_multibyte_delimiter() {
        let ns = FeatureNamespace::prefix("::").unwrap();
        assert_eq!(ns.parse("n", "color::red"), Some(("color", "red")));
        let ns = FeatureNamespace::prefix("\u{2192}").unwrap();
        assert_eq!(ns.parse("n", "color\u{2192}red"), Some(("color", "red")));
        assert_eq!(ns.parse("n", "plain"), None);
    }

    #[test]
    fn test_prefix_namespace_rejects_empty_delimiter() {
        assert!(FeatureNamespace::prefix("").is_err());
    }

    /// Lines for nodes outside the vocab are skipped; prefixless tokens get the type "feat".
    #[test]
    fn test_load_into() {
        let mut vocab = Vocab::new();
        vocab.get_or_insert("n", "a");
        let mut fs = FeatureSet::new(Arc::new(vocab));
        let path = std::env::temp_dir().join(format!("cloverleaf-fs-{}", std::process::id()));
        std::fs::write(&path, "n\ta\tcolor:red big\nn\tzzz\tcolor:blue\n").unwrap();
        let res = fs.load_into(path.to_str().unwrap(), &FeatureNamespace::prefix(":").unwrap());
        std::fs::remove_file(&path).unwrap();
        res.unwrap();
        assert_eq!(fs.get_features("n", "a").unwrap(),
                   vec![("color".to_string(), "red".to_string()), ("feat".to_string(), "big".to_string())]);
        assert!(matches!(fs.get_features("n", "zzz"), Err(Error::UnknownNode { .. })));
    }

    fn two_type_set(lines: &str, ns: &FeatureNamespace, dedup: bool, tag: &str) -> Result<FeatureSet, FeatureFileError> {
        let mut vocab = Vocab::new();
        vocab.get_or_insert("q", "a");
        vocab.get_or_insert("p", "b");
        vocab.get_or_insert("p", "c");
        let mut fs = FeatureSet::new(Arc::new(vocab));
        let path = std::env::temp_dir().join(format!("cloverleaf-fs-{}-{}", std::process::id(), tag));
        std::fs::write(&path, lines).unwrap();
        let res = fs.load_into_with(path.to_str().unwrap(), ns, &LoadOptions { dedup });
        std::fs::remove_file(&path).unwrap();
        res.map(|_| fs)
    }

    fn pairs(v: &[(&str, &str)]) -> Vec<(String, String)> {
        v.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
    }

    /// Repeats count every time by default, once with dedup, in first-seen order.
    #[test]
    fn test_load_dedup() {
        let ns = FeatureNamespace::node_type();
        let kept = two_type_set("q\ta\tx y x\n", &ns, false, "keep").unwrap();
        assert_eq!(kept.get_features("q", "a").unwrap(), pairs(&[("q", "x"), ("q", "y"), ("q", "x")]));
        let deduped = two_type_set("q\ta\tx y x\n", &ns, true, "dedup").unwrap();
        assert_eq!(deduped.get_features("q", "a").unwrap(), pairs(&[("q", "x"), ("q", "y")]));
        // Pruning counts what was kept: x on one node counts once with dedup.
        assert_eq!(deduped.prune_min_count(2).num_features(), 0);
        assert_eq!(kept.prune_min_count(2).num_features(), 1);
    }

    /// Each token gets its node type's feature and the shared one, so the same word on two node
    /// types shares a feature.
    #[test]
    fn test_node_type_and_shared_namespace() {
        let ns = FeatureNamespace::node_type_and_shared("w").unwrap();
        assert_eq!(ns.keys("q", "red").collect::<Vec<_>>(), vec![("q", "red"), ("w", "red")]);
        let fs = two_type_set("q\ta\tred\np\tb\tred shoe\n", &ns, true, "shared").unwrap();
        assert_eq!(fs.get_features("q", "a").unwrap(), pairs(&[("q", "red"), ("w", "red")]));
        assert_eq!(fs.get_features("p", "b").unwrap(), pairs(&[("p", "red"), ("w", "red"), ("p", "shoe"), ("w", "shoe")]));
        assert_eq!(fs.num_features(), 5);
        // Only "w red" is on two nodes.
        assert_eq!(fs.prune_min_count(2).num_features(), 1);
    }

    #[test]
    fn test_shared_namespace_rejects_clashes() {
        assert!(FeatureNamespace::node_type_and_shared("node").is_err());
        assert!(FeatureNamespace::node_type_and_shared("").is_err());
        let clash = FeatureNamespace::node_type_and_shared("p").unwrap();
        assert!(matches!(two_type_set("q\ta\tred\n", &clash, false, "clash"),
                         Err(FeatureFileError::SharedNamespaceClash(ns)) if ns == "p"));
    }

    /// Flat arrays: offsets per node and feature ids in stored order; filled-in nodes included.
    #[test]
    fn test_csr_and_fill_missing_nodes() {
        let mut fs = two_type_set("q\ta\tx y\np\tc\ty\n", &FeatureNamespace::single("f"), false, "csr").unwrap();
        assert_eq!(fs.store().csr(), (vec![0, 2, 2, 3], vec![0, 1, 1]));
        fs.fill_missing_nodes();
        let (offsets, ids) = fs.store().csr();
        assert_eq!(offsets, vec![0, 2, 3, 4]);
        assert_eq!(fs.store().get_pretty_feature(ids[2]), ("node".to_string(), "1".to_string()));
    }
}
