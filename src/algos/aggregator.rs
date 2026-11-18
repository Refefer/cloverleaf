//! Aggregators take models, feature embeddings, and a feature set and convert them into
//! embeddings.  They are constructed adhoc so can be used in parallel as well as within the python
//! interface
use crate::autograd::*;
use rand::prelude::*;
use rand_xorshift::XorShiftRng;

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::sync::Arc;

use rayon::prelude::*;

use crate::feature_store::{FeatureStore, FeatureSet};
use crate::embeddings::{EmbeddingStore, NodeEmbeddings};
use crate::distance::Distance;
use crate::error::Error;
use crate::vocab::Vocab;
use crate::algos::ep::attention::{attention_mean,AttentionType,MultiHeadedAttention};

pub trait EmbeddingBuilder {
    fn construct( &self, features: &[usize], out: &mut [f32]) -> ();
}

/// Simply averages feature embeddings together to create a new embedding
pub struct AvgAggregator<'a> {
    embs: &'a EmbeddingStore
}

impl <'a> AvgAggregator<'a> {
    pub fn new(embs: &'a EmbeddingStore) -> Self {
        AvgAggregator { embs }
    }

    #[inline]
    fn isum(out: &mut [f32], x: &[f32]) {
        out.iter_mut().zip(x.iter()).for_each(|(outi, xi)| {
            *outi += xi;
        });
    }
}
impl <'a> EmbeddingBuilder for AvgAggregator<'a> {
    fn construct(
        &self, 
        features: &[usize],
        out: &mut [f32]
    ) {
        out.fill(0f32);
        for feat_id in features {
            let e = self.embs.get_embedding(*feat_id); 
            AvgAggregator::isum(out, e);
        }

        let len = features.len() as f32;
        out.iter_mut().for_each(|outi| *outi /= len);
    }
}

/// Computes the P(feature) for a feature set.  
pub struct UnigramProbability {
    p_w: Vec<f32>
}

impl UnigramProbability {
    pub fn new(features: &FeatureStore) -> Self{
        let mut counts = vec![0usize; features.num_features()];
        let mut total = 0usize;
        for feats in features.iter() {
            for feat_id in feats.iter() {
                counts[*feat_id] += 1;
                total += 1;
            }
        }

        let p_w = counts.into_iter()
            .map(|c| c as f32 / total as f32)
            .collect();

        UnigramProbability { p_w }
    }

    pub fn from_vec(p_w: Vec<f32>) -> Self {
        UnigramProbability { p_w }
    }

    pub fn iter(&self) -> impl Iterator<Item=&f32> {
        self.p_w.iter()
    }
}

/// Uses IDF to weight the features so rarer tokens have a larger impact on the embedding.
pub struct WeightedAggregator<'a> {
    embs: &'a EmbeddingStore,
    up: &'a UnigramProbability,
    alpha: f32
}

impl <'a> WeightedAggregator<'a> {
    pub fn new(embs: &'a EmbeddingStore, up: &'a UnigramProbability, alpha: f32) -> Self {
        WeightedAggregator { embs, up, alpha }
    }

    #[inline]
    fn imulsum(out: &mut [f32], x: &[f32], scalar: f32) {
        out.iter_mut().zip(x.iter()).for_each(|(outi, xi)| {
            *outi += scalar * xi;
        });
    }
}

impl <'a> EmbeddingBuilder for WeightedAggregator<'a> {
    fn construct(
        &self, 
        features: &[usize],
        out: &mut [f32]
    ) {
        out.fill(0f32);
        let mut weight = 0f32;
        for feat_id in features.iter() {
            let p_wi = self.up.p_w[*feat_id];
            let w = self.alpha / (self.alpha + p_wi);
            weight += w;
            let e = self.embs.get_embedding(*feat_id); 
            WeightedAggregator::imulsum(out, e, w);
        }

        out.iter_mut().for_each(|outi| *outi /= weight);
    }
}

/// Constructs an embedding from a sequenced feature set using attention.  Expensive but relatively
/// effective for text features
pub struct AttentionAggregator<'a> {
    embs: &'a EmbeddingStore,
    mha: MultiHeadedAttention
}

impl <'a> AttentionAggregator<'a> {
    pub fn new(embs: &'a EmbeddingStore, mha: MultiHeadedAttention) -> Self {
        AttentionAggregator { embs, mha }
    }
}

impl <'a> EmbeddingBuilder for AttentionAggregator<'a> {
    fn construct(
        &self, 
        features: &[usize],
        out: &mut [f32]
    ) {
        out.fill(0f32);
        let it = features.iter().map(|feat_id| {
            let e = self.embs.get_embedding(*feat_id); 
            (Constant::new(e.to_vec()), 1f32)
        }).collect::<Vec<_>>();

        // No-op RNG
        let mut rng = XorShiftRng::seed_from_u64(0);
        let v = attention_mean(it.iter(), &self.mha, &mut rng);
        v.value().iter().zip(out.iter_mut()).for_each(|(vi, outi)| {
            *outi = *vi;
        });
    }
}

/// How a node's feature embeddings combine into its embedding.  Can be saved alongside the
/// feature embeddings it was trained with.
#[derive(Clone)]
pub struct FeatureAggregator(AggregatorKind);

#[derive(Clone)]
enum AggregatorKind {
    Averaged,
    /// The unigram probabilities are indexed by the ids of `vocab`, the feature vocab they were
    /// counted over; it's kept so they can be saved by name.
    Weighted { alpha: f32, vocab: Arc<Vocab>, unigrams: Arc<UnigramProbability> },
    Attention { num_heads: usize, d_k: usize, window: Option<usize> }
}

impl FeatureAggregator {
    /// The mean of the feature embeddings.
    pub fn averaged() -> Self {
        FeatureAggregator(AggregatorKind::Averaged)
    }

    /// Attention over the features, for embeddings trained with an attention model.  `window`
    /// limits attention to a sliding window of features.
    pub fn attention(num_heads: usize, d_k: usize, window: Option<usize>) -> Self {
        FeatureAggregator(AggregatorKind::Attention { num_heads, d_k, window })
    }

    /// A weighted mean that favours rare features (SIF weighting), with feature frequencies
    /// counted over `features`.
    pub fn weighted(alpha: f32, features: &FeatureSet) -> Self {
        let unigrams = Arc::new(UnigramProbability::new(&features.features));
        let vocab = Arc::new(features.features.clone_vocab());
        FeatureAggregator(AggregatorKind::Weighted { alpha, vocab, unigrams })
    }

    /// Writes the aggregator in the format `load` reads.
    pub fn save(&self, path: &str) -> Result<(), Error> {
        let mut bw = BufWriter::new(File::create(path)?);
        match &self.0 {
            AggregatorKind::Averaged => writeln!(bw, "Averaged")?,
            AggregatorKind::Attention { num_heads, d_k, window } => {
                writeln!(bw, "Attention")?;
                writeln!(bw, "{}", num_heads)?;
                writeln!(bw, "{}", d_k)?;
                writeln!(bw, "{}", window.unwrap_or(0))?;
            },
            AggregatorKind::Weighted { alpha, vocab, unigrams } => {
                writeln!(bw, "Weighted")?;
                writeln!(bw, "{}", alpha)?;
                for (feat_id, p_wi) in unigrams.iter().enumerate() {
                    let (f_node_type, f_name) = vocab.get_name(feat_id).ok_or_else(||
                        Error::InvalidArgument("feature vocab and unigrams mismatch".into()))?;
                    writeln!(bw, "{}\t{}\t{}", f_node_type, f_name, *p_wi as f64)?;
                }
            }
        }
        bw.flush()?;
        Ok(())
    }

    /// Reads an aggregator written by `save`.
    pub fn load(path: &str) -> Result<Self, Error> {
        let mut lines = BufReader::new(File::open(path)?).lines().enumerate();
        let mut next_line = |what: &str| -> Result<(usize, String), Error> {
            match lines.next() {
                Some((i, line)) => Ok((i + 1, line?)),
                None => Err(Error::Malformed { line: 0, message: format!("missing {}", what) })
            }
        };
        fn parse<T: std::str::FromStr>(line: usize, value: &str, what: &str) -> Result<T, Error> {
            value.trim_end().parse::<T>().map_err(|_| Error::Malformed {
                line, message: format!("invalid {}: {:?}", what, value) })
        }

        let (_, kind) = next_line("aggregator type")?;
        match kind.trim_end() {
            "Averaged" => Ok(FeatureAggregator::averaged()),
            "Attention" => {
                let (i, l) = next_line("num_heads")?;
                let num_heads = parse(i, &l, "num_heads")?;
                let (i, l) = next_line("d_k")?;
                let d_k = parse(i, &l, "d_k")?;
                let (i, l) = next_line("window")?;
                let window: usize = parse(i, &l, "window")?;
                Ok(FeatureAggregator::attention(num_heads, d_k, if window == 0 { None } else { Some(window) }))
            },
            "Weighted" => {
                let (i, l) = next_line("alpha")?;
                let alpha = parse(i, &l, "alpha")?;
                let mut vocab = Vocab::new();
                let mut p_w = Vec::new();
                for (i, line) in lines {
                    let line = line?;
                    let pieces: Vec<_> = line.split('\t').collect();
                    if pieces.len() != 3 {
                        return Err(Error::Malformed { line: i + 1, message:
                            "Malformed feature line! Need node_type<TAB>name<TAB>weight".into() })
                    }
                    let p_wi: f64 = parse(i + 1, pieces[2], "weight")?;
                    let feat_id = vocab.get_or_insert(pieces[0], pieces[1]);
                    if feat_id < p_w.len() {
                        return Err(Error::Malformed { line: i + 1, message: format!(
                            "Duplicate feature found: {} {}", pieces[0], pieces[1]) })
                    }
                    p_w.push(p_wi as f32);
                }
                let unigrams = Arc::new(UnigramProbability::from_vec(p_w));
                Ok(FeatureAggregator(AggregatorKind::Weighted { alpha, vocab: Arc::new(vocab), unigrams }))
            },
            other => Err(Error::Malformed { line: 1, message: format!("Unknown aggregator type: {}", other) })
        }
    }

    fn builder<'a>(&'a self, es: &'a EmbeddingStore) -> Box<dyn EmbeddingBuilder + Send + Sync + 'a> {
        match &self.0 {
            AggregatorKind::Weighted { alpha, unigrams, .. } =>
                Box::new(WeightedAggregator::new(es, unigrams, *alpha)),
            AggregatorKind::Averaged => Box::new(AvgAggregator::new(es)),
            AggregatorKind::Attention { num_heads, d_k, window } => {
                let at = match window {
                    Some(window_size) => AttentionType::Sliding { window_size: *window_size },
                    None => AttentionType::Full
                };
                Box::new(AttentionAggregator::new(es, MultiHeadedAttention::new(*num_heads, *d_k, at)))
            }
        }
    }

    /// Dimensions of the node embeddings built from `feature_dims`-dimensional feature
    /// embeddings.  Attention feature embeddings also hold the query and key projections.
    pub fn node_dims(&self, feature_dims: usize) -> Result<usize, Error> {
        match self.0 {
            AggregatorKind::Attention { num_heads, d_k, .. } => {
                let attention_dims = 2 * num_heads * d_k;
                if num_heads == 0 || feature_dims <= attention_dims {
                    return Err(Error::InvalidArgument(format!(
                        "{}-dimensional feature embeddings are too small for {} heads with d_k={}",
                        feature_dims, num_heads, d_k)))
                }
                Ok((feature_dims - attention_dims) / num_heads)
            },
            _ => Ok(feature_dims)
        }
    }
}

impl std::fmt::Display for FeatureAggregator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            AggregatorKind::Averaged => write!(f, "Averaged"),
            AggregatorKind::Weighted { alpha, .. } => write!(f, "Weighted<alpha={}>", alpha),
            AggregatorKind::Attention { num_heads, d_k, window } =>
                write!(f, "Attention<num_heads={},d_k={},window={:?}>", num_heads, d_k, window)
        }
    }
}

/// Builds node embeddings from feature embeddings.
#[derive(Clone)]
pub struct NodeEmbedder {
    aggregator: FeatureAggregator
}

impl NodeEmbedder {
    pub fn new(aggregator: FeatureAggregator) -> Self {
        NodeEmbedder { aggregator }
    }

    pub fn aggregator(&self) -> &FeatureAggregator {
        &self.aggregator
    }

    /// Embeds every node in `features`.  Features without an embedding are skipped, and a node
    /// with none left gets a zero embedding.
    pub fn embed_feature_set(
        &self,
        features: &FeatureSet,
        feature_embeddings: &NodeEmbeddings
    ) -> Result<NodeEmbeddings, Error> {
        let num_nodes = features.vocab.len();
        let dims = self.aggregator.node_dims(feature_embeddings.dims())?;
        let es = EmbeddingStore::new(num_nodes, dims, Distance::Cosine);
        let agg = self.aggregator.builder(&feature_embeddings.embeddings);
        let fs_vocab = features.features.get_vocab();
        (0..num_nodes).into_par_iter().for_each(|node| {
            let feats: Vec<_> = features.features.get_features(node).iter()
                .filter_map(|feat_id| feature_embeddings.vocab.translate_node(fs_vocab, *feat_id))
                .collect();
            if !feats.is_empty() {
                // Each node writes only its own row.
                agg.construct(&feats, es.get_embedding_mut_hogwild(node));
            }
        });
        Ok(NodeEmbeddings { vocab: features.vocab.clone(), embeddings: es })
    }

    /// Embeds one set of `(feature type, feature name)` features.  Unknown features are an error
    /// when `strict`, and skipped otherwise.
    pub fn embed_adhoc<A: AsRef<str>, B: AsRef<str>>(
        &self,
        features: &[(A, B)],
        feature_embeddings: &NodeEmbeddings,
        strict: bool
    ) -> Result<Vec<f32>, Error> {
        let mut ids = Vec::with_capacity(features.len());
        for (node_type, name) in features {
            match feature_embeddings.node_id(node_type.as_ref(), name.as_ref()) {
                Ok(feat_id) => ids.push(feat_id),
                Err(e) if strict => return Err(e),
                Err(_) => {}
            }
        }
        let mut embedding = vec![0.; self.aggregator.node_dims(feature_embeddings.dims())?];
        if !ids.is_empty() {
            self.aggregator.builder(&feature_embeddings.embeddings).construct(&ids, &mut embedding);
        }
        Ok(embedding)
    }

    /// `embed_adhoc` over many feature sets in parallel.
    pub fn bulk_embed_adhoc<A: AsRef<str> + Sync, B: AsRef<str> + Sync>(
        &self,
        feature_sets: &[Vec<(A, B)>],
        feature_embeddings: &NodeEmbeddings,
        strict: bool
    ) -> Result<Vec<Vec<f32>>, Error> {
        feature_sets.par_iter()
            .map(|features| self.embed_adhoc(features, feature_embeddings, strict))
            .collect()
    }
}

#[cfg(test)]
mod aggregator_tests {
    use super::*;

    use std::sync::Arc;

    fn feature_set() -> FeatureSet {
        let mut node_vocab = Vocab::new();
        node_vocab.get_or_insert("node", "n0");
        node_vocab.get_or_insert("node", "n1");
        let mut fs = FeatureSet::new(Arc::new(node_vocab));
        // More features than nodes.
        fs.set_features("node", "n0", [("f", "a"), ("f", "b"), ("f", "c")].into_iter()).unwrap();
        fs.set_features("node", "n1", [("f", "c")].into_iter()).unwrap();
        fs
    }

    fn temp_path(name: &str) -> String {
        std::env::temp_dir().join(format!("cloverleaf-agg-{}-{}", std::process::id(), name))
            .to_str().unwrap().to_string()
    }

    /// Unigram probabilities are indexed by feature id, so the stored vocab must be the feature
    /// vocab rather than the node vocab.
    #[test]
    fn test_weighted_aggregator_stores_feature_vocab() {
        let fs = feature_set();
        let agg = FeatureAggregator::weighted(0.5, &fs);
        let AggregatorKind::Weighted { vocab, .. } = &agg.0 else {
            panic!("expected a Weighted aggregator")
        };
        let feat_vocab = fs.features.get_vocab();
        assert_eq!(vocab.len(), feat_vocab.len());
        for feat_id in 0..feat_vocab.len() {
            assert_eq!(vocab.get_name(feat_id), feat_vocab.get_name(feat_id));
        }
    }

    /// Every kind of aggregator reads back as it was saved.
    #[test]
    fn test_save_load_round_trip() {
        let fs = feature_set();
        for agg in [FeatureAggregator::averaged(), FeatureAggregator::attention(2, 3, Some(4)),
                    FeatureAggregator::attention(1, 2, None), FeatureAggregator::weighted(0.25, &fs)] {
            let path = temp_path("rt");
            agg.save(&path).unwrap();
            let loaded = FeatureAggregator::load(&path).unwrap();
            std::fs::remove_file(&path).unwrap();
            assert_eq!(loaded.to_string(), agg.to_string());
            if let (AggregatorKind::Weighted { vocab: v1, unigrams: u1, .. },
                    AggregatorKind::Weighted { vocab: v2, unigrams: u2, .. }) = (&agg.0, &loaded.0) {
                assert_eq!(v1.len(), v2.len());
                assert!(u1.iter().zip(u2.iter()).all(|(a, b)| (a - b).abs() < 1e-6));
            }
        }
    }

    #[test]
    fn test_load_rejects_unknown_type() {
        let path = temp_path("bad");
        std::fs::write(&path, "Median\n").unwrap();
        let res = FeatureAggregator::load(&path);
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(res, Err(Error::Malformed { .. })));
    }

    /// Attention feature embeddings too small for the projections are an error, not an
    /// underflow.
    #[test]
    fn test_attention_node_dims() {
        let agg = FeatureAggregator::attention(2, 3, None);
        assert_eq!(agg.node_dims(2 * 2 * 3 + 2 * 5).unwrap(), 5);
        assert!(agg.node_dims(12).is_err());
        assert!(agg.node_dims(4).is_err());
    }

    /// Averaging two features gives their mean; unknown features are skipped unless strict.
    #[test]
    fn test_embed_adhoc() {
        let mut b = crate::embeddings::NodeEmbeddingsBuilder::new(Distance::Cosine);
        b.add_embedding("f", "a", &[1., 0.]).unwrap();
        b.add_embedding("f", "b", &[0., 1.]).unwrap();
        let fe = b.build().unwrap();
        let embedder = NodeEmbedder::new(FeatureAggregator::averaged());
        assert_eq!(embedder.embed_adhoc(&[("f", "a"), ("f", "b")], &fe, true).unwrap(), vec![0.5, 0.5]);
        assert!(matches!(embedder.embed_adhoc(&[("f", "zzz")], &fe, true), Err(Error::UnknownNode { .. })));
        assert_eq!(embedder.embed_adhoc(&[("f", "a"), ("f", "zzz")], &fe, false).unwrap(), vec![1., 0.]);
    }
}
