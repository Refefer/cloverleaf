//! Aggregators take models, feature embeddings, and a feature set and convert them into
//! embeddings.  They are constructed adhoc so can be used in parallel as well as within the python
//! interface
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::sync::Arc;

use rayon::prelude::*;

use crate::feature_store::{FeatureStore, FeatureSet};
use crate::embeddings::{EmbeddingStore, NodeEmbeddings};
use crate::distance::Distance;
use crate::error::Error;
use crate::vocab::Vocab;
use crate::algos::ep::attention::AttentionWeights;

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

/// Attention pooling with trained weights (see [`AttentionWeights`]); every feature has weight 1.
pub struct AttentionAggregator<'a> {
    embs: &'a EmbeddingStore,
    weights: &'a AttentionWeights
}

impl <'a> AttentionAggregator<'a> {
    pub fn new(embs: &'a EmbeddingStore, weights: &'a AttentionWeights) -> Self {
        AttentionAggregator { embs, weights }
    }
}

impl <'a> EmbeddingBuilder for AttentionAggregator<'a> {
    fn construct(&self, features: &[usize], out: &mut [f32]) {
        let x: Vec<&[f32]> = features.iter().map(|feat_id| self.embs.get_embedding(*feat_id)).collect();
        self.weights.pool(&x, &vec![1f32; x.len()], out);
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
    Attention(Arc<AttentionWeights>)
}

impl FeatureAggregator {
    /// The mean of the feature embeddings.
    pub fn averaged() -> Self {
        FeatureAggregator(AggregatorKind::Averaged)
    }

    /// Attention pooling with trained weights, from EP's attention model or the GPU trainer.
    pub fn attention(weights: AttentionWeights) -> Self {
        FeatureAggregator(AggregatorKind::Attention(Arc::new(weights)))
    }

    /// The attention weights, for an attention aggregator.
    pub fn attention_weights(&self) -> Option<&AttentionWeights> {
        match &self.0 {
            AggregatorKind::Attention(weights) => Some(weights),
            _ => None
        }
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
            AggregatorKind::Attention(w) => {
                // Each row of a projection on one line; ryu writes the shortest text that reads
                // back as the same f32.
                writeln!(bw, "Attention")?;
                writeln!(bw, "{}\t{}\t{}", w.dims(), w.heads(), w.head_dims())?;
                let mut buf = ryu::Buffer::new();
                for (rows, width) in [(w.qkv(), w.dims()), (w.out(), w.heads() * w.head_dims())] {
                    for row in rows.chunks_exact(width) {
                        let mut first = true;
                        for v in row {
                            if !first { bw.write_all(b" ")?; }
                            bw.write_all(buf.format_finite(*v).as_bytes())?;
                            first = false;
                        }
                        bw.write_all(b"\n")?;
                    }
                }
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
                let (i, l) = next_line("dims, heads and head_dims")?;
                let shape: Vec<&str> = l.split('\t').collect();
                if shape.len() != 3 {
                    return Err(Error::Malformed { line: i, message: if shape.len() == 1 {
                        "this attention aggregator is from Cloverleaf's earlier attention, which \
                         stored projections inside each feature embedding; it can't be loaded, \
                         retrain the model".into()
                    } else {
                        "expected dims<TAB>heads<TAB>head_dims".into()
                    }})
                }
                let dims: usize = parse(i, shape[0], "dims")?;
                let heads: usize = parse(i, shape[1], "heads")?;
                let head_dims: usize = parse(i, shape[2], "head_dims")?;
                let inner = heads * head_dims;
                let mut read_rows = |rows: usize, width: usize, what: &str| -> Result<Vec<f32>, Error> {
                    let mut values = Vec::with_capacity(rows * width);
                    for _ in 0..rows {
                        let (i, l) = next_line(what)?;
                        let before = values.len();
                        for v in l.split_ascii_whitespace() {
                            values.push(parse::<f32>(i, v, what)?);
                        }
                        if values.len() - before != width {
                            return Err(Error::Malformed { line: i, message: format!(
                                "{} row has {} values, expected {}", what, values.len() - before, width) })
                        }
                    }
                    Ok(values)
                };
                let qkv = read_rows(3 * inner, dims, "qkv")?;
                let out = read_rows(dims, inner, "out")?;
                Ok(FeatureAggregator::attention(AttentionWeights::new(dims, heads, head_dims, qkv, out)?))
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
            AggregatorKind::Attention(weights) => Box::new(AttentionAggregator::new(es, weights))
        }
    }

    /// Dimensions of the node embeddings built from `feature_dims`-dimensional feature
    /// embeddings: the same, and for attention they must be the width its weights were trained
    /// for.
    pub fn node_dims(&self, feature_dims: usize) -> Result<usize, Error> {
        match &self.0 {
            AggregatorKind::Attention(w) if w.dims() != feature_dims =>
                Err(Error::InvalidArgument(format!(
                    "attention weights are for {}-dimensional feature embeddings, got {}",
                    w.dims(), feature_dims))),
            _ => Ok(feature_dims)
        }
    }
}

impl std::fmt::Display for FeatureAggregator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            AggregatorKind::Averaged => write!(f, "Averaged"),
            AggregatorKind::Weighted { alpha, .. } => write!(f, "Weighted<alpha={}>", alpha),
            AggregatorKind::Attention(w) =>
                write!(f, "Attention<dims={},heads={},head_dims={}>", w.dims(), w.heads(), w.head_dims())
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
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

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
        for agg in [FeatureAggregator::averaged(), attention(), FeatureAggregator::weighted(0.25, &fs)] {
            let path = temp_path("rt");
            agg.save(&path).unwrap();
            let loaded = FeatureAggregator::load(&path).unwrap();
            std::fs::remove_file(&path).unwrap();
            assert_eq!(loaded.to_string(), agg.to_string());
            // Attention weights read back bit-exactly.
            assert_eq!(agg.attention_weights(), loaded.attention_weights());
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

    /// Attention weights for 3-dimensional embeddings, 2 heads of width 8, with values that
    /// need all of f32's digits to round trip.
    fn attention() -> FeatureAggregator {
        let mut rng = XorShiftRng::seed_from_u64(7);
        let w = AttentionWeights::init(3, 2, &mut rng).unwrap();
        let out = (0..3 * 16).map(|i| (i as f32 * 0.1234567).sin() / 3.0).collect();
        FeatureAggregator::attention(AttentionWeights::new(3, 2, 8, w.qkv().to_vec(), out).unwrap())
    }

    /// Attention embeddings are as wide as the feature embeddings its weights were trained on.
    #[test]
    fn test_attention_node_dims() {
        let agg = attention();
        assert_eq!(agg.node_dims(3).unwrap(), 3);
        assert!(agg.node_dims(4).is_err());
    }

    /// Files from the earlier attention (num_heads, d_k and window on their own lines) are
    /// rejected with a message saying why, as are truncated files.
    #[test]
    fn test_load_rejects_old_and_truncated_attention() {
        let path = temp_path("old-att");
        std::fs::write(&path, "Attention\n2\n3\n0\n").unwrap();
        let res = FeatureAggregator::load(&path);
        assert!(matches!(&res, Err(Error::Malformed { message, .. }) if message.contains("retrain")));
        attention().save(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let truncated: Vec<&str> = text.lines().take(10).collect();
        std::fs::write(&path, truncated.join("\n")).unwrap();
        assert!(FeatureAggregator::load(&path).is_err());
        std::fs::remove_file(&path).unwrap();
    }

    /// The aggregator embeds with its weights: adhoc and per-node agree with `pool`.
    #[test]
    fn test_attention_embed_adhoc_uses_the_weights() {
        let mut b = crate::embeddings::NodeEmbeddingsBuilder::new(Distance::Cosine);
        b.add_embedding("f", "a", &[1., 0., 2.]).unwrap();
        b.add_embedding("f", "b", &[0., 1., -1.]).unwrap();
        let fe = b.build().unwrap();
        let agg = attention();
        let mut expected = vec![0f32; 3];
        agg.attention_weights().unwrap().pool(&[&[1., 0., 2.], &[0., 1., -1.]], &[1., 1.], &mut expected);
        let embedder = NodeEmbedder::new(agg);
        assert_eq!(embedder.embed_adhoc(&[("f", "a"), ("f", "b")], &fe, true).unwrap(), expected);
        assert_ne!(expected, vec![0.5, 0.5, 0.5]);   // not just the mean
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
