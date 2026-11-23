//! Reading and writing graphs, embeddings and feature files.  Files may be gzipped.
use crate::graph::NodeID;
use std::fs::File;
use std::io::{Write,BufWriter,Result as IOResult,BufReader,BufRead};
use std::convert::AsRef;
use std::collections::HashSet;

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use itertools::Itertools;
use rayon::prelude::*;
use ryu::Buffer;

use crate::vocab::Vocab;
use crate::error::Error;
use crate::embeddings::EmbeddingStore;
use crate::distance::Distance;
use crate::graph::{CSR,CumCSR,EdgeType};

/// Streaming writer for NodeEmbeddings.  Since Embeddings are often gigantic, creating them adhoc
/// then streaming them to disk is beneficial.
pub struct EmbeddingWriter<'a> {
    vocab: &'a Vocab,
    output: Box<dyn Write>,
    buffer: String
}

impl <'a> EmbeddingWriter<'a> {

    pub fn new(path: &str, vocab: &'a Vocab, comp_level: Option<u32>) -> IOResult<Self> {
        let encoder = open_file_for_writing(path, comp_level)?;
        let s = String::new();
        Ok(EmbeddingWriter { 
            vocab,
            output: encoder,
            buffer: s
        })
    }

    pub fn stream<A: AsRef<[f32]>>(
        &mut self, 
        it: impl Iterator<Item=(NodeID, A)>
    ) -> IOResult<()> {

        let mut formatter = Buffer::new();

        for (node_id, emb) in it {
            let (node_type, name) = self.vocab.get_name(node_id)
                .expect("Programming error!");

            // Build the embedding to string
            self.buffer.clear();

            // Write out embedding quickly
            EmbeddingWriter::format_embedding(&mut formatter, &mut self.buffer, emb.as_ref());

            // Spit it out 
            writeln!(&mut self.output, "{}\t{}\t[{}]", node_type, name, self.buffer)?

        }

        Ok(())
    }

    /// Flushes buffered output, reporting any write error that dropping the writer would hide.
    pub fn finish(mut self) -> IOResult<()> {
        self.output.flush()
    }

    fn format_embedding(buff: &mut Buffer, output: &mut String, emb: &[f32]) {
        for (idx, wi) in emb.iter().enumerate() {
            if idx > 0 {
                output.push_str(",");
            }
            output.push_str(buff.format(*wi));
        }
    }
}

struct RecordReader {
    chunk_size: usize,
    skip: usize
}

impl RecordReader {
    pub fn new(chunk_size: usize, skip: usize) -> Self {
        RecordReader { chunk_size, skip }
    }

    pub fn read<F: Sync,D,A:Send + Sync,E>(
        &self, 
        mut it: impl Iterator<Item=String>,
        mapper: F,
        mut drain: D
    ) -> Result<(),E>
        where F: Fn(usize, String) -> Option<A>,
              D: FnMut(usize, A) -> Result<(),E>
    {
        // Skip records, such as headers of tsvs
        (&mut it).take(self.skip).for_each(|_|{});

        let mut i = 0;
        if self.chunk_size <= 1 {
            for (i, line) in it.enumerate() {
                if let Some(record) = mapper(i, line) {
                    drain(i, record)?
                }
            }
        } else {
            let mut buffer = Vec::with_capacity(self.chunk_size);
            let mut p_buffer = Vec::with_capacity(self.chunk_size);
            for chunk in &(it).chunks(self.chunk_size) {
                buffer.clear();
                
                // Read lines into a buffer for parallelizing
                chunk.for_each(|l| buffer.push(l));

                buffer.par_drain(..).enumerate().map(|(idx, line)| {
                    mapper(i+idx, line)
                }).collect_into_vec(&mut p_buffer);

                for r in p_buffer.drain(..) {
                    if let Some(record) = r {
                        drain(i, record)?;
                    }
                    i += 1;
                }
            }
        }
        Ok(())
    }

}

pub struct EmbeddingReader; 

/// Errors from loading an embeddings file.
#[derive(Debug)]
pub enum EmbeddingLoadError {
    Io(std::io::Error),
    Parse(String),
    DuplicateNode(usize),
    DimensionMismatch
}


impl std::fmt::Display for EmbeddingLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingLoadError::Io(e) => write!(f, "{:?}", e),
            EmbeddingLoadError::Parse(line) => write!(f, "Error parsing line: {}", line),
            EmbeddingLoadError::DuplicateNode(i) => write!(f, "found duplicate node at {}!", i),
            EmbeddingLoadError::DimensionMismatch => write!(f, "Embeddings have different sizes!")
        }
    }
}

impl std::error::Error for EmbeddingLoadError {}

/// Errors reading a `node_type<TAB>name<TAB>features` file.
#[derive(Debug)]
pub enum FeatureFileError {
    Io(std::io::Error),
    /// 1-based line number
    Malformed(usize),
    /// A shared feature namespace with the same name as a node type, whose features would merge
    SharedNamespaceClash(String)
}


impl std::fmt::Display for FeatureFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeatureFileError::Io(e) => write!(f, "{:?}", e),
            FeatureFileError::Malformed(line) => write!(f,
                "Line {}: Malformed feature line! Need node_type<TAB>name<TAB>f1 f2 ...", line),
            FeatureFileError::SharedNamespaceClash(ns) => write!(f,
                "The shared feature namespace {:?} is also a node type; their features would merge", ns)
        }
    }
}

impl std::error::Error for FeatureFileError {}

/// Reads a `node_type<TAB>name<TAB>value` file, such as Policy Evaluation rewards, as
/// `(node id, value)` pairs.  Every node must be in `vocab`.
pub fn read_node_values(path: &str, vocab: &Vocab) -> Result<Vec<(NodeID, f32)>, Error> {
    let reader = open_file_for_reading(path)?;
    let mut values = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let pieces: Vec<_> = line.split('\t').collect();
        if pieces.len() != 3 {
            return Err(Error::Malformed { line: i + 1,
                message: "Malformed line! Need node_type<TAB>name<TAB>value".into() })
        }
        let value: f32 = pieces[2].trim().parse().map_err(|_| Error::Malformed {
            line: i + 1, message: format!("Invalid value: {}", pieces[2]) })?;
        values.push((vocab.node_id(pieces[0], pieces[1].trim())?, value));
    }
    Ok(values)
}

/// Reads the nodes of a feature file into a vocab, in file order.
pub fn read_feature_vocab(path: &str) -> Result<Vocab, FeatureFileError> {
    let mut vocab = Vocab::new();
    let reader = open_file_for_reading(path).map_err(FeatureFileError::Io)?;
    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(FeatureFileError::Io)?;
        let pieces: Vec<_> = line.split('\t').collect();
        if pieces.len() != 3 {
            return Err(FeatureFileError::Malformed(i + 1))
        }
        vocab.get_or_insert(pieces[0], pieces[1]);
    }
    Ok(vocab)
}

impl EmbeddingReader {

    /// Reads a `node_type<TAB>name<TAB>[e1,e2,...]` file, keeping only the node types in
    /// `filter_type` when given.
    pub fn load(
        path: &str, 
        distance: Distance, 
        filter_type: &Option<&HashSet<String>>, 
        chunk_size: Option<usize>,
        skip_rows: Option<usize>
    ) -> Result<(Vocab, EmbeddingStore), EmbeddingLoadError> {

        // Convert filter types to something more efficient
        let ft_strs = filter_type
            .map(|hs| hs.iter().map(|s| s.as_str()).collect());
        let ft_refs = ft_strs.as_ref();

        let num_embeddings = count_lines(path, ft_refs, skip_rows.unwrap_or(0))
            .map_err(EmbeddingLoadError::Io)?;

        let reader = open_file_for_reading(path)
            .map_err(EmbeddingLoadError::Io)?;

        let mut vocab = Vocab::new();
        
        // Place holder
        let mut es = EmbeddingStore::new(0, 0, Distance::Cosine);
        let rr = RecordReader::new(chunk_size.unwrap_or(1_000), skip_rows.unwrap_or(0));
        let mut i = 0;
        

        rr.read(reader.lines().map(|l| l.unwrap()),
            |_, line| {
               if let Some(node_types) = ft_refs {
                   if !is_node_type(&line, node_types) {
                       return None
                   }
               }

               Some(line_to_embedding(&line)
                    .ok_or_else(|| EmbeddingLoadError::Parse(line.clone())))
            },
            |_, record| {
                let (node_type, node_name, emb) = record?;

                if i == 0 {
                    es = EmbeddingStore::new(num_embeddings, emb.len(), distance);
                }

                let node_id = vocab.get_or_insert(node_type, node_name);
                if node_id < i {
                    return Err(EmbeddingLoadError::DuplicateNode(i));
                }
                let m = es.get_embedding_mut(node_id);
                if m.len() != emb.len() {
                    return Err(EmbeddingLoadError::DimensionMismatch);
                }
                m.copy_from_slice(&emb);
                i += 1;
                Ok(())
            })?;

        Ok((vocab, es))
    }
}

pub fn open_file_for_reading(path: &str) -> IOResult<Box<dyn BufRead>> {
    let f = File::open(path)?;

    let f = BufReader::new(f);
    let result: Box<dyn BufRead> = if path.ends_with(".gz") {
        let decoder = BufReader::new(GzDecoder::new(f));
        Box::new(decoder)
    } else {
        Box::new(f)
    };
    Ok(result)
}

/// Writes a row-major matrix of embeddings, `dims` floats per row, as `type<TAB>name<TAB>[e1,...]`
/// lines, the format `NodeEmbeddings::load` reads, one per key in order.  Rows are formatted in
/// parallel, a chunk at a time, so large exports are bounded by the disk rather than formatting.
/// A path ending in `.gz` is gzipped, at `comp_level` when given.
pub fn write_embedding_matrix<K: AsRef<str> + Sync>(
    path: &str,
    keys: &[(K, K)],
    values: &[f32],
    dims: usize,
    comp_level: Option<u32>
) -> Result<(), Error> {
    if values.len() != keys.len() * dims {
        return Err(Error::InvalidArgument(format!(
            "{} values can't be {} rows of {} dims", values.len(), keys.len(), dims)))
    }
    const CHUNK: usize = 16_384;
    let mut output = open_file_for_writing(path, comp_level)?;
    for (chunk_idx, chunk) in keys.chunks(CHUNK).enumerate() {
        let first = chunk_idx * CHUNK;
        let lines: Vec<String> = chunk.par_iter().enumerate().map(|(i, (node_type, name))| {
            let row = &values[(first + i) * dims..(first + i + 1) * dims];
            let mut line = String::with_capacity(dims * 12 + 32);
            line.push_str(node_type.as_ref());
            line.push('\t');
            line.push_str(name.as_ref());
            line.push_str("\t[");
            EmbeddingWriter::format_embedding(&mut Buffer::new(), &mut line, row);
            line.push_str("]\n");
            line
        }).collect();
        for line in lines {
            output.write_all(line.as_bytes())?;
        }
    }
    output.flush()?;
    Ok(())
}

pub fn open_file_for_writing(path: &str, compression: Option<u32>) -> IOResult<Box<dyn Write>> {
    let comp_level = compression.map(|l| Compression::new(l));
    let f = File::create(path)?;
    let bw = BufWriter::new(f);
    let encoder: Box<dyn Write> = if path.ends_with(".gz") {
        let e = GzEncoder::new(bw, comp_level.unwrap_or(Compression::fast()));
        Box::new(e)
    } else {
        Box::new(bw)
    };
    Ok(encoder)
}

fn is_node_type(line: &str, filter_types: &HashSet<&str>) -> bool {
    match line.find('\t') {
        Some(idx) => filter_types.contains(&line[..idx]),
        None => false
    }
}

/// Count the number of lines in an embeddings file so we only have to do one allocation.  If
/// NodeEmbeddings internal memory structure changes, such as using slabs, this might be less
/// relevant.
fn count_lines(path: &str, node_type: Option<&HashSet<&str>>, skip: usize) -> IOResult<usize> {
    let reader = open_file_for_reading(path)?;
    let mut count = 0;
    // Skipped rows (eg. headers) are never loaded, so don't count them.
    let lines = reader.lines().skip(skip);
    if let Some(fts) = node_type.as_ref() {
        for line in lines {
            let line = line?;
            if is_node_type(&line, fts) { count += 1}
        }
    } else {
        for line in lines {
            line?;
            count += 1;
        }
    }
    Ok(count)
}

/// Reads a line and converts it to a node type, node name, and embedding.
/// Blows up if it doesn't meet the formatting.
fn line_to_embedding(line: &String) -> Option<(String,String,Vec<f32>)> {
    let pieces:Vec<_> = line.split('\t').collect();
    if pieces.len() != 3 {
        return None
    }

    let node_type = pieces[0];
    let name = pieces[1];
    // Strip the enclosing brackets by character, so short or non-ASCII fields can't panic.
    let mut chars = pieces[2].chars();
    chars.next()?;
    chars.next_back()?;
    let emb: Result<Vec<f32>,_> = chars.as_str().split(',')
        .map(|wi| wi.trim().parse::<f32>()).collect();

    emb.ok().map(|e| (node_type.to_string(), name.to_string(), e))
}

/// Errors from loading a graph file.
#[derive(Debug)]
pub enum GraphLoadError {
    Io(std::io::Error),
    /// A line didn't have the five tab separated fields.  `record` is the 0-based record number,
    /// after any skipped rows.
    WrongFieldCount { record: usize },
    /// The weight field didn't parse as a float.
    BadWeight { record: usize, value: String, error: std::num::ParseFloatError }
}

impl std::fmt::Display for GraphLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphLoadError::Io(e) => write!(f, "{:?}", e),
            GraphLoadError::WrongFieldCount { record } =>
                write!(f, "{}: Malformed graph file: Expected 5 fields!", record),
            GraphLoadError::BadWeight { record, value, error } =>
                write!(f, "{}: Malformed graph file! {} - {:?}", record, error, value)
        }
    }
}

impl std::error::Error for GraphLoadError {}

/// Loads graphs from `node_type<TAB>name<TAB>node_type<TAB>name<TAB>weight` files.
pub struct GraphReader; 

impl GraphReader {
    
    /// Reads a graph file, reading `chunk_size` lines at a time and skipping the first
    /// `skip_rows`.  Weights are ignored (set to 1) unless `weighted`; `deduplicate` merges
    /// repeated edges.
    pub fn load(
        path: &str, 
        edge_type: EdgeType,
        chunk_size: usize,
        skip_rows: usize,
        weighted: bool,
        deduplicate: bool
    ) -> Result<(Vocab,CumCSR), GraphLoadError> {
        // Stop at the first read error and report it once the records read so far are drained.
        let mut read_error = None;
        let reader = open_file_for_reading(path)
            .map_err(GraphLoadError::Io)?
            .lines()
            .map_while(|l| l.map_err(|e| read_error = Some(e)).ok());

        let mut vocab = Vocab::new();
        let mut edges = Vec::new();
        let rr = RecordReader::new(chunk_size, skip_rows);
        rr.read(reader,
            |i, line| {
                let pieces: Vec<_> = line.split('\t').collect();
                if pieces.len() != 5 {
                    return Some(Err(GraphLoadError::WrongFieldCount { record: i }))
                }
                let from_node = (pieces[0].to_string(), pieces[1].to_string());
                let to_node = (pieces[2].to_string(), pieces[3].to_string());
                let w = if weighted {
                    match pieces[4].parse::<f32>() {
                        Err(error) => return Some(Err(GraphLoadError::BadWeight {
                            record: i, value: pieces[4].to_string(), error
                        })),
                        Ok(w) => w
                    }
                } else {
                    1f32
                };
                Some(Ok((from_node, to_node, w)))
            },
            |_i, record| {
                let (from_node, to_node, w) = record?;
                let f_id = vocab.get_or_insert(from_node.0, from_node.1);
                let t_id = vocab.get_or_insert(to_node.0, to_node.1);
                edges.push((f_id, t_id, w));
                if edge_type == EdgeType::Undirected {
                    edges.push((t_id, f_id, w));
                }
                Ok(())
            })?;

        if let Some(e) = read_error {
            return Err(GraphLoadError::Io(e))
        }

        let csr = CSR::construct_from_edges(edges, deduplicate);

        Ok((vocab, CumCSR::convert(csr)))
    }
}

#[cfg(test)]
mod io_tests {
    use super::*;

    /// Skipped header rows must not be counted as embeddings.
    #[test]
    fn test_load_store_skip_rows_sizes_store_to_vocab() {
        let path = std::env::temp_dir()
            .join(format!("cloverleaf-skip-rows-{}.emb", std::process::id()));
        std::fs::write(&path, "node_type\tname\tembedding\nn\ta\t[1,2]\nn\tb\t[3,4]\n").unwrap();

        let result = EmbeddingReader::load(
            path.to_str().unwrap(), Distance::Euclidean, &None, None, Some(1));
        std::fs::remove_file(&path).unwrap();

        let (vocab, es) = result.expect("load failed");
        assert_eq!(vocab.len(), 2);
        assert_eq!(es.len(), vocab.len());
        assert_eq!(es.get_embedding(1), &[3., 4.]);
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("cloverleaf-{}-{}", std::process::id(), name))
    }

    /// A written matrix reads back exactly: the same nodes in order and bit-identical values,
    /// plain or gzipped, across more rows than one formatting chunk.
    #[test]
    fn test_write_embedding_matrix_round_trips() {
        let rows = 20_000;
        let dims = 3;
        let keys: Vec<(String, String)> = (0..rows)
            .map(|i| (if i % 2 == 0 { "q" } else { "p" }.to_string(), format!("n{}", i)))
            .collect();
        let values: Vec<f32> = (0..rows * dims).map(|i| (i as f32).sin() * 1e3 / 7.0).collect();
        for name in ["matrix.emb", "matrix.emb.gz"] {
            let path = temp_path(name);
            write_embedding_matrix(path.to_str().unwrap(), &keys, &values, dims, None).unwrap();
            let (vocab, es) = EmbeddingReader::load(
                path.to_str().unwrap(), Distance::Euclidean, &None, None, None).unwrap();
            std::fs::remove_file(&path).unwrap();
            assert_eq!(vocab.len(), rows);
            for (i, (t, n)) in keys.iter().enumerate() {
                let id = vocab.get_node_id(t, n).expect("node missing");
                assert_eq!(es.get_embedding(id), &values[i * dims..(i + 1) * dims], "row {}", i);
            }
        }
    }

    #[test]
    fn test_write_embedding_matrix_rejects_a_shape_mismatch() {
        let keys = vec![("n".to_string(), "a".to_string())];
        let err = write_embedding_matrix(
            temp_path("bad.emb").to_str().unwrap(), &keys, &[1., 2., 3.], 2, None);
        assert!(matches!(err, Err(Error::InvalidArgument(_))));
    }

    /// Feature files may be gzipped, as `load_into` already supports.
    #[test]
    fn test_read_feature_vocab_gz() {
        let path = temp_path("features.gz");
        {
            let mut w = open_file_for_writing(path.to_str().unwrap(), Some(6)).unwrap();
            w.write_all(b"user\ta\tf1 f2\nuser\tb\tf3\n").unwrap();
        }
        let vocab = read_feature_vocab(path.to_str().unwrap());
        std::fs::remove_file(&path).unwrap();
        let vocab = vocab.expect("gzipped feature file should load");
        assert_eq!(vocab.len(), 2);
        assert_eq!(vocab.get_node_id("user", "b"), Some(1));
    }

    /// Malformed embedding fields are parse failures, not panics.
    #[test]
    fn test_line_to_embedding_rejects_malformed_fields() {
        for line in ["a\tb\t", "a\tb\t[", "a\tb\t[1,x]"] {
            let res = std::panic::catch_unwind(|| line_to_embedding(&line.to_string()));
            assert!(matches!(res, Ok(None)), "{:?} -> {:?}", line, res);
        }
        // Brackets are stripped by position, as before; a multi-byte one must not panic.
        let res = std::panic::catch_unwind(|| line_to_embedding(&"a\tb\t\u{e9}1,2\u{e9}".to_string()));
        assert!(res.is_ok());
        assert_eq!(line_to_embedding(&"a\tb\t[1,2]".to_string()),
                   Some(("a".to_string(), "b".to_string(), vec![1., 2.])));
    }

    fn load_graph(contents: &str) -> Result<(Vocab, CumCSR), GraphLoadError> {
        let path = temp_path(&format!("graph-{}.edges", contents.len()));
        std::fs::write(&path, contents).unwrap();
        let result = GraphReader::load(path.to_str().unwrap(), EdgeType::Directed, 1, 1, true, false);
        std::fs::remove_file(&path).unwrap();
        result
    }

    fn load_graph_err(contents: &str) -> GraphLoadError {
        load_graph(contents).err().expect("load should fail")
    }

    /// Malformed graph files report the record, counted after the skipped header.
    #[test]
    fn test_graph_load_errors() {
        let header = "from_type\tfrom\tto_type\tto\tweight\n";
        let ok = load_graph(&format!("{}a\t1\ta\t2\t2.5\n", header)).unwrap();
        assert_eq!((ok.0.len(), crate::graph::Graph::edges(&ok.1)), (2, 1));

        let e = load_graph_err(&format!("{}a\t1\ta\t2\t1\na\t1\ta\n", header));
        assert!(matches!(e, GraphLoadError::WrongFieldCount { record: 1 }), "{:?}", e);
        assert_eq!(e.to_string(), "1: Malformed graph file: Expected 5 fields!");

        let e = load_graph_err(&format!("{}a\t1\ta\t2\theavy\n", header));
        assert!(matches!(&e, GraphLoadError::BadWeight { record: 0, value, .. } if value == "heavy"), "{:?}", e);
        assert!(e.to_string().starts_with("0: Malformed graph file! "), "{}", e);

        let missing = GraphReader::load("/nonexistent/cloverleaf.edges", EdgeType::Directed, 1, 0, true, false);
        assert!(matches!(missing, Err(GraphLoadError::Io(_))));
    }

    #[test]
    fn test_read_node_values_errors() {
        let mut vocab = Vocab::new();
        vocab.get_or_insert("n", "a");
        let path = temp_path("values.txt");
        let read = |contents: &str| {
            std::fs::write(&path, contents).unwrap();
            read_node_values(path.to_str().unwrap(), &vocab)
        };
        assert_eq!(read("n\ta\t0.5\n").unwrap(), vec![(0, 0.5)]);
        assert!(matches!(read("n\ta\n"), Err(Error::Malformed { line: 1, .. })));
        assert!(matches!(read("n\ta\t1\nn\ta\tlots\n"), Err(Error::Malformed { line: 2, .. })));
        assert!(matches!(read("n\tb\t1\n"), Err(Error::UnknownNode { .. })));
        std::fs::remove_file(&path).unwrap();
    }
}
