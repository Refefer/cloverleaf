//! The beginnings of a refactor of load/save methods currently defined within lib.rs
use crate::graph::NodeID;
use std::fs::File;
use std::io::{Write,BufWriter,Result as IOResult,BufReader,BufRead};
use std::convert::AsRef;

use fast_float::parse;
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use itertools::Itertools;
use pyo3::exceptions::{PyValueError,PyKeyError,PyIOError};
use pyo3::prelude::{PyResult,PyErr};
use rayon::prelude::*;
use ryu::Buffer;

use crate::vocab::Vocab;
use crate::embeddings::{EmbeddingStore,Distance};
use crate::graph::CumCSR;
use crate::{CSR,EdgeType};

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

impl EmbeddingReader {

    pub fn load(
        path: &str, 
        distance: Distance, 
        filter_type: Option<String>, 
        chunk_size: Option<usize>,
        skip_rows: Option<usize>
    ) -> PyResult<(Vocab, EmbeddingStore)> {
        let num_embeddings = count_lines(path, &filter_type)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        let reader = open_file_for_reading(path)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        let mut vocab = Vocab::new();
        
        // Place holder
        let mut es = EmbeddingStore::new(0, 0, Distance::Cosine);
        let rr = RecordReader::new(chunk_size.unwrap_or(1_000), skip_rows.unwrap_or(0));
        let mut i = 0;
        
        let filter_node = filter_type.as_ref();
        rr.read(reader.lines().map(|l| l.unwrap()),
            |_, line| {
               if let Some(node_type) = filter_node {
                   if !line.starts_with(node_type) {
                       return None
                   }
               }

               Some(line_to_embedding(&line)
                    .ok_or_else(|| PyValueError::new_err(format!("Error parsing line: {}", line))))
            },
            |_, record| {
                let (node_type, node_name, emb) = record?;

                if i == 0 {
                    es = EmbeddingStore::new(num_embeddings, emb.len(), distance);
                }

                let node_id = vocab.get_or_insert(node_type, node_name);
                if node_id < i {
                    return Err(PyKeyError::new_err(format!("found duplicate node at {}!", i)));
                }
                let m = es.get_embedding_mut(node_id);
                if m.len() != emb.len() {
                    return Err(PyValueError::new_err("Embeddings have different sizes!"));
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


/// Count the number of lines in an embeddings file so we only have to do one allocation.  If
/// NodeEmbeddings internal memory structure changes, such as using slabs, this might be less
/// relevant.
fn count_lines(path: &str, node_type: &Option<String>) -> IOResult<usize> {
    let reader = open_file_for_reading(path)?;
    let mut count = 0;
    let filter_node = node_type.as_ref();
    for line in reader.lines() {
        let line = line?;
        if let Some(p) = filter_node {
            if line.starts_with(p) {
                count += 1;
            }
        } else {
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
    let e = pieces[2];
    let emb: Result<Vec<f32>,_> = e[1..e.len() - 1].split(',')
        .map(|wi| parse(wi.trim())).collect();

    emb.ok().map(|e| (node_type.to_string(), name.to_string(), e))
}

pub struct GraphReader; 

impl GraphReader {
    
    fn deduplicate_edges(
        edges: &mut Vec<(NodeID, NodeID, f32)>
    ) -> () {
        // Sort edges and combine duplicates
        edges.par_sort_unstable_by_key(|e| (e.0, e.1));
        let mut i = 0;
        let mut j = 0;
        while j < edges.len() {
            let (from_node, to_node, _) = edges[j];
            let mut w = 0f32;
            while j < edges.len() && edges[j].0 == from_node && edges[j].1 == to_node {
                w += edges[j].2;
                j += 1
            }
            edges[i] = (from_node, to_node, w);
            i += 1;
        }
        edges.truncate(i);
    }

    pub fn load(
        path: &str, 
        edge_type: EdgeType,
        chunk_size: usize,
        skip_rows: usize,
        weighted: bool
    ) -> PyResult<(Vocab,CumCSR)> {
        let reader = open_file_for_reading(path)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?
            .lines().map(|l| l.unwrap());

        let mut vocab = Vocab::new();
        let mut edges = Vec::new();
        let rr = RecordReader::new(chunk_size, skip_rows);
        rr.read(reader,
            |i, line| {
                let pieces: Vec<_> = line.split('\t').collect();
                if pieces.len() != 5 {
                    return Some(Err(PyValueError::new_err(format!("{}: Malformed graph file: Expected 5 fields!", i))))
                }
                let from_node = (pieces[0].to_string(), pieces[1].to_string());
                let to_node = (pieces[2].to_string(), pieces[3].to_string());
                let w = if weighted {
                    let w = pieces[4].parse::<f32>();
                    match w {
                        Err(e) => return Some(Err(PyValueError::new_err(format!("{}: Malformed graph file! {} - {:?}", i, e, pieces[4])))),
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
                if matches!(edge_type, EdgeType::Undirected) {
                    edges.push((t_id, f_id, w));
                }
                Ok::<(), PyErr>(())
            })?;

        GraphReader::deduplicate_edges(&mut edges);

        let csr = CSR::construct_from_edges(edges);

        Ok((vocab, CumCSR::convert(csr)))
    }
}
