//! The beginnings of a refactor of load/save methods currently defined within lib.rs
use crate::graph::NodeID;
use std::fs::File;
use std::io::{Write,BufWriter,Result as IOResult,BufReader,BufRead};
use std::convert::AsRef;

use rayon::prelude::*;
use flate2::read::GzDecoder;
use ryu::Buffer;
use fast_float::parse;
use pyo3::prelude::PyResult;
use pyo3::exceptions::{PyValueError,PyKeyError,PyIOError};
use itertools::Itertools;

use crate::vocab::Vocab;
use crate::embeddings::{EmbeddingStore,Distance};

/// Streaming writer for NodeEmbeddings.  Since Embeddings are often gigantic, creating them adhoc
/// then streaming them to disk is beneficial.
pub struct EmbeddingWriter<'a> {
    vocab: &'a Vocab,
    output: BufWriter<File>,
    buffer: String
}

impl <'a> EmbeddingWriter<'a> {

    pub fn new(path: &str, vocab: &'a Vocab) -> IOResult<Self> {
        let f = File::create(path)?;
        let bw = BufWriter::new(f);

        let s = String::new();
        Ok(EmbeddingWriter { 
            vocab,
            output: bw,
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

pub struct EmbeddingReader; 

impl EmbeddingReader {

    pub fn load(
        path: &str, 
        distance: Distance, 
        filter_type: Option<String>, 
        chunk_size: Option<usize>
    ) -> PyResult<(Vocab, EmbeddingStore)> {
        let num_embeddings = count_lines(path, &filter_type)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        let reader = open_file_for_reading(path)
            .map_err(|e| PyIOError::new_err(format!("{:?}", e)))?;

        let mut vocab = Vocab::new();
        
        // Place holder
        let mut es = EmbeddingStore::new(0, 0, Distance::Cosine);
        let filter_node = filter_type.as_ref();
        let mut i = 0;
        let mut buffer = Vec::with_capacity(chunk_size.unwrap_or(1_000));
        let mut p_buffer = Vec::with_capacity(buffer.capacity());
        for chunk in &reader.lines().map(|l| l.unwrap()).chunks(buffer.capacity()) {
            buffer.clear();
            p_buffer.clear();
            
            // Read lines into a buffer for parallelizing
            chunk.filter(|line| {
                // If it doesn't match the pattern, move along
                if let Some(node_type) = filter_node {
                    line.starts_with(node_type)
                } else {
                    true
                }
            }).for_each(|l| buffer.push(l));

            // Parse lines
            buffer.par_drain(..).map(|line| {
                line_to_embedding(&line)
                    .ok_or_else(|| PyValueError::new_err(format!("Error parsing line: {}", line)))
            }).collect_into_vec(&mut p_buffer);

            for record in p_buffer.drain(..) {
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
            }
        }

        Ok((vocab, es))
    }
}

fn open_file_for_reading(path: &str) -> IOResult<Box<dyn BufRead>> {
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

