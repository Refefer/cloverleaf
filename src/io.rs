use crate::graph::NodeID;
use std::fs::File;
use std::io::{Write,BufWriter,Result as IOResult};
use std::convert::AsRef;

use ryu::Buffer;

use crate::vocab::Vocab;

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

