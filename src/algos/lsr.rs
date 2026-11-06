extern crate hashbrown;

use rayon::prelude::*;

use crate::graph::CDFGraph;
use crate::algos::pagerank::PageRank;
use crate::embeddings::EmbeddingStore;

static EPS: f64 = 1e-8;

pub struct LSR {
    pub passes: usize
}

impl LSR {
    
    pub fn compute(
        &self, 
        graph: &impl CDFGraph, 
        degrees: &EmbeddingStore,
        indicator: bool
    ) -> Vec<f64> {
        // Find the page rank of the tournament graph
        let page_rank = PageRank::new(self.passes, 1f64, EPS);
        let mut scores = page_rank.compute(graph, indicator);

        // Get the log norm of the scores.  A player who never wins has a stationary probability
        // of zero, so its log score is -inf; leave it out of the mean so it doesn't turn every
        // other score into NaN.
        scores.par_iter_mut().enumerate().for_each(|(node_id, s)| {
            *s = (*s / degrees.get_embedding(node_id)[0] as f64).ln();
        });
        let (total, count) = scores.iter()
            .filter(|s| s.is_finite())
            .fold((0f64, 0usize), |(t, c), s| (t + s, c + 1));
        let mean = if count > 0 { total / count as f64 } else { 0f64 };

        // Center the scores
        scores.par_iter_mut().for_each(|s| {
            *s -= mean;
        });

        scores
        
    }

}


#[cfg(test)]
mod lsr_tests {
    use super::*;
    use crate::graph::{CSR, CumCSR};
    use crate::distance::Distance;

    /// Player 2 loses to both 0 and 1 and never wins.  Its score can be -inf, but it must not
    /// turn everyone else's score into NaN.
    #[test]
    fn test_lsr_player_without_wins_does_not_poison_scores() {
        // Edges go loser -> winner.
        let edges = vec![(0, 1, 1.), (1, 0, 1.), (2, 0, 1.), (2, 1, 1.)];
        let graph = CumCSR::convert(CSR::construct_from_edges(edges, false));
        let degrees = EmbeddingStore::new_with_vec(3, 1, Distance::Euclidean, vec![2., 2., 2.])
            .unwrap();
        let scores = LSR { passes: 20 }.compute(&graph, &degrees, false);
        assert!(scores[0].is_finite() && scores[1].is_finite(), "{:?}", scores);
        assert!(scores[2] < scores[0] && scores[2] < scores[1], "{:?}", scores);
        assert!(!scores[2].is_nan(), "{:?}", scores);
    }
}
