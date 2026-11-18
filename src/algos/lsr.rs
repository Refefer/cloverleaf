extern crate hashbrown;

use rayon::prelude::*;

use crate::graph::CDFGraph;
use crate::algos::pagerank::PageRank;
use crate::embeddings::{EmbeddingStore, NodeEmbeddings};
use crate::distance::Distance;
use crate::graph::EdgeType;
use crate::named_graph::{GraphBuilder, NamedGraph};

static EPS: f64 = 1e-8;

/// Builds a [`Tournament`] from game outcomes.
pub struct TournamentBuilder {
    gb: GraphBuilder,
    degrees: Vec<f32>
}

impl TournamentBuilder {
    pub fn new() -> Self {
        TournamentBuilder { gb: GraphBuilder::new(), degrees: Vec::new() }
    }

    /// Records that `winner` beat `loser`; the edge points from loser to winner.
    pub fn add_outcome(&mut self, winner: (&str, &str), loser: (&str, &str), weight: f32) {
        let d = weight / 2f32;
        self.gb.add_edge(loser, winner, d, EdgeType::Directed);
        // New players start with a degree of 1.
        self.degrees.resize(self.gb.nodes(), 1f32);
        let loser_id = self.gb.vocab().get_node_id(loser.0, loser.1).expect("just added");
        self.degrees[loser_id] += d;
    }

    /// Records a ranking, best first: every player beats everyone ranked below them.
    pub fn add_ranked_outcomes<A: AsRef<str>, B: AsRef<str>>(&mut self, order: &[(A, B)], weight: f32) {
        for i in 0..order.len() {
            for j in (i + 1)..order.len() {
                let winner = (order[i].0.as_ref(), order[i].1.as_ref());
                let loser = (order[j].0.as_ref(), order[j].1.as_ref());
                self.add_outcome(winner, loser, weight);
            }
        }
    }

    pub fn nodes(&self) -> usize {
        self.gb.nodes()
    }

    pub fn edges(&self) -> usize {
        self.gb.edges()
    }

    /// Takes the outcomes recorded so far, leaving the builder empty.  None if there are none.
    pub fn build(&mut self) -> Option<Tournament> {
        let graph = self.gb.build(true)?;
        let degrees = std::mem::take(&mut self.degrees);
        let es = EmbeddingStore::new_with_vec(graph.nodes(), 1, Distance::Euclidean, degrees)
            .expect("one degree per player");
        let norms = NodeEmbeddings { vocab: graph.vocab().clone(), embeddings: es };
        Some(Tournament { graph, norms })
    }
}

impl Default for TournamentBuilder {
    fn default() -> Self {
        TournamentBuilder::new()
    }
}

/// Game outcomes between players, as a graph from losers to winners, ready for [`LSR`].
pub struct Tournament {
    graph: NamedGraph,
    norms: NodeEmbeddings
}

impl Tournament {
    pub fn graph(&self) -> &NamedGraph {
        &self.graph
    }
}

pub struct LSR {
    pub passes: usize
}

impl LSR {
    /// Each player's score, as one-dimensional embeddings.
    pub fn learn(&self, tournament: &Tournament, indicator: bool) -> NodeEmbeddings {
        let scores = self.compute(tournament.graph.graph(), &tournament.norms.embeddings, indicator);
        NodeEmbeddings::from_scores(tournament.graph.vocab().clone(), &scores).expect("one score per player")
    }
    
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

    use crate::graph::Graph;

    /// [A, B, C] means A > B, A > C, B > C.  Edges point loser -> winner.
    #[test]
    fn test_add_ranked_outcomes_pairs() {
        let mut tb = TournamentBuilder::new();
        tb.add_ranked_outcomes(&[("team", "A"), ("team", "B"), ("team", "C")], 1.0);
        let t = tb.build().expect("outcomes were added");
        let g = t.graph();
        let id = |n: &str| g.node_id("team", n).unwrap();
        let (a, b, c) = (id("A"), id("B"), id("C"));

        let mut edges: Vec<_> = (0..g.nodes())
            .flat_map(|f| g.graph().get_edges(f).0.iter().map(move |t| (f, *t)))
            .collect();
        edges.sort();
        let mut expected = vec![(b, a), (c, a), (c, b)];
        expected.sort();
        assert_eq!(edges, expected);
    }

    /// Players start at a degree of 1 and losers gain half the game's weight.
    #[test]
    fn test_tournament_degrees() {
        let mut tb = TournamentBuilder::new();
        tb.add_outcome(("p", "w"), ("p", "l"), 2.0);
        let t = tb.build().unwrap();
        let w = t.graph().node_id("p", "w").unwrap();
        let l = t.graph().node_id("p", "l").unwrap();
        assert_eq!(t.norms.embeddings.get_embedding(w), &[1.0]);
        assert_eq!(t.norms.embeddings.get_embedding(l), &[2.0]);
    }
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
