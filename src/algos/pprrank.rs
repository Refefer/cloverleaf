//! 
use std::fmt::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use float_ord::FloatOrd;
use rayon::prelude::*;
use hashbrown::HashMap;
use std::collections::{HashMap as CHashMap};
use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_xorshift::XorShiftRng;
use simple_grad::*;

use crate::algos::rwr::{Steps,RWR};
use crate::sampler::Weighted;
use crate::graph::{Graph as CGraph, CDFGraph, NodeID};
use crate::embeddings::{EmbeddingStore,Distance,randomize_embedding_store};
use crate::progress::CLProgressBar;
use crate::feature_store::FeatureStore;
use crate::algos::ep::attention::softmax;
use crate::algos::ep::model::{construct_node_embedding,NodeCounts};
//use crate::algos::grad_utils::optimizer::
use crate::algos::grad_utils::node_sampler::{RandomWalkHardStrategy,NodeSampler,BatchSamplerStrategy};
use crate::algos::grad_utils::optimizer::{Optimizer,AdamOptimizer};
use crate::algos::grad_utils::scheduler::LRScheduler;

/// Defines PPR Rank
pub struct PprRank {
    /// Learning rate for updating feature embeddings
    pub alpha: f32,

    /// Batch size to use.  Larger batches have fewer updates, but also lower variance
    pub batch_size: usize,

    /// Size of the node embeddings.  Feature embeddings can be larger if they use attention
    pub dims: usize,

    /// Number of passes to optimize for
    pub passes: usize,

    /// Whether we use hard negatives or not.  We might strip this out since I've had difficulty
    /// using it to improve test loss
    pub negatives: usize,
    
    /// Whether we use hard negatives or not.  We might strip this out since I've had difficulty
    /// using it to improve test loss
    pub num_walks: usize,

    /// Whether we use hard negatives or not.  We might strip this out since I've had difficulty
    /// using it to improve test loss
    pub steps: Steps,

    /// Beta to use for the RWR algorithm
    pub beta: f32,

    /// Number of items to select from the RWR algorithm for optimization
    pub k: usize,

    /// Randomly selects K features
    pub num_features: Option<usize>,

    /// Compresses or flattens distribution
    pub compression: f32,

    /// Random seed
    pub seed: u64,

    /// We split out valid_pct of nodes to use for validation.
    pub valid_pct: f32,

    /// Whether to show a pretty indicator
    pub indicator: bool
}

impl PprRank {

    /// Learns the feature embeddings.
    pub fn learn<G: CGraph + CDFGraph + Send + Sync>(
        &self, 
        graph: &G, 
        features: &FeatureStore,
        feature_embeddings: Option<EmbeddingStore>
    ) -> EmbeddingStore {
        let feat_embeds = self.learn_feature_embeddings(graph, features, feature_embeddings);
        feat_embeds
    }
    
    fn learn_feature_embeddings<G: CGraph + CDFGraph + Send + Sync>(
        &self,
        graph: &G,
        features: &FeatureStore,
        feature_embeddings: Option<EmbeddingStore>,
    ) -> EmbeddingStore {

        // 
        // Initialization
        //
        let mut rng = XorShiftRng::seed_from_u64(self.seed);

        let feature_embeddings = if let Some(embs) = feature_embeddings {
            embs
        } else {
            let mut fe = EmbeddingStore::new(features.num_features(), self.dims, Distance::Cosine);
            // Initialize embeddings as random
            randomize_embedding_store(&mut fe, &mut rng);
            fe
        };

        // Initializer SGD optimizer.  Right now we hard code the parameters for the optimizer but
        // in the future we could allow for this to be parameterized.
        let optimizer = AdamOptimizer::new(0.9, 0.999,
            feature_embeddings.dims(), 
            feature_embeddings.len()); 

        // Pull out validation idxs;
        let mut node_idxs: Vec<_> = (0..graph.len()).into_iter().collect();
        node_idxs.shuffle(&mut rng);
        let valid_idx = (graph.len() as f32 * self.valid_pct) as usize;
        let valid_idxs = node_idxs.split_off(graph.len() - valid_idx);

        // Number of update stpes
        let steps_per_pass = (node_idxs.len() as f32 / self.batch_size as f32) as usize;

        let pb = CLProgressBar::new((self.passes * steps_per_pass) as u64, self.indicator);
        
        // Enable/disable shared memory pool
        use_shared_pool(true);

        let total_updates = steps_per_pass * self.passes;
        let lr_scheduler = {
            let warm_up_steps = (total_updates as f32 / 5f32) as usize;
            let max_steps = self.passes * steps_per_pass;
            LRScheduler::cos_decay(self.alpha / 100f32, self.alpha, warm_up_steps, max_steps)
        };

        // Initialize samplers for negatives.
        let random_sampler = RandomWalkHardStrategy::new(0, &node_idxs);
        let valid_random_sampler = RandomWalkHardStrategy::new(0, &valid_idxs);

        let mut last_error = std::f32::INFINITY;
        let step = AtomicUsize::new(1);
        let mut valid_error = std::f32::INFINITY;

        // Generate the top K for each node once
        let walk_list = self.generate_random_walks(graph, self.seed+1,);

        println!("{} -> {:?}", 0, walk_list[0]);
        println!("");
        
        for pass in 1..(self.passes + 1) {

            pb.update_message(|msg| {
                msg.clear();
                let cur_step = step.load(Ordering::Relaxed);
                let alpha = lr_scheduler.compute(cur_step);
                write!(msg, "Pass {}/{}, Train: {:.5}, Valid: {:.5}, LR: {:.5}", pass, self.passes, 
                       last_error, valid_error, alpha)
                    .expect("Error writing out indicator message!");
            });

            if pass % 10 == 0 && self.indicator {
                println!();
            }

            // Shuffle for SGD
            node_idxs.shuffle(&mut rng);
            let err: Vec<_> = node_idxs.par_iter().chunks(self.batch_size).enumerate().map(|(i, nodes)| {

                let mut grads = Vec::with_capacity(self.batch_size);
                
                let sampler = (&random_sampler).initialize_batch(
                    &nodes,
                    graph,
                    features);
                
                // Compute grads for batch
                nodes.par_iter().map(|node_id| {
                    let mut rng = XorShiftRng::seed_from_u64(self.seed + (i + **node_id) as u64);
                    let (loss, feat_maps) = self.run_forward_pass(
                        graph, **node_id, &walk_list, &features, &feature_embeddings, &sampler, &mut rng);

                    let grads = self.extract_gradients(&loss, feat_maps);
                    (loss.value()[0], grads)
                }).collect_into_vec(&mut grads);

                let mut error = 0f32;
                let mut cnt = 0f32;
                
                // We are using std Hashmap instead of hashbrown due to a weird bug
                // where the optimizer, for whatever reason, has trouble draining it
                // on 0.13.  We'll keep testing it on subsequent fixes but until then
                // std is the way to go.
                let mut all_grads = CHashMap::new();

                // Since we're dealing with multiple reconstructions with likely shared features,
                // we aggregate all the gradients
                for (err, grad_set) in grads.drain(..nodes.len()) {
                    for (feat, grad) in grad_set.into_iter() {
                        let e = all_grads.entry(feat).or_insert_with(|| vec![0.; grad.len()]);
                        e.iter_mut().zip(grad.iter()).for_each(|(ei, gi)| *ei += *gi);
                    }
                    error += err;
                    cnt += 1f32;
                }

                let cur_step = step.fetch_add(1, Ordering::Relaxed);

                // Backpropagate embeddings
                let alpha = lr_scheduler.compute(cur_step);
                optimizer.update(&feature_embeddings, all_grads, alpha, pass as f32);

                // Update progress bar
                pb.inc(1);
                if cnt > 0f32 { error / cnt } else { 0f32 }
            }).collect();

            // Some losses go toward infinity.  This is a bug we should fix.
            last_error = err.iter()
                .filter(|x| !x.is_infinite() )
                .sum::<f32>() / err.len() as f32;
            
            if valid_idxs.len() > 0 {
                // Validate.  We use the same random seed for consistency across iterations.
                let valid_errors = valid_idxs.par_iter().chunks(self.batch_size).map(|nodes| {
                    let sampler = (&valid_random_sampler).initialize_batch(&nodes, graph, features);

                    nodes.par_iter().map(|node_id| {
                        let mut rng = XorShiftRng::seed_from_u64(self.seed + **node_id as u64);
                        let loss = self.run_forward_pass(
                            graph, **node_id, &walk_list, &features, &feature_embeddings, 
                            &sampler, &mut rng).0;

                        loss.value()[0]
                    })
                    .filter(|l| !l.is_infinite())
                    .sum::<f32>()
                }).sum::<f32>();
                
                valid_error = valid_errors / valid_idxs.len() as f32;
            }
        }
        pb.finish();
        feature_embeddings
    }

    fn construct_avg_node(
        &self,
        node_id: NodeID,
        feature_store: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        rng: &mut impl Rng
    ) -> (NodeCounts, ANode) {
        construct_node_embedding(
            node_id, 
            feature_store,
            feature_embeddings,
            self.num_features,
            rng)
    }

    fn generate_random_walks<G: CGraph + CDFGraph + Send + Sync>(
        &self, 
        graph: &G,
        seed: u64
    ) -> Vec<Vec<(NodeID,f32)>> {
        (0..graph.len()).into_par_iter().map(|node_id| {
            let rwr = RWR {
                steps: self.steps.clone(),
                walks: self.num_walks,
                beta: self.beta,
                seed: seed + node_id as u64
            };

            let scores = rwr.sample(graph, &Weighted, node_id).into_iter();
            let mut scores: Vec<_> = scores.collect();
            scores.sort_by_key(|(_k, v)| FloatOrd(-*v));
            let mut s = scores.into_iter()
                .filter(|(k,_v)| *k != node_id)
                .map(|(k, v)| (k, v.powf(self.compression)))
                .take(self.k)
                .collect::<Vec<_>>();

            let sum = s.iter().map(|(_k, v)| v).sum::<f32>();
            s.iter_mut().for_each(|(_k, v)| *v /= sum);
            s
        }).collect()
    }

    fn run_forward_pass<G: CGraph + Send + Sync, R: Rng, S: NodeSampler>(
        &self, 
        graph: &G,
        node: NodeID,
        walk_map: &[Vec<(NodeID, f32)>],
        features: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        sampler: &S,
        rng: &mut R
    ) -> (ANode, Vec<NodeCounts>) {
        let mut ranked_ids = Vec::with_capacity(self.k + self.negatives);

        // Add the positives
        ranked_ids.extend_from_slice(&walk_map[node]);

        // Add the negatives
        let mut negatives = Vec::with_capacity(self.negatives);
        
        // Sample random negatives
        sampler.sample_negatives(graph, node, &mut negatives, self.negatives, rng);
        negatives.into_iter().for_each(|n| ranked_ids.push((n, 0f32)));

        // Create the embeddings
        let mut ranked_embeddings = Vec::with_capacity(ranked_ids.len());
        let mut feat_maps = Vec::with_capacity(ranked_ids.len());
        ranked_ids.iter().for_each(|(node_id, _)| {
            let (fm, emb) = self.construct_avg_node(*node_id, features, feature_embeddings, rng);
            feat_maps.push(fm);
            ranked_embeddings.push(emb);
        });

        let (fm, query_node) = self.construct_avg_node(node, features, feature_embeddings, rng);
        feat_maps.push(fm);
        
        // Compute error
        let loss = self.loss(&query_node, &ranked_embeddings, &ranked_ids);

        (loss, feat_maps)
    }

    fn loss(
        &self,
        query_node: &ANode,
        ranked_nodes: &[ANode],
        node_weights: &[(NodeID, f32)]
    ) -> ANode {

        // Compute dot score, then the softmax
        let scores = ranked_nodes.iter().map(|n| {
            query_node.dot(n)
        }).collect::<Vec<_>>().concat();

        let sm_scores = softmax(scores);
        //println!("{:?} -> {:?}", node_weights, sm_scores.value());
        let ordered = (0..ranked_nodes.len())
            .filter(|idx| node_weights[*idx].1 > 0f32)
            .map(|idx| {
                let s = node_weights[idx].1;
                let k = sm_scores.slice(idx, 1);
                k.ln() * s
            }).collect::<Vec<ANode>>();

        -ordered.sum_all()
    }

    fn extract_gradients(
        &self, 
        loss: &ANode,
        feat_maps: Vec<NodeCounts>
    ) -> HashMap<usize, Vec<f32>> {

        // Compute gradients
        let mut agraph = Graph::new();

        agraph.backward(&loss);

        let mut grads = HashMap::new();
        feat_maps.into_iter().for_each(|fm| {
            extract_grads(&agraph, &mut grads, fm.into_iter());
        });

        grads

    }

}

fn il2norm(v: &ANode) -> ANode {
    let norm = v.pow(2f32).sum().pow(0.5);
    v / norm
}

/// We extract the gradients for each unique feature
fn extract_grads(
    graph: &Graph, 
    grads: &mut HashMap<usize, Vec<f32>>, 
    vars: impl Iterator<Item=(usize, (ANode, usize))>
) {
    for (feat_id, (var, _)) in vars {
        if grads.contains_key(&feat_id) { continue }

        if let Some(grad) = graph.get_grad(&var) {
            if grad.iter().all(|gi| !(gi.is_nan() || gi.is_infinite())) {
                // Can get some nans in weird cases, such as the distance between
                // a node and it's reconstruction when it shares all features.
                // Since that's not all that helpful anyways, we simply ignore it and move on
                grads.insert(feat_id, grad.to_vec());
            }
        }
    }
}

#[cfg(test)]
mod ep_tests {
    use super::*;
    use crate::graph::{CumCSR,CSR};

    fn build_star_edges() -> Vec<(usize, usize, f32)> {
        let mut edges = Vec::new();
        let max = 100;
        for ni in 0..max {
            for no in (ni+1)..max {
                edges.push((ni, no, 1f32));
                edges.push((no, ni, 1f32));
            }
        }
        edges
    }

    #[test]
    fn test_simple_learn_dist() {
        let edges = build_star_edges();
        let csr = CSR::construct_from_edges(edges);
        let ccsr = CumCSR::convert(csr);
        
        let mut feature_store = FeatureStore::new(ccsr.len(), "feat".to_string());
        feature_store.fill_missing_nodes();

    }

}