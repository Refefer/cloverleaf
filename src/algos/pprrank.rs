//! 
use std::fmt::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use float_ord::FloatOrd;
use rayon::prelude::*;
use hashbrown::HashMap;
use std::collections::{HashMap as CHashMap};
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use simple_grad::*;

use crate::algos::rwr::{Steps,RWR};
use crate::sampler::Weighted;
use crate::graph::{Graph as CGraph, CDFGraph, NodeID};
use crate::embeddings::{EmbeddingStore,randomize_embedding_store};
use crate::distance::Distance;
use crate::progress::CLProgressBar;
use crate::feature_store::FeatureStore;
use crate::algos::ep::attention::softmax;
use crate::algos::ep::model::{construct_node_embedding,NodeCounts};
use crate::algos::ep::extract_grads;
use crate::algos::grad_utils::node_sampler::{RandomWalkHardStrategy,NodeSampler,BatchSamplerStrategy};
use crate::algos::grad_utils::optimizer::{Optimizer,AdamOptimizer};
use crate::algos::grad_utils::scheduler::LRScheduler;

#[derive(Copy,Clone)]
pub enum Loss {
    ListNet { passive:bool, weight_decay: f32 },
    ListMLE { weight_decay: f32 }
}

/// Defines PPR Rank
pub struct PprRank {
    /// Optimization to use for computing loss
    pub loss: Loss,

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

        // Enable/disable shared memory pool
        use_shared_pool(true);

        let total_updates = steps_per_pass * self.passes;
        let lr_scheduler = {
            let warm_up_steps = (total_updates as f32 / 5f32) as usize;
            let max_steps = self.passes * steps_per_pass;
            LRScheduler::cos_decay(self.alpha / 100f32, self.alpha, warm_up_steps, max_steps)
            //let min_alpha = self.alpha / 100f32; 
            //let gamma = (min_alpha.ln() / max_steps as f32).exp();
            //LRScheduler::exp_decay(min_alpha, self.alpha, gamma)
        };

        // Initialize samplers for negatives.
        let random_sampler = RandomWalkHardStrategy::new(0, &node_idxs);
        let valid_random_sampler = RandomWalkHardStrategy::new(0, &valid_idxs);

        let mut last_error = std::f32::INFINITY;
        let step = AtomicUsize::new(1);
        let mut valid_error = std::f32::INFINITY;

        // Generate the top K for each node once
        let walk_lib = self.generate_random_walks(graph, self.seed+1,);

        //println!("{} -> {:?}", 0, walk_lib.get(0));
        //println!("");
        let pb = CLProgressBar::new((self.passes * steps_per_pass) as u64, self.indicator);
        
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
                        graph, **node_id, &walk_lib, &features, &feature_embeddings, &sampler, &mut rng);

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
                            graph, **node_id, &walk_lib, &features, &feature_embeddings, 
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
            1f32,
            feature_store,
            feature_embeddings,
            self.num_features,
            rng)
    }

    fn generate_random_walks<G: CGraph + CDFGraph + Send + Sync>(
        &self, 
        graph: &G,
        seed: u64
    ) -> WalkLibrary {
        let mut walk_lib = WalkLibrary::new(graph.len(), self.k);
        let rwr = RWR {
            steps: self.steps.clone(),
            walks: self.num_walks,
            beta: self.beta,
            single_threaded: false,
            seed: seed + 13
        };

        let pb = CLProgressBar::new(graph.len() as u64, self.indicator);
        pb.update_message(|msg| write!(msg, "Computing random walks...").unwrap());
        let idxs = (0..graph.len()).collect::<Vec<_>>();
        idxs.chunks(1024).for_each(|node_ids| {
            let groups: Vec<_> = node_ids.par_iter().map(|node_id| {
                let mut nodes = Vec::with_capacity(self.k);
                let mut weights = Vec::with_capacity(self.k);
                
                let scores = rwr.sample(graph, &Weighted, *node_id).into_iter();
                let mut scores: Vec<_> = scores.collect();
                scores.sort_by_key(|(_k, v)| FloatOrd(-*v));
                scores.into_iter()
                    .filter(|(k,_v)| k != node_id)
                    .map(|(k, v)| (k, v.powf(self.compression)))
                    .take(self.k)
                    .for_each(|(k, v)| {
                        nodes.push(k);
                        weights.push(v);
                    });

                let sum = weights.iter().sum::<f32>();
                weights.iter_mut().for_each(|v| *v /= sum);
                pb.inc(1);
                (*node_id, nodes, weights)
            }).collect();

            groups.into_iter().for_each(|(node_id, nodes, weights)| {
                walk_lib.set(node_id, &nodes, &weights);
            });
        });
        pb.finish();
        walk_lib
    }

    fn run_forward_pass<G: CGraph + Send + Sync, R: Rng, S: NodeSampler>(
        &self, 
        graph: &G,
        node: NodeID,
        walk_lib: &WalkLibrary,
        features: &FeatureStore,
        feature_embeddings: &EmbeddingStore,
        sampler: &S,
        rng: &mut R
    ) -> (ANode, Vec<NodeCounts>) {
        let mut ranked_ids = Vec::with_capacity(self.k + self.negatives);
        let mut ranked_scores = Vec::with_capacity(self.k + self.negatives);
        let mut neg_ids = Vec::with_capacity(self.negatives);

        // Add the positives
        let (nodes, scores) = walk_lib.get(node);
        ranked_ids.extend_from_slice(nodes);
        ranked_scores.extend_from_slice(scores);

        // Sample random negatives
        sampler.sample_negatives(graph, node, &mut neg_ids, self.negatives, rng);
        neg_ids.into_iter().for_each(|neg_id| {
            ranked_ids.push(neg_id);
            ranked_scores.push(0.);
        });

        // Create the embeddings
        let mut ranked_embeddings = Vec::with_capacity(ranked_ids.len());
        let mut feat_maps = Vec::with_capacity(ranked_ids.len());
        ranked_ids.iter().for_each(|node_id| {
            let (fm, emb) = self.construct_avg_node(*node_id, features, feature_embeddings, rng);
            feat_maps.push(fm);
            ranked_embeddings.push(emb);
        });

        let (fm, query_node) = self.construct_avg_node(node, features, feature_embeddings, rng);
        feat_maps.push(fm);
        
        // Compute error
        let loss = match self.loss {
            Loss::ListNet { passive, weight_decay } => {
                let mut list_loss = self.listnet_loss(&query_node, &ranked_embeddings, &ranked_scores, node, passive);
                if weight_decay > 0f32 {
                    list_loss = list_loss + weight_decay * comp_weight_decay(&query_node, &ranked_embeddings, 0.1f32)
                }
                list_loss
            },
            Loss::ListMLE { weight_decay } => {
                let mut list_loss = self.list_mle_loss(&query_node, &ranked_embeddings, &ranked_scores, node);
                if weight_decay > 0f32 {
                    list_loss = list_loss + weight_decay * comp_weight_decay(&query_node, &ranked_embeddings, 0.1f32)
                }
                list_loss
            }
        };

        (loss, feat_maps)
    }

    fn listnet_loss(
        &self,
        query_node: &ANode,
        ranked_nodes: &[ANode],
        node_weights: &[f32],
        node_id: NodeID,
        passive: bool
    ) -> ANode {

        let scores = compute_distances(query_node, ranked_nodes, false);
        let sm_scores = softmax(scores, true);

        let ordered = node_weights.iter().enumerate()
            .filter(|(i, s)| {
                let nonzero = **s > 0f32;
                if passive {
                    nonzero && sm_scores.value()[*i] <= **s
                } else { nonzero }
            })
            .map(|(idx, s)| {
                let k = sm_scores.slice(idx, 1);
                k.ln() * *s
            }).collect::<Vec<ANode>>();

        if ordered.len() == 0 {
            //println!("Node:{}, {:?} -> {:?}", node_id, node_weights, sm_scores.value());
            Constant::scalar(0f32)
        } else {
            let loss = -ordered.sum_all();
            if node_id == 0 {
                println!("loss:{}, {:?} -> {:?}", loss.value()[0], node_weights, sm_scores.value());
            }
            loss
        }
    }

    fn list_mle_loss(
        &self,
        query_node: &ANode,
        ranked_nodes: &[ANode],
        node_weights: &[f32],
        node_id: NodeID
    ) -> ANode {

        let yi = compute_distances(query_node, ranked_nodes, false);

        // Compute the plackett luce model for each score
        let n = ranked_nodes.len();
        let pl: Vec<_> = (0..n).map(|i| {
            yi.slice(i,1) / yi.slice(i, n - i).sum()
        }).collect();

        let pl_loss = pl.iter().fold(Constant::scalar(1f32), |acc, x| acc * x);

        let loss = -pl_loss.ln();
        if loss.value()[0].is_nan() || loss.value()[0].is_infinite() {
            println!("yi: {:?}",yi.value());
            println!("pl: {:?}", pl.concat().value());
            Constant::scalar(0f32)
        } else {
            if node_id == 0 {
                println!("loss:{}, {:?} -> {:?}", loss.value()[0], node_weights, yi.value());
            }
            loss
        }

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

fn compute_distances(query_node: &ANode, ranked_nodes: &[ANode], cosine: bool) -> ANode {
    if cosine {
        // Compute the dot products to construct our yis
        let qn = il2norm(query_node);
        (ranked_nodes.iter().map(|n| {
            qn.dot(&il2norm(n))
        }).collect::<Vec<_>>().concat() * 5f32).exp()
    } else {
        ranked_nodes.iter().map(|n| {
            query_node.dot(n)
        }).collect::<Vec<_>>().concat().exp()
        //ranked_nodes.iter().map(|n| {
        //    1f32 / (1f32 + l2norm(&(query_node - n)))
        //}).collect::<Vec<_>>().concat()
        
    }
}

fn il2norm(v: &ANode) -> ANode {
    v / l2norm(v)
}

fn l2norm(v: &ANode) -> ANode {
    v.pow(2f32).sum().pow(0.5)
}

fn comp_weight_decay(query_node: &ANode, ranked_nodes: &[ANode], threshold: f32) -> ANode {
    let t = Constant::scalar(threshold);
    let mut mag = Vec::with_capacity(ranked_nodes.len() + 1);


    let qn = query_node.pow(2f32).sum() - &t;
    if qn.value()[0] > 0f32 {
        mag.push(qn);
    }

    ranked_nodes.iter().for_each(|n| {
        let nn = n.pow(2f32).sum() - &t;
        if nn.value()[0] > 0f32 {
            mag.push(nn)
        }
    });
    mag.concat().sum()
}

struct WalkLibrary {
    k: usize,
    nodes: Vec<NodeID>,
    weights: Vec<f32>,
    lens: Vec<usize>
}

impl WalkLibrary {
    fn new(num_nodes: usize, k: usize) -> Self {
        let nodes = vec![0; num_nodes * k];
        let weights = vec![0f32; num_nodes * k];
        let lens = vec![0; num_nodes];
        WalkLibrary { k, nodes, weights, lens }
    }

    fn set(&mut self, node_id: NodeID, nodes: &[NodeID], weights: &[f32]) {
        let offset = node_id * self.k;
        self.lens[node_id] = weights.len().min(self.k); 
        nodes.iter().zip(weights.iter()).take(self.k).enumerate()
            .for_each(|(i, (ni, wi))| {
                let idx = offset + i;
                self.nodes[idx] = *ni;
                self.weights[idx] = *wi;
            });
    }

    fn get(&self, node_id: NodeID) -> (&[NodeID], &[f32]) {
        let offset: usize = node_id * self.k;
        let len = self.lens[node_id];
        let ns = &self.nodes[offset..(offset+len)];
        let ws = &self.weights[offset..(offset+len)];
        (ns, ws)
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

    #[test]
    fn test_walk_library() {
        let mut walk_lib = WalkLibrary::new(10, 5);
        let (ns, ws) = walk_lib.get(3);
        assert_eq!(ns, vec![0;0]);
        assert_eq!(ws, vec![0.;0]);

        walk_lib.set(3, &[1, 2], &[1., 2.]);
        let (ns, ws) = walk_lib.get(3);
        assert_eq!(ns, &[1, 2]);
        assert_eq!(ws, &[1., 2.]);
    }

}
