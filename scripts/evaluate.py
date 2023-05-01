import argparse
import random
import sys
import cloverleaf

def load(ne_fname, fe_fname, agg_fname):
    print("Loading Node Embeddings...")
    ne_embeddings = cloverleaf.NodeEmbeddings.load(ne_fname, cloverleaf.Distance.Cosine)
    print("Loading Feature Embeddings...")
    fe_embeddings = cloverleaf.NodeEmbeddings.load(fe_fname, cloverleaf.Distance.Cosine)
    print("Loading aggregator")
    aggregator = cloverleaf.FeatureAggregator.load(agg_fname)
    
    return ne_embeddings, fe_embeddings, aggregator

def reservoir_sample(it, k, seed=20232023):
    rs = random.Random(seed)
    buff = []
    for i, item in enumerate(it):
        if i < k:
            buff.append(item)
        else:
            idx = rs.randint(0, i)
            if idx < k:
                buff[idx] = item

    return iter(buff)

def main(args):
    dname = args.model
    ne_emb, fe_emb, agg = load(dname+'.node-embeddings', 
                               dname+'.feature-embeddings',
                               dname+'.embedder')

    recall = 0
    n = 0
    with open(args.test) as f:
        it = f
        if args.sample > 0:
            it = reservoir_sample(it, args.sample)

        for line in it:
            line = line.rstrip('\n')
            fnt, fn, tnt, tn, w = line.split('\t')

            if not ne_emb.contains((fnt, fn)):
                continue
            
            if not ne_emb.contains((tnt, tn)):
                continue

            emb = ne_emb.get_embedding((fnt, fn))
            top_k = ne_emb.nearest_neighbor(emb, args.k)
            key = (tnt, tn)
            recall += any(node == key for node, s in top_k)
            n += 1

            if n % 1000 == 0:
                print("{} - Recall@{}: {}".format(n, args.k, recall / n))

    print("Recall@{}: {}".format(args.k, recall / n))

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Evaluates dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("model",
        help="Path to model.")

    parser.add_argument("test",
        help="Path to test dataset")

    parser.add_argument("--k",
        type=int,
        default=100,
        help="Recall@K")

    parser.add_argument("--sample",
        type=int,
        default=0,
        help="If enabled, subsamples the dataset.")


    return parser

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    main(args)
