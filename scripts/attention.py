import argparse
import sys

import cloverleaf
import numpy as np
import tabulate

def softmax(values):
    max_v = np.max(values, axis=1).reshape((-1, 1))
    numers = np.exp(values - max_v)
    return numers / np.sum(numers, axis=1).reshape((-1, 1))

# query, key, query, key, value
def get_key_query_value(emb, head_num, num_heads, d_k):
    query_start = head_num * d_k
    key_start   = num_heads * d_k + head_num * d_k
    value_start = num_heads * d_k * 2;
    return emb[query_start:query_start+d_k], emb[key_start:key_start+d_k], emb[value_start:]

def get_attention(embs, feats, head_num, num_heads, d_k, context_window):
    terms, query, key, value = [],[],[],[]
    for f in feats:
        if embs.contains(('feat', f)):
            q, k, v = get_key_query_value(embs.get_embedding(('feat', f)), head_num, num_heads, d_k)
            terms.append(f)
            query.append(q)
            key.append(k)
            value.append(v)

    qs = np.vstack(query)
    keys = np.vstack(key)
    values = np.vstack(value)

    rows = [[] for _ in range(len(qs))]
    for i in range(len(qs)):
        if context_window is None:
            start, stop = 0, len(qs)
        else:
            start, stop = max(i - context_window, 0), min(i+1+context_window, len(qs))

        for j in range(len(qs)):
            if start <= j < stop:
                rows[i].append(qs[i].dot(keys[j]))
            else:
                rows[i].append(0)

    attention = np.array(rows)
    sm = softmax(attention / np.sqrt(qs[0].shape[0]))
    return terms, sm, (values * sm.sum(axis=0).reshape((-1, 1))).mean(axis=0)

def cosine(e1, e2):
    return e1.dot(e2) / (e1.dot(e1) ** 0.5 * e2.dot(e2) ** 0.5)

def format_row(row):
    return [round(v, 3) for v in row]

def parse_embedder(fname):
    with open(fname) as f:
        etype = f.readline().strip()
        if etype != 'Attention':
            raise TypeError("Embedder type is not Attention!")

        num_heads = int(f.readline().strip())
        d_k = int(f.readline().strip())
        window = int(f.readline().strip())
        if window == 0:
            window = None

        return num_heads, d_k, window

def main(args):
    embs = cloverleaf.NodeEmbeddings.load(args.features, cloverleaf.Distance.Cosine)
    num_heads, d_k, context_window = parse_embedder(args.embedder)
    print(f"Num Heads:{num_heads}, d_k: {d_k}, sliding: {context_window}")
    while True:
        terms = input("> ")
        terms = terms.split()
        sms = []
        for head_num in range(num_heads):
            headers = ['Head {}'.format(head_num)] + terms
            terms, mat, embedding = get_attention(embs, terms, head_num, num_heads, d_k, context_window)
            rows = [[term] + format_row(row) for term, row in zip(terms, mat)]
            rows.append(tabulate.SEPARATING_LINE)
            summed = mat.sum(axis=0)
            sm = summed / summed.sum()
            rows.append(['Softmax'] + format_row(sm))
            sms.append(sm)
            rows.append(tabulate.SEPARATING_LINE)
            idxs = np.argsort(sm)[::-1]
            rows.append(['Sorted'] + [terms[i] for i in idxs])
            print(tabulate.tabulate(rows, headers=headers, tablefmt="fancy_grid"))
            print()

        avg = np.sum(sms, axis=0) / len(sms)
        idxs = np.argsort(avg)[::-1]
        header = ['Terms'] + [terms[i] for i in idxs]
        sorted_scores = ['Average'] + format_row([avg[i] for i in idxs])
        print(tabulate.tabulate([sorted_scores], headers=header, tablefmt="fancy_grid"))
        print()

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Examine Attention Matrix',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("features",
        help="Path to feature embeddings.")
    parser.add_argument("embedder",
        help="Path to embedder spec.")

    return parser.parse_args()

if __name__ == '__main__':
    main(build_arg_parser())
