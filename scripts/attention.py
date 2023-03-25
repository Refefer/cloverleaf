import tabulate
import sys
import cloverleaf
import numpy as np

def softmax(values):
    max_v = np.max(values, axis=1).reshape((-1, 1))
    numers = np.exp(values - max_v)
    return numers / np.sum(numers, axis=1).reshape((-1, 1))

def get_key_query_value(emb, ASIZE):
    return emb[:ASIZE], emb[ASIZE:ASIZE*2], emb[ASIZE*2:]

def get_attention(embs, feats, size):
    terms, query, key, value = [],[],[],[]
    for f in feats:
        if embs.contains(('feat', f)):
            q, k, v = get_key_query_value(embs.get_embedding(('feat', f)), size)
            terms.append(f)
            query.append(q)
            key.append(k)
            value.append(v)

    qs = np.vstack(query)
    keys = np.vstack(key)
    values = np.vstack(value)

    rows = [[] for _ in range(len(qs))]
    for i in range(len(qs)):
        for j in range(len(qs)):
            rows[i].append(qs[i].dot(keys[j]))

    attention = qs.dot(keys.T)
    sm = softmax(attention / np.sqrt(qs[0].shape[0]))
    return terms, sm, (values * sm.sum(axis=0).reshape((-1, 1))).mean(axis=0)

def cosine(e1, e2):
    return e1.dot(e2) / (e1.dot(e1) ** 0.5 * e2.dot(e2) ** 0.5)

def format_row(row):
    return [round(v, 3) for v in row]

def main():
    embs = cloverleaf.NodeEmbeddings.load(sys.argv[1], cloverleaf.Distance.Cosine)
    size = int(sys.argv[2])
    while True:
        terms = input("> ")
        if '/' in terms:
            before, after = terms.split('/')
            before = before.split()
            after = after.split()
            e1 = get_attention(embs, before, size)[2]
            print("e1:",e1)
            e2 = get_attention(embs, after, size)[2]
            print("e2:",e2)
            print(cosine(e1, e2))
        else:
            terms = terms.split()
            terms, mat, embedding = get_attention(embs, terms, size)
            headers = terms
            rows = [[term] + format_row(row) for term, row in zip(terms, mat)]
            rows.append(tabulate.SEPARATING_LINE)
            summed = mat.sum(axis=0)
            rows.append(['Softmax'] + format_row(summed / summed.sum()))
            print(tabulate.tabulate(rows, headers=headers))
            print(embedding)
        
if __name__ == '__main__':
    main()
