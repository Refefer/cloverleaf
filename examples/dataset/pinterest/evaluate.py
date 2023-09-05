import numpy as np
import sys
import cloverleaf

embs = cloverleaf.NodeEmbeddings.load(sys.argv[1], cloverleaf.Distance.Cosine)
embs.l2norm()

def dist(user, board):
    ue = embs.get_embedding(('user', str(user)))
    be = embs.get_embedding(('board', str(board)))
    
    return np.dot(ue, be)

hr5 = 0
hr10 = 0
total = 0
with open(sys.argv[2]) as f:
    for i, line in enumerate(f):
        user, pos, negs = line.strip().split('\t')
        pos_score = dist(user, pos)
        pos = 0
        for neg in negs.split():
            if dist(user, neg) > pos_score:
                pos += 1

        if pos < 5:
            hr5 += 1

        if pos < 10:
            hr10 += 1

        total += 1

        if i % 5000 == 0 and i > 0:
            print(i, 'HR@5', hr5, total, hr5/total)
            print(i, 'HR@10', hr10, total, hr10/total)

print('HR@5', hr5, total, hr5/total)
print('HR@10', hr10, total, hr10/total)

