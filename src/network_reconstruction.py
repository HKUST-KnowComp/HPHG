import random
import numpy as np
from tqdm import tqdm
import argparse
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Network reconstruction.")
    parser.add_argument('--load-emb', nargs='?',
                        help='Input embedding path.')
    parser.add_argument('--load-model', nargs='?',
                        help='Input model path.')
    return parser.parse_args()


def load_embeddings(filename):
    f = open(filename, 'r', encoding="utf8")
    embeddings = {}
    f.readline()
    while True:
        line = f.readline().strip().split()
        if line==[]:
            break
        name = int(line[0])
        del line[0]
    
        vect = []
        for m in line:
            vect.append(float(m))
        embeddings[name] = vect
    return embeddings

def main(args):
    dataset = None
    if args.load_emb:
        dataset = args.load_emb.split('/')[1]
    elif args.load_model:
        dataset = args.load_model.split('/')[1]

    edges = set()
    nums_type = None

    file = 'graph/' + dataset + '/full.edgelist'
    with open(file) as f:
        graph_type = f.readline().strip()
        if graph_type == '1': # heterogeneous
            nums_type = f.readline().split()
            nums_type = list(map(int, nums_type))

        for line in f.readlines():
            line = line.strip().split()
            edges.add(tuple(sorted(list(map(int,line[1:])))))

    cumsum = [0] + list(np.cumsum(nums_type))

    all_edges = []
    for x in range(cumsum[0],cumsum[1]):
        for y in range(cumsum[1],cumsum[2]):
            for z in range(cumsum[2],cumsum[3]):
                all_edges.append([x,y,z])

    ds = []
    inds = []
    if args.load_emb:
        embs = load_embeddings(args.load_emb)
        for edge in tqdm(all_edges,ascii=True):
            # d = hyperedge_dist(edge,embs,L1D)
            d = hyperedge_dist(edge,embs,L2D)
            # d = -hyperedge_dist(edge,embs,COS)
            ds.append(d)
        inds = np.argsort(ds)

    elif args.load_model:
        from keras.models import load_model
        model = load_model(args.load_model)
        
        ds = model.predict(np.array(all_edges))
        ds = [-ds[i][0] for i in range(len(ds))]
        inds = np.argsort(ds)

    scores = []

    for ratio in np.arange(0.1,1.1,0.1):
        cnt = 0
        for ind in inds[:int(ratio*len(edges))]:
            edge = tuple(sorted(all_edges[ind]))
            if edge in edges:
                cnt+=1

        print("ratio:", ratio)
        acc = cnt/int(ratio*len(edges))
        print("acc:", acc)
        scores.append(acc)
    print("random acc:", len(edges)/len(all_edges))
    print(scores)


if __name__ == "__main__":
    args = parse_args()
    main(args)

