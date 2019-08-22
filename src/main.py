'''
Reference implementation of HPHG and HPSG.

Author: Jie Huang

For more details, refer to the paper:
Hyper-Path-Based Representation Learning for Hyper-Networks
Jie Huang, Xin Liu, Yangqiu Song
'''

import numpy as np
import argparse
import networkx as nx
from gensim.models import Word2Vec
from hyperpath import *
from hypergraph import *
from hypergram import *
from time import time

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# tensorflow config
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def parse_args():
    parser = argparse.ArgumentParser(description="Run algorithm.")

    parser.add_argument('--input', nargs='?',
                        help='Input graph path.')

    parser.add_argument('--output', nargs='?',
                        help='Output representation path.')

    parser.add_argument('--load-model', nargs='?',
                        help='Input model path.')

    parser.add_argument('--save-model', nargs='?',
                        help='Output model path.')

    parser.add_argument('--dimensions', type=int, default=32,
                        help='Number of dimensions. Default is 32.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=6,
                        help='Context size for optimization. Default is 6.')

    parser.add_argument('--iter', default=5, type=int,
                        help='Number of epochs in SGD. Default is 5.')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--alpha', type=int, default=100,
                        help='Indecomposable hyperparameter. Default is 100.')

    parser.add_argument('--lambda0', type=float, default=1.,
                        help='Parameter to balance pairwise and tuplewise loss. Default is 1.')

    parser.add_argument('--pair-ratio', type=float, default=0.,
                        help='Pair ratio of negative hyperedge sampling. Default is 0.')

    parser.add_argument('--r', type=int, default=30,
                        help='The multiple of the number of random samples relative to the number of edges \
                         when calculating the indecomposable factor. Default is 30.')

    parser.add_argument('--method', choices=['hphg', 'hpsg'], default='hphg',
                        help='The learning method. Default is HPHG.')

    return parser.parse_args()


def read_graph():
    '''
    Reads the input hyper-network in networkx.
    '''
    f = open(args.input, 'r', encoding='utf8')

    graph_type = f.readline().strip()
    nums_type = None
    if graph_type == '1': # heterogeneous
        nums_type = f.readline().split()
        nums_type = list(map(int, nums_type))

    G = Hypergraph(graph_type,nums_type)
    print(graph_type,nums_type)

    for line in f.readlines():
        line = line.split()
        edge_name = line[0]
        G.add_edge(edge_name, map(int,line[1:]))

    f.close()
    return G

def load_embeddings(filename):
    '''
    Load embeddings from file.
    '''
    f = open(filename, 'r', encoding="utf8")

    embeddings = {}
    f.readline()

    while True:
        line = f.readline().strip().split()
        if line==[]:
            break
        name = int(line[0]) # id
        vect = []
        for m in line[1:]:
            vect.append(float(m))
        embeddings[name] = vect

    f.close()
    return embeddings

def learn_embeddings(walks, G, dataset):
    '''
    Learn embeddings by HPHG or HPSG model.
    '''
    fout = open(args.output, 'w', encoding='utf8')

    if args.method=='hphg':
        hg = hypergram(G,dataset,walks,size=args.dimensions,window=args.window_size,
                                pair_ratio=args.pair_ratio,epochs=args.iter,lambda0=args.lambda0)

        fout.write("{} {}\n".format(len(G.nodes()), args.dimensions))
        for i in range(0,hg.wv.shape[0]):
            fout.write("{} {}\n".format(str(i),
                                        ' '.join([str(x) for x in hg.wv[i]])))
        hg.model_t.save(args.save_model)

    elif args.method=='hpsg':
        walks = [list(map(str,walk)) for walk in walks]
        word2vec = Word2Vec(walks,size=args.dimensions,window=args.window_size,min_count=0,
                                sg=1,workers=args.workers,iter=args.iter,negative=5,compute_loss=True)
        vectors = {}

        for word in map(str,list(G.nodes())):
            vectors[word] = word2vec[word]

        node_num = len(vectors.keys())
        fout.write("{} {}\n".format(node_num, args.dimensions))
        for node, vec in vectors.items():
            fout.write("{} {}\n".format(str(node),' '.join([str(x) for x in vec])))
    fout.close()


def main(args):
    t0 = time()
    dataset = args.input.split('/')[1]  # extract dataset name
    print("Dataset:", dataset)

    if args.load_model:
        from keras.models import load_model
        model = load_model(args.load_model)
        print(link_prediction(model,dataset))
        return 0

    if not args.save_model:
        args.save_model = "model/"+dataset+"/model.h5"

    print('\n##### reading hypergraph...')
    G = read_graph()
    print('time:',time()-t0)

    print('\n##### initializing hypergraph...')
    walker = Walker(G, dataset, args.r)
    print('time:',time()-t0)

    print('\n##### preprocessing transition probs...')
    walker.preprocess_transition_probs(alpha=args.alpha)
    print('time:',time()-t0)

    print('\n##### walking...')
    walks = walker.simulate_walks(args.num_walks, args.walk_length)
    print('time:',time()-t0)
    del walker  # release memory

    print("\n##### embedding...")
    learn_embeddings(walks, G, dataset)
    print('time:',time()-t0)


if __name__ == "__main__":
    args = parse_args()
    main(args)



