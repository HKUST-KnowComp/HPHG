import numpy as np
import random
from hypergraph import *
from tqdm import tqdm
import os


class Walker(object):
    def __init__(self, G, dataset, r):
        self.G = G      
        
        if os.path.exists('graph/'+dataset+'/indecom_factor.txt'):
            self.fs = []
            with open('graph/'+dataset+'/indecom_factor.txt') as f:
                for line in  f.readlines():
                    self.fs.append(float(line))
        else:
            print('calculating indecomposable factors...')
            fs = get_indecom_factor(G, r)
            self.fs = fs
            with open('graph/'+dataset+'/indecom_factor.txt','w') as f:
                for factor in fs:
                    print(factor,file=f)

        print("indecomposable factors:", self.fs)

    def hyperpath(self, walk_length, start_node):
        G = self.G
        walk = [start_node]

        while len(walk) < walk_length:
            pre = walk[-2] if len(walk)>1 else 'None'
            cur = walk[-1]

            cur_nbrs = list(G.neighbors(cur))

            if len(cur_nbrs) > 0:
                nxt = cur_nbrs[alias_draw(self.alias[(pre,cur)][0],
                        self.alias[(pre,cur)][1])]
                walk.append(nxt)
            else:
                break
        return walk

    def get_alias_node(self, cur):
        G = self.G
        nb = len(G.neighbors(cur))
        p = [1/nb]*nb
        
        return alias_setup(p)

    def get_alias_edge(self, pre, cur, alpha):
        G = self.G
        cur_nbrs = list(G.neighbors(cur))
        ps = {}

        ls = list(range(len(G.nums_type)))
        ls.remove(G.node_type(pre))
        ls.remove(G.node_type(cur))
        c = np.e**(self.fs[ls[0]]*alpha)

        for cur_nbr in cur_nbrs:
            ew = G.edge_weight(tuple(sorted([pre,cur,cur_nbr])))
            if ew != 0:
                ps[cur_nbr] = ew*c
            else:
                ps[cur_nbr] = 1

        p = list(ps.values())
        sum_p = sum(p)
        p = [pi/sum_p for pi in p]

        return alias_setup(p)

    def preprocess_transition_probs(self, alpha=100):
        '''
        Preprocess transition probability.
        '''
        G = self.G
        alias = {}
        n = len(G.nodes())

        for cur in G.nodes():
            alias[('None',cur)] = self.get_alias_node(cur)

        for pre in tqdm(G.nodes(),ascii=True):
            for cur in G.neighbors(pre):
                alias[(pre,cur)] = self.get_alias_edge(pre,cur,alpha)
        self.alias = alias

    def simulate_walks(self, num_walks, walk_length):
        '''
        Simulate random walks for each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)

            for node in tqdm(nodes,ascii=True):
                walks.append(self.hyperpath(walk_length=walk_length, start_node=node))

        return walks


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/.
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]