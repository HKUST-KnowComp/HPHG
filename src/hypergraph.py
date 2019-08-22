import numpy as np
from tqdm import tqdm


class Hypergraph(object):
    def __init__(self,graph_type='0',nums_type=None):
        self._nodes = {}  # node set
        self._edges = {}  # edge set (hash index)
        self.graph_type = graph_type  # graph type, homogeneous:0, heterogeneous:1
        self.nums_type = nums_type  # for heterogeneous graph, number of different node type
        self.cumsum = np.cumsum(self.nums_type) if self.graph_type=='1' else None  # cumsum of nums_type

    def add_edge(self, edge_name, e):
        '''
        Add a hyperedge.
        edge_name: name of hyperedge
        edge: node list of hyperedge
        weight: weight of hyperedge
        '''
        edge = tuple(sorted(e))

        self._edges[edge] = self._edges.get(edge,0)+1

        for v in edge:
            node_dict = self._nodes.get(v, {})

            neighbors = node_dict.get('neighbors', set())
            for v0 in edge:
                if v0!=v:
                    neighbors.add(v0)
            node_dict['neighbors'] = neighbors

            if self.graph_type=='1':
                for i,k in enumerate(self.cumsum):
                    if int(v) < k:
                        break
                node_dict['type'] = i

            self._nodes[v] = node_dict

    def edge_weight(self, e):
        '''weight of weight e'''
        return self._edges.get(e,0)

    def nodes(self):
        '''node set'''
        return self._nodes.keys()

    def edges(self):
        '''edge set'''
        return self._edges.keys()

    def neighbors(self, n):
        '''neighbors of node n'''
        return self._nodes[n]['neighbors']

    def node_type(self, n):
        '''type of node n'''
        return self._nodes[n]['type']       


def get_indecom_factor(G, r):
    '''
    Get the indecomposable factor of heterogeneous hyper-network G.
    '''
    edges = list(G.edges())

    k = len(G.nums_type)
    m = len(edges)

    dcnt = []
    for i in range(k):
        dcnt.append({})
    for edge in edges:
        edge = list(edge)
        for i in range(k):
            subedge = tuple(sorted(edge[:i]+edge[i+1:]))
            dcnt[i][subedge] = dcnt[i].get(subedge,0)+1

    factors = [0]*k
    for edge in edges:
        edge = list(edge)
        for i in range(k):
            subedge = tuple(sorted(edge[:i]+edge[i+1:]))
            if dcnt[i].get(subedge,0)>1:
                factors[i]+=1

    factors = [factor/m for factor in factors]

    cumsum = [0]+list(G.cumsum)
    ps = [0]*k
    neg_num = m*r  # sample enough random edges

    for i in tqdm(range(neg_num),ascii=True):
        random_edge = []
        for i in range(k):
            random_edge.append(np.random.randint(cumsum[i],cumsum[i+1]))
        for i in range(k):
            subedge = tuple(sorted(random_edge[:i]+random_edge[i+1:]))
            if dcnt[i].get(subedge,0)>1 or (dcnt[i].get(subedge,0)>0 and tuple(random_edge) not in edges):
                ps[i]+=1

    ps = [p/neg_num for p in ps]
    indecom_factors = [ps[i]/factors[i] for i in range(k)]

    return indecom_factors
