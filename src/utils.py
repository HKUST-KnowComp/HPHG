import math
import numpy as np


def L2D(x, y):
    ret = 0
    for i in range(len(x)):
        ret += (x[i]-y[i])**2
    return math.sqrt(ret)

def L1D(x, y):
    ret = 0
    for i in range(len(x)):
        ret += np.abs(x[i]-y[i])
    return ret

def COS(x,y):
    return np.dot(x,y)/math.sqrt(np.dot(x,x)*np.dot(y,y))

def hyperedge_dist(edge, emb, sim):
    dists = []
    for a in edge:
        for b in edge:
            if a!=b:
                dists.append(sim(emb[a], emb[b]))
    return np.mean(dists)