import numpy as np
import argparse
from sklearn import metrics
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Link prediction.")
    parser.add_argument('--input', nargs='?',
                        help='Input embedding path.')
    return parser.parse_args()


def link_predict(filename, sim, dataset):
    f = open(filename, 'r', encoding="utf8")

    embs = {}
    f.readline()

    while True:
        line = f.readline().strip().split()
        if line==[]:
            break
    
        name = line[0]
        del line[0]
    
        vect = []
        for m in line:
            try:
                vect.append(float(m))
            except BaseException:
                vect.append(0)
        embs[name] = vect
    
    test = []
    test_neg = []
    
    f1 = open('graph/' + dataset + '/test.edgelist', 'r', encoding="utf8")
    
    while True:
        line = f1.readline().strip().split()
        if line==[]:
            break
    
        name = line[0]
        del line[0]
    
        vect = []
        for m in line:
            vect.append(m)
        test.append(vect)
    
    f1 = open('graph/' + dataset + '/test_negative.edgelist', 'r', encoding="utf8")
    
    while True:
        line = f1.readline().strip().split()
        if line==[]:
            break
    
        vect = []
        for m in line:
            vect.append(m)
        test_neg.append(vect)
    
    y = []
    pred = []
    
    for edge in test:
        flag = False
        for node in edge:
            if embs.get(node)==None:
                flag = True
                break
        if flag:
            continue
        d = hyperedge_dist(edge, embs, sim)
        y.append(1)
        pred.append(d)
    
    for edge in test_neg:
        flag = False
        for node in edge:
            if embs.get(node)==None:
                flag = True
                break
        if flag:
            continue
        d = hyperedge_dist(edge, embs, sim)
        y.append(0)
        pred.append(d)
    
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    if sim == COS:
        return metrics.auc(fpr, tpr)
    else:
        return 1-metrics.auc(fpr, tpr)

def main(args):
    funcs = [L1D, L2D, COS]
    dataset = args.input.split('/')[1]  # extract dataset name

    for sim in funcs:
        print(sim.__name__, end=': ')
        auc = link_predict(args.input, sim, dataset)
        print(auc)


if __name__ == "__main__":
    args = parse_args()
    main(args)



