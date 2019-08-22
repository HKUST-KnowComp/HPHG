from keras.models import Model
from keras.layers import Input, Dense, Reshape, concatenate, Activation, Conv1D, GlobalMaxPooling1D
from keras.layers.merge import Dot
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from sklearn import metrics
from tqdm import tqdm
import random
import numpy as np
import copy


def link_prediction(model, dataset):
    X, y = [], []
    
    with open('graph/' + dataset + '/test.edgelist', 'r', encoding="utf8") as f:
        while True:
            line = f.readline().strip().split()
            if line==[]:
                break
            X.append(list(map(int,line[1:])))
            y.append(1)

    with open('graph/' + dataset + '/test_negative.edgelist', 'r', encoding="utf8") as f:
        while True:
            line = f.readline().strip().split()
            if line==[]:
                break
            X.append(list(map(int,line)))
            y.append(0)

    pred = model.predict(np.array(X))
    pred = [pred[i][0] for i in range(len(pred))]

    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    return metrics.auc(fpr, tpr)  


class hypergram():
    def __init__(self, G, dataset, walks=None, size=32, window=5, negative=5, pair_ratio=0., epochs=5, lambda0=1.):
        self.G = G
        if walks!=None:
            self.train(walks, dataset, size, window, negative, pair_ratio, epochs, lambda0)

    def _tuple_negative_sample(self, edge, num_neg_samples, pair_ratio):
        edges = self.G.edges()
        k = len(self.G.nums_type)
        ks = list(range(k))
        cumsum = [0] + list(self.G.cumsum)

        n_neg = 0
        neg_data = []
        while n_neg < num_neg_samples:
            index = edge.copy()
            mode = np.random.rand()
            if mode < pair_ratio:
                type_ = random.randint(0,k-1)
                node = random.randint(cumsum[type_],cumsum[type_+1]-1)
                index[type_] = node
            else:
                types_ = random.sample(ks,2)
                node_1 = random.randint(cumsum[types_[0]],cumsum[types_[0]+1]-1)
                node_2 = random.randint(cumsum[types_[1]],cumsum[types_[1]+1]-1)
                index[types_[0]] = node_1
                index[types_[1]] = node_2
            if tuple(sorted(index)) in edges:
                continue
            n_neg += 1
            neg_data.append(index)
        return neg_data

    def _is_hyperedge(self, edge):
        for i,node in enumerate(edge):
            if self.G.node_type(node)!=i:
                return False
        return True

    def _tuple_sample(self, walk, negative_samples, pair_ratio):
        x = []
        y = []
        k = len(self.G.nums_type)
        for pos in range(len(walk)):
            pos_l = pos-(k-1)
            pos_r = pos+(k-1)
            inds = [(pos_l,pos+1),(pos,pos_r+1)]

            for ind in inds:
                if ind[0]>=0 and ind[1]<len(walk):
                    edge = sorted(walk[ind[0]:ind[1]])
                    if not self._is_hyperedge(edge):
                        continue
                    x.append(edge)
                    y.append(1)
                    neg_edges = self._tuple_negative_sample(edge, num_neg_samples=negative_samples, pair_ratio=pair_ratio)
                    for neg_edge in neg_edges:
                        x.append(neg_edge)
                        y.append(0)
        # shuffle
        c = list(zip(x, y))
        if c:
            random.shuffle(c)
            x,y = zip(*c)
            return x, y
        else:
            return [],[]

    def _compute_num_of_tuples(self, walk):
        k = len(self.G.nums_type)
        cnt = 0
        for pos in range(len(walk)):
            pos_l = pos-(k-1)
            pos_r = pos+(k-1)
            inds = [(pos_l,pos+1),(pos,pos_r+1)]

            for ind in inds:
                if ind[0]>=0 and ind[1]<len(walk):
                    edge = sorted(walk[ind[0]:ind[1]])
                    if self._is_hyperedge(edge):
                        cnt+=1
        return cnt

    def train(self, walks, dataset, size, window, negative, pair_ratio, epochs, lambda0):
        vocab_size = len(self.G.nodes())
        k = len(self.G.nums_type)

        # pairwise model
        input_target = Input((1,))
        embedding_target = Embedding(vocab_size, size, name="emb_0")
        target = embedding_target(input_target)

        input_context = Input((1,))

        context = Embedding(vocab_size, size)(input_context)

        dot_product = Dot(axes=2)([target, context])
        dot_product = Reshape((1,))(dot_product)
        output = Activation('sigmoid')(dot_product)

        model_p = Model(input=[input_target, input_context], output=output)
        model_p.summary()
        model_p.compile(loss='binary_crossentropy', optimizer='rmsprop', loss_weights=[1])

        # tuplewise model
        input_tuplew = Input(shape=(k, ), name='input', dtype='int32')
        tuplew = embedding_target(input_tuplew)
        conv = Conv1D(32, 3, activation='relu',name='conv')(tuplew)
        pooling = GlobalMaxPooling1D(name='pooling')(conv)
        output_tuplew = Dense(1,activation='sigmoid')(pooling)

        model_t = Model(input=input_tuplew, output=output_tuplew)
        model_t.summary()
        model_t.compile(loss='binary_crossentropy', optimizer='rmsprop', loss_weights=[lambda0])

        for epoch in range(epochs):

            random.shuffle(walks)

            loss_p = 0.
            loss_t = 0.
            for walk in tqdm(walks,ascii=True):
                pairs, labels_p = skipgrams(walk, vocab_size, negative_samples=negative, window_size=window)
                tuples, labels_t = self._tuple_sample(walk, negative_samples=negative, pair_ratio=pair_ratio)

                if pairs:
                    x_pair = [np.array(x) for x in zip(*pairs)]
                    y_pair = np.array(labels_p, dtype=np.int32)
                    loss_p += model_p.train_on_batch(x_pair,y_pair)

                if tuples:
                    x_tuple = np.asarray(tuples)
                    y_tuple = np.array(labels_t, dtype=np.int32)
                    loss_t += model_t.train_on_batch(x_tuple,y_tuple)                 

            print("epoch:",epoch+1)
            print("loss:",loss_p/len(walks),loss_t/len(walks),(loss_p+lambda0*loss_t)/len(walks))

        self.wv = model_p.get_layer('emb_0').get_weights()[0]
        self.model_t = model_t





