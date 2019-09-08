import numpy as np
import sys
sys.path.append('C:\\Users\\dsz62\\Desktop\\2019_Fall\\cs231n\\assignment1\\cs231n')
from data_utils import load_CIFAR10


class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X, k):
        num_test = X.shape[0]
        ypred = np.zeros(num_test, dtype=self.ytr.dtype)
        for i in range(num_test):
            if i % 100 == 0:
                print('%d / %d' % (i, num_test))
            # distances = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)  # L1
            distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis=1))  # L2
            min_index = np.argmin(distances)
            ypred[i] = self.ytr[min_index]
        return ypred


Xtr, ytr, Xte, yte = load_CIFAR10('C:/Users/dsz62/Desktop/2019_Fall/cs231n/assignment1/cs231n/datasets/cifar-10-batches-py/')
Xtr_rows = Xtr.reshape(Xtr.shape[0], -1)
Xte_rows = Xte.reshape(Xte.shape[0], -1)
dists = np.sqrt(np.sum(np.square(np.reshape(Xte_rows, (Xte_rows.shape[0], 1, -1)) - Xtr_rows), axis=2))
