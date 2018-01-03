import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial.distance import cosine


def euclidean_distance(a, b):
    return np.sqrt(np.dot(a - b, a - b))

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))

class KNearestNeighbors(object):
    def __init__(self, k, distance):
        self.k = k
        self.distance = distance
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):  
        y_predict = []
        for i in range(X.shape[0]):
            dist_i = []
            for j in range(self.X.shape[0]):             
                dist_i.append(self.distance(X[i], self.X[j]))
            rank_index = np.argsort(dist_i)
            y_k_i = self.y[rank_index][:self.k]
            y_predict.append(Counter(y_k_i).most_common(1)[0][0])
        return np.array(y_predict)
 

    def score(self, X, y):
        return 1. * np.mean(self.predict(X) == y)


        
