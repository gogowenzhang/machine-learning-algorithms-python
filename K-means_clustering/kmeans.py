import numpy as np
from scipy.spatial.distance import cdist

class Kmeans(object):
    """kmeans algorithm"""
    def __init__(self, k=10, max_iter=1000):
        '''
        - clusters: array of cluster labels of obervations
        - centroids: array of centers of each cluster
        - centroids_history: list of centroids in each iteration
        - distances: a distance matrix between observations and centroids
        '''

        self.k = k
        self.max_iter = max_iter
        self.X = None
        self.clusters = None
        self.centroids = None
        self.centroids_history = []
        self.distances = None

    def fit(self, X):
        self.X = X
        self.clusters = np.zeros(X.shape[0])
        start_point_index = np.random.choice(X.shape[0], size=self.k, replace=False)
        self.centroids = X[start_point_index]
        for j in range(self.max_iter):
            self.distances = cdist(X, self.centroids)
            self.centroids_history.append(self.centroids)
            for i in range(self.X.shape[0]):
                self.clusters[i] = np.argmin(self.distances[i])
            for i in range(self.k):
                self.centroids[i] = np.mean(X[self.clusters==i], axis=0)
            if j >= 2 and np.all(self.centroids == self.centroids_history[-1]):
                break
        return self










        

