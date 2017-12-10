import numpy as np

class FactorizationRecommender(object):
    """Implementation of Matrix Factorization Recommender using regularized 
    gradient descent algorithm"""
    def __init__(self, max_iter=5000, learning_rate=0.0002, 
        regualization=0.02):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.regualization = regualization
        self.rating_mat = None
        self.W = None
        self.H = None

    def fit(self, rating_mat, K, missing=0):
        self.rating_mat = rating_mat
        self.K = K
        self.missing = missing
        n = self.rating_mat.shape[0]
        m = self.rating_mat.shape[1]
        self.W = np.random.rand(n, self.K)
        self.H = np.random.rand(self.K, m)
        for step in range(self.max_iter):
            for i in range(n):
                for j in range(m):
                    if self.rating_mat[i][j] != missing:
                        eij = self.rating_mat[i][j] - np.dot(self.W[i], self.H[:,j])
                        for k in range(self.K):
                            self.W[i][k] += self.learning_rate * (2 * eij *
                                self.H[k][j] - self.regualization * self.W[i][k])
                            self.H[k][j] += self.learning_rate * (2 * eij *
                                self.W[i][k] - self.regualization * self.H[k][j])
        eR = np.dot(self.W, self.H)
        e = 0
        for i in range(n):
            for j in range(m):
                if self.rating_mat[i][j] != missing:
                    e = e + pow(self.rating_mat[i][j] - np.dot(self.W[i,:], self.H[:,j]), 2)
                    for k in range(self.K):
                        e = e + (self.regualization/2) * (pow(self.W[i][k],2) + pow(self.H[k][j],2))
            if e < 0.001:
                break

    def prediction(self):
        return np.dot(self.W, self.H)

    def reconstruction_error(self):
        e = 0
        for i in range(self.rating_mat.shape[0]):
            for j in range(self.rating_mat.shape[1]):
                if self.rating_mat[i][j] != self.missing:
                    e = e + pow(self.rating_mat[i][j] - np.dot(self.W[i,:], self.H[:,j]), 2)
        return np.sqrt(np.mean(e))

