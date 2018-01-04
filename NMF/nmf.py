import numpy as np

class NMF(object):
    """A Non-Negative Matrix Factorization (NMF) model using the Alternating Least
    Squares (ALS) algorithm."""
    
    def __init__(self, k=10, max_iter=100, threshold=0.1):

        self.k = k
        self.max_iter = max_iter
        self.threshold = threshold

    def fit(self, V):
        self.V = V
        self.W = np.random.rand(V.shape[0], self.k)
        self.H = np.random.rand(self.k, V.shape[1])
        for i in range(self.max_iter):
            self.H = np.linalg.lstsq(self.W, self.V)[0].clip(min=0)
            self.W = np.linalg.lstsq(self.H.T, self.V.T)[0].T.clip(min=0)
            if i >=2 and self.reconstruction_error() < self.threshold:
                break
        return self

    def reconstruction_error(self):
        cost = self.V-np.dot(self.W, self.H)
        return np.sqrt(np.sum(cost**2))





