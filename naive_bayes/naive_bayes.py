from __future__ import division
from collections import Counter, defaultdict
import numpy as np
import itertools

class NaiveBayes(object):
    def __init__(self, alpha=1.):
        """
        INPUT:
        -alpha: float, laplace smoothing constant.

        ATTRIBUTES:
        - class_counts: the number of samples per class; keys=labels
        - class_feature_counts: the number of samples per feature per label;
                               keys=labels, values=Counter with key=feature
        - class_freq: the frequency of each class in the data
        - p: the number of features
        """
        self.class_counts = defaultdict(int)
        self.class_feature_counts = defaultdict(Counter)
        self.class_freq = None
        self.alpha = float(alpha)
        self.p = None

    def _compute_likelihoods(self, X, y):
        '''
        INPUT:
        - X: List of list of tokens.
        - y: numpy array, labels

        OUTPUT: None

        Compute the word count for each class and the frequency of each feature
        per class.  (Compute class_counts and class_feature_counts).
        '''
        for i in range(y.shape[0]):
            self.class_counts[y[i]] += len(X[i])
            self.class_feature_counts[y[i]] += Counter(X[i])
        
    def fit(self, X, y):
        '''
        INPUT:
        - X: List of list of tokens.
        - y: numpy array, labels

        OUTPUT: None
        '''
        # Compute class frequency P(y)
        self.class_freq = Counter(y)

        # Compute number of features
        self.p = len(set(itertools.chain(*X)))

        # Compute likelihoods
        self._compute_likelihoods(X, y)

    def posteriors(self, X):
        '''
        INPUT:
        - X: List of list of tokens.

        OUTPUT:
        List of dictionaries with key=label, value=log(p(label|X))
        '''
        result = []
        for row in X:
            posteriors_dict = defaultdict(float)
            c = Counter(row) 
            for label, count in self.class_freq.iteritems():
                posteriors_dict[label] += np.log(1. * count/sum(self.class_freq.values()))
                for word, w_count in c.iteritems():
                    posteriors_dict[label] += w_count * np.log(1. * (self.class_feature_counts[label][word] 
                        + self.alpha) / (self.class_counts[label] + self.alpha* self.p))
            result.append(posteriors_dict)
        return result


    def predict(self, X):
        """
        INPUT:
        - X: A list of lists of tokens.

        OUTPUT:
        - predictions: a numpy array with predicted labels.

        """
        predictions = []
        for post in self.posteriors(X):
            pred = max(post.iterkeys(), key=(lambda label: post[label]))
            predictions.append(pred)
        return np.array(predictions)

    def score(self, X, y):
        '''
        INPUT:
        - X: List of list of tokens.
        - y: numpy array, labels

        OUTPUT:
        - accuracy: float between 0 and 1

        Calculate the accuracy, the percent predicted correctly.
        '''

        return np.mean(self.predict(X) == y)
