from DecisionTree import DecisionTree
import numpy as np
from collections import Counter

class RandomForest(object):
    '''A Random Forest class'''

    def __init__(self, num_trees, num_features):
        '''
           num_trees:  number of trees to create in the forest:
        num_features:  the number of features to consider when choosing the
                           best split for each node of the decision trees
        '''
        self.num_trees = num_trees
        self.num_features = num_features
        self.forest = None

    def fit(self, X, y):
        '''
        X:  two dimensional numpy array representing feature matrix
                for test data
        y:  numpy array representing labels for test data
        '''
        self.forest = self.build_forest(X, y, self.num_trees, self.num_features)

    def build_forest(self, X, y, num_trees, num_features):
        '''
        Return a list of num_trees DecisionTrees.
        '''
        X_train = []
        y_train = []
        forest = []
        # Bagging
        for _ in range(num_trees):
            for _ in range(X.shape[0]):
                row_selected = np.random.choice(X.shape[0], X.shape[0], 
                    replace=True)
                X_train.append(X[row_selected])
                y_train.append(y[row_selected])
            dt = DecisionTree(num_features=self.num_features)
            dt.fit(np.array(X_train), np.array(y_train))
            forest.append(dt)
        return forest 
        

    def predict(self, X):
        '''
        Return a numpy array of the labels predicted for the given test data.
        '''    
        answers = np.array([tree.predict(X) for tree in self.forest]).T
        return np.array([Counter(row).most_common(1)[0][0] for row in answers])

        return np.array(result)

    def score(self, X, y):
        '''
        Return the accuracy of the Random Forest for the given test data and
        labels.
        '''
        y_predict = self.predict(X)
        return np.mean(y_predict==y)
        
