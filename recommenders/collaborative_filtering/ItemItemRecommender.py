import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


class ItemItemRecommender(object):
    def __init__(self, neighborhood_size):
        '''
        Initialize the parameters of the model.
        '''
        self.neighborhood_size = neighborhood_size
        self.ratings_mat = None
        self.items_cos_sim = None
        self.neighborhoods = None
        self.prediction = None


    def fit(self, ratings_mat):
        '''
        Implement the model and fit it to the data passed as an argument.

        Store objects for describing model fit as class attributes.
        '''
        self.ratings_mat = ratings_mat
        self._set_neighborhoods()

    
    def _set_neighborhoods(self):
        '''
        Get the items most similar to each other item.

        Should set a class attribute with a matrix that is has
        number of rows equal to number of items and number of 
        columns equal to neighborhood size. Entries of this matrix
        will be indexes of other items.

        You will call this in your fit method.
        '''
        self.items_cos_sim = cosine_similarity(self.ratings_mat.T)
        least_to_most_sim_indexes = np.argsort(self.items_cos_sim, 1)
        self.neighborhoods = least_to_most_sim_indexes[:, -self.neighborhood_size:]

    def pred_one_user(self, user_id):
        '''
        Accept user id as arg. Return the predictions for a single user.
        
        Optional argument to specify whether or not timing should be provided
        on this operation.
        '''
        n_items = self.ratings_mat.shape[1]
        items_rated_by_this_user = self.ratings_mat[user_id].nonzero()[1]
        # Just initializing so we have somewhere to put rating preds
        output = np.zeros(n_items)
        for item_to_rate in range(n_items):
            relevant_items = np.intersect1d(self.neighborhoods[item_to_rate],
                                            items_rated_by_this_user,
                                            assume_unique=True)
                                        # assume_unique speeds up intersection op
            output[item_to_rate] = self.ratings_mat[user_id, relevant_items] * \
                self.items_cos_sim[item_to_rate, relevant_items] / \
                self.items_cos_sim[item_to_rate, relevant_items].sum()
        return np.nan_to_num(output)

        
    def pred_all_users(self, verbose=False):
        '''
        Repeated calls of pred_one_user, are combined into a single matrix.
        Return value is matrix of users (rows) items (columns) and predicted
        ratings (values).

        Optional argument to specify whether or not timing should be provided
        on this operation.
        '''
        predictions = []
        for i in range(self.ratings_mat.shape[0]):
            predictions.append(self.pred_one_user(i))
            if verbose:
                print 'user:', i, 'predicted rate:', self.pred_one_user(i)
        return np.array(predictions)
        

    def top_n_recs(self, user_id, n):
        '''
        Take user_id argument and number argument.

        Return that number of items with the highest predicted ratings, after
        removing items that user has already rated.
        '''
        ratings = self.pred_one_user(user_id)
        item_index_sorted_by_pred_rating = list(np.argsort(ratings))
        items_rated_by_this_user = self.ratings_mat[user_id].nonzero()[1]
        unrated_items_by_pred_rating = [item for item in item_index_sorted_by_pred_rating
                                        if item not in items_rated_by_this_user]
        return unrated_items_by_pred_rating[-n:]

