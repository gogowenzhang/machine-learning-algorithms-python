import numpy as np


class Bandits(object):
    ''' This class represent n bandit machines.

    Methods
    --------
    pull(i) : Sample from the ith bandit

    Attributes
    -----------
    p_array : Array of probabilities (probability of conversion)
    optimal : Index of the optimal bandit
    '''

    def __init__(self, p_array):
        '''
        Parameters
        -----------
        p_array : Array of probabilities to initialize bandits with
        '''
        self.p_array = p_array
        self.optimal = np.argmax(p_array)

    def pull(self, i):
        ''' Sample from a given bandit

        Parameters
        -----------
        i : int indicating the index of a bandit

        Returns
        --------
        bool indicating whether the bandit returned a reward or not
        '''
        return np.random.random() < self.p_array[i]

    def __len__(self):
        return len(self.p_array)
