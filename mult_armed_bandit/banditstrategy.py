import scipy.stats as scs
import numpy as np
import random


class BanditStrategy(object):
    '''
    Implements a online, learning strategy to solve the Multi-Armed Bandit
      problem.

    Methods
    --------
    sample_bandits(n): sample and train on n pulls.

    Attributes
    -----------
    N : the cumulative number of samples
    choices : the historical choices as a (N,) array
    bb_score: the historical score as a (N,) array
    '''

    def __init__(self, bandits, choice_function, seed=None, **kwargs):
        '''
        Initializes the BanditStrategy given an instance of the Bandits class
          and a choice function.

        Parameters
        -----------
        bandits : a Bandit class with .pull method
    	choice_function : str indicating the choice function or a custom
            choice fuction accepting a self argument (which gives access to
            all the BanditStrategy's attributes) which returns an int
            between 0 and n-1
            Accepted str inputs include 'max_mean', 'random_choice',
            'epsilon_greedy', 'softmax', 'ucb1', and 'bayesian_bandit'
        seed : int setting the random seed or None to not seed the random
            number generator
        Also supports passing keyword args which can be used in a
            particular choice function.  Namely 'epsilon_greedy' accepts
            epsilon=float and 'softmax' accepts tau=float
            e.g. strat = BanditStrategy(bandits, 'epsilon_greedy',
              epsilon=0.1)
        '''
        choice_dict = {'max_mean': self._max_mean,
                       'random_choice': self._random_choice,
                       'epsilon_greedy': self._epsilon_greedy,
                       'softmax': self._softmax,
                       'ucb1': self._ucb1,
                       'bayesian_bandit': self._bayesian_bandit}
        self.bandits = bandits
        n_bandits = len(self.bandits)
        self.wins = np.zeros(n_bandits)
        self.trials = np.zeros(n_bandits)
        self.N = 0
        self.choices = []
        self.score = []
        self.seed = seed
        self._kwargs = kwargs
        if isinstance(choice_function, str):
            if choice_function in choice_dict:
                self.choice_function = choice_dict[choice_function]
            else:
                raise ValueError("{0} not a valid choice function. Valid choices include {1}".format(
                    choice_function, choice_dict.keys()))
        elif not hasattr(choice_function, '__call__'):
            raise ValueError("{0} is not a valid input. Must be str or callable function accepting self".format(
                choice_function))
        else:
            BanditStrategy.choice_function = choice_function

    def sample_bandits(self, n=1):
        ''' Simulate n rounds of running the bandit machine

        Parameters
        -----------
        n : int number of rounds
        '''
        score = np.zeros(n)
        choices = np.zeros(n)

        # seed the random number generators so you get the same results every
        # time.
        if self.seed:
            np.random.seed(self.seed)
            random.seed(self.seed)

        for k in range(n):
            #sample from the bandits's priors, and select the largest sample
            choice = self.choice_function()

            #sample the chosen bandit
            result = self.bandits.pull(choice)

            #update priors and score
            self.wins[choice] += result
            self.trials[choice] += 1
            score[k] = result
            self.N += 1
            choices[k] = choice

        self.score = np.r_[self.score, score]
        self.choices = np.r_[self.choices, choices]

    def _max_mean(self):
        ''' Pick the bandit with the current best observed proportion of winning

        Returns
        --------
        int : index of the winning bandit
        '''
        # make sure to play each bandit at least once
        if self.trials.min() == 0:
            return np.argmin(self.trials)
        return np.argmax(self.wins / self.trials)

    def _random_choice(self):
        ''' Pick a bandit uniformly at random

        Returns
        --------
        int : index of the winning bandit
        '''
        return np.random.randint(0, len(self.wins))

    def _epsilon_greedy(self):
        '''
        Pick a bandit uniformly at random epsilon percent of the time.
        Otherwise pick the bandit with the best observed proportion of winning

        Returns
        --------
        int : index of the winning bandit
        '''
        # Set default value of epsilon if not provided in init
        if self.trials.min() == 0:
            return np.argmin(self.trials)
        epsilon = self._kwargs.get('epsilon', 0.1)
        if random.random < epsilon:
            return np.random.randint(len(self.bandits))
        else: 
            return np.argmax(self.wins / self.trials)
        


    def _softmax(self):
        ''' Pick an bandit according to the Boltzman Distribution

        Returns
        --------
        int : index of the winning bandit
        '''
        # Set default value of tau if not provided in init
        tau = self._kwargs.get('tau', 0.01)
        if self.trials.min() == 0:
            return np.argmin(self.trials)
        prop = np.exp(self.wins/self.trials/tau) 
        prop_sum = np.sum(prop)
        p = prop/prop_sum
        return np.random.choice(range(len(self.wins)), p=p)
        


    def _ucb1(self):
        ''' Pick the bandit according to the UCB1 strategy

        Returns
        --------
        int : index of the winning bandit
        '''
        if self.trials.min() == 0:
            return np.argmin(self.trials)
        return np.argmax(self.wins/self.trials + np.sqrt((2. * np.log(self.N))/self.trials))
        


    def _bayesian_bandit(self):
        '''
        Randomly sample from a beta distribution for each bandit and pick
        the one with the largest value

        Returns
        --------
        int : index of the winning bandit
        '''
        if self.trials.min() == 0:
            return np.argmin(self.trials)
        samples = [np.random.beta(a=1+ wins, b=1+trials-wins)
                    for wins, trials in zip(self.wins, self.trials)]
        return np.argmax(samples)        
        
