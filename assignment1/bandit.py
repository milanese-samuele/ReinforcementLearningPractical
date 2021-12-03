#!/usr/bin/env python3

import random
from scipy.stats import bernoulli
import numpy as np

## Distributions charachteristics
MU = 1.0
SIGMA = 2.0

class Bandit(object):

    def __init__(self, idx):
        self.mu = np.random.normal(0,2)
        self.sigma = np.random.normal(1,3)
        self.idx = idx
        self.generator = random.Random()
        self.p = 0.5
        """
        Constructor which instantiates a unique generator that does not
        share state with other random generators in the program
        """

    def get_reward(self) -> float :
        return self.generator.gauss(self.mu, self.sigma)
        # return np.random.normal(self.mu, self.sigma)
        """
        Returns the reward by sampling the distribution
        """

    def get_reward_bernoulli(self) -> float:
        return np.random.binomial(1, self.p, size=1)[0]
    """
    Random reward with bernoulli
    """
