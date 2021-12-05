#!/usr/bin/env python3

import random
from scipy.stats import bernoulli
import numpy as np

## Distributions charachteristics intervals
MU = (1,2)
SIGMA = (3,5)


class Bandit(object):

    def __init__(self, idx):
        self.mu = np.random.normal(*MU)
        self.sigma = np.random.normal(*SIGMA)
        self.idx = idx
        self.generator = random.Random()
        self.p = random.random()
        """
        Constructor which instantiates a unique generator that does not
        share state with other random generators in the program
        """

    def reset(self):
        self.__init__(self.idx)

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
