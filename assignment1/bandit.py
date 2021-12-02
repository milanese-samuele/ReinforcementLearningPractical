#!/usr/bin/env python3

import random
from scipy.stats import bernoulli
import numpy as np

## Distributions charachteristics
MU = 0.0
SIGMA = 3.0

class Bandit(object):

    def __init__(self, seed):
        self.mu = MU
        self.sigma = SIGMA
        self.generator = random.Random(seed)
        self.p = 0.5
        """
        Constructor which instantiates a unique generator that does not
        share state with other random generators in the program
        """

    def get_reward(self) -> float :
        return self.generator.gauss(self.mu, self.sigma)
        """
        Returns the reward by sampling the distribution
        """

    def get_reward_bernoulli(self) -> float:
        return bernoulli.rvs(self.p)
    """
    Random reward with bernoulli
    """
