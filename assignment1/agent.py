#!/usr/bin/env python3

import numpy as np
from bandit import *

class Agent(object):

    def __init__(self, bandits : list[Bandit]):
        self.bandits = bandits
        """
        initializes every agent with a list of bandits
        """

class RandomAgent(Agent):

    def __init__(self, bandits : list[Bandit]):
        super(RandomAgent, self).__init__(bandits)
        """
        Initializes random agent with bandits list
        """

    def _random_bandit(self) -> Bandit:
        return np.random.choice(self.bandits)
        """
        selects a random bandit from list
        """


    def act(self) -> float:
        return self._random_bandit().get_reward()
        """
        method where the agent should use the strategy
        """
