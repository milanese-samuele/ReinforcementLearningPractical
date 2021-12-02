#!/usr/bin/env python3
import random

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


class EpsylonGreedyAgent(Agent):

    def __init__(self, k, eps, iters):
        # Number of arms
        self.k = k
        # Epsilon
        self.eps = eps
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 0
        # Step count for each arm
        self.k_n = np.zeros(k)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)
        # Random mean generated not sure if that is correct tho
        self.mu = np.random.normal(0, 1, k)
        # Random sd generated not sure if that is correct tho
        self.sd = np.random.normal(0, 2, k)

    def pull(self):
        # Random number to decide whether to exploit or explore
        p = np.random.rand()
        # Explore in the first run
        if self.eps == 0 and self.n == 0:
            a = np.random.choice(self.k)
        # Explore
        elif p < self.eps:
            # Randomly select an action
            a = np.random.choice(self.k)
        # Exploit with the largest reward until t
        else:
            # Take greedy action
            a = np.argmax(self.k_reward)

        # Draw a reward from Gaussian distribution randomly with a random mu, dont know if that correct tho
        reward = np.random.normal(self.mu[a], 1) # Not using the function from bandit at the moment

        # Update counts
        self.n += 1
        self.k_n[a] += 1

        # Update total reward
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n

        # Update results for mean reward at the a_th k arm
        self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) / self.k_n[a]

    # Run the modell
    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward

class GreedyAgent(Agent):
    def __init__(self, k, iters):
        # Number of arms
        self.k = k
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 0
        # Step count for each arm
        self.k_n = np.zeros(k)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)
        # Random mean generated not sure if that is correct tho
        self.mu = np.random.normal(0, 1, k)
        # Random sd generated not sure if that is correct tho
        self.sd = np.random.normal(0, 2, k)

    def pull(self):
        a = np.argmax(self.k_reward)
        # Draw a reward from Gaussian distribution randomly with a random mu, dont know if that correct tho
        reward = np.random.normal(self.mu[a], 1) # Not using the function from bandit at the moment

        # Update counts
        self.n += 1
        self.k_n[a] += 1

        # Update total reward
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n

        # Update results for mean reward at the a_th k arm
        self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) / self.k_n[a]

    # Run the modell
    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward
