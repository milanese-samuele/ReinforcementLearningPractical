#!/usr/bin/env python3
import random

import numpy as np
from bandit import *

class Agent(object):

    def __init__(self, bandits : list[Bandit]):
        self.bandits = bandits
        self.bestcount = 0
        """
        initializes every agent with a list of bandits
        """
    def update_best_count(self, idx, best):
        if(idx == best):
            self.bestcount += 1

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
        super(EpsylonGreedyAgent, self).__init__([Bandit(random.randint(0,100)) for k in range(k)])
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

    def choose_bandit(self):
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
        return a
    """
    Chooses a bandit from the input list and returns the index of
    the chosen bandit
    """

    def update_results(self, idx, reward):
        # Update counts
        self.n += 1
        self.k_n[idx] += 1
        # Update total reward
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n

        # Update results for mean reward at the a_th k arm
        self.k_reward[idx] = self.k_reward[idx] + (reward - self.k_reward[idx]) / self.k_n[idx]

    def update_step(self, step):
        self.reward[step] = self.mean_reward

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
        reward = self.bandits[a].get_reward()

        # Update counts
        self.n += 1
        self.k_n[a] += 1

        # Update total reward
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n

        # Update results for mean reward at the a_th k arm
        self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) / self.k_n[a]

    def reset(self):
        self.__init__(self.k, self.eps, self.iters)

    # Run the modell
    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward

class GreedyAgent(Agent):
    def __init__(self, k, iters):
        super(GreedyAgent, self).__init__([Bandit(random.randint(0,100)) for k in range(k)])
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

    def choose_bandit(self):
        return np.argmax(self.k_reward)

    def update_results(self, idx, reward):
        # Update counts
        self.n += 1
        self.k_n[idx] += 1

        # Update total reward
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n

        # Update results for mean reward at the a_th k arm
        self.k_reward[idx] = self.k_reward[idx] + (reward - self.k_reward[idx]) / self.k_n[idx]

    def update_step(self, idx):
        self.reward[idx] = self.mean_reward

    def pull(self):
        a = np.argmax(self.k_reward)
        # Draw a reward from Gaussian distribution randomly with a random mu, dont know if that correct tho
        # reward = np.random.normal(self.mu[a], 1) # Not using the function from bandit at the moment
        reward = self.bandits[a].get_reward()

        # Update counts
        self.n += 1
        self.k_n[a] += 1

        # Update total reward
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n

        # Update results for mean reward at the a_th k arm
        self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) / self.k_n[a]

    def reset(self):
        self.__init__(self.k, self.iters)

    # Run the modell
    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward
