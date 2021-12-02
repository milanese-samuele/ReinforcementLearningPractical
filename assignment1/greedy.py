#!/usr/bin/env python3

import numpy as np

class Greedy(object):

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
        # Keeps track of how many times the optimal soulution was picked
        self.bestcount = 0

    def reset(self):
        self.__init__(self.k, self.iters)

    def update_best_count(self, idx, best):
        if(idx == best):
            self.bestcount += 1

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

class EpsilonGreedy(Greedy):

    def __init__(self, k, epsilon, iters):
        super(EpsilonGreedy, self).__init__(k, iters)
        self.eps = epsilon

    def reset(self):
        self.__init__(self.k, self.eps, self.iters)

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
