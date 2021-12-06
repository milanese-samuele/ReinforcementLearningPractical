import numpy as np
import math


class UCB():
    def __init__(self, k, c, iters):
        self.label = "UCB"
        self.k = k
        self.c = c
        self.k_n = np.zeros(k)
        self.k_reward = np.zeros(k)
        self.bestcount = np.zeros(iters)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.iters = iters
        self.max_upper_bound = 0
        self.sum_reward = np.zeros(k)
        self.n = 0

    def _compute_uncertainty(self, idx):
        if (self.k_n[idx] > 0):
            ret = self.k_reward[idx] + (self.c*math.sqrt(math.log(self.n)/self.k_n[idx]))
        else:
            ret = 1e100
        return ret


    def choose_bandit(self):
        uncertain_rewards = [self._compute_uncertainty(i) for i in range(self.k)]
        return np.argmax(uncertain_rewards)

    def update_results(self, idx, reward):
        self.n += 1
        self.k_n[idx] += 1
        # Update total reward
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n

        # Update results for mean reward at the a_th k arm
        self.k_reward[idx] = self.k_reward[idx] + (reward - self.k_reward[idx]) / self.k_n[idx]

    def reset(self):
        self.__init__(self.k, self.c, self.iters)

    def update_best_count(self, step, idx, best):
        if(idx == best):
            self.bestcount[step] = 1

    def update_bernoulli_count(self, step, reward):
        if (reward == 1):
            self.bestcount[step] = 1

    def update_step(self, idx):
        self.reward[idx] = self.mean_reward
