import numpy as np
import math


class UCB():
    def __init__(self, k, iters):
        self.label = "UCB"
        self.k = k
        self.k_n = np.zeros(k)
        self.k_reward = np.zeros(k)
        self.bestcount = 0
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.iters = iters
        self.max_upper_bound = 0
        self.sum_reward = np.zeros(k)
        self.n = 0

    def choose_bandit(self):
        for i in range(1, self.iters + 1):
            max_upper_bound = 0
            for idx in range(0, self.k):
                if self.k_n[idx] > 0:
                    avg_rank = self.sum_reward[idx] / self.k_n[idx]
                    delta_i = math.sqrt(1.5 * (math.log(i) / self.k_n[idx]))
                    upper_bound = avg_rank + delta_i
                else:
                    upper_bound = 1e300
                if upper_bound > max_upper_bound:
                    max_upper_bound = upper_bound
                    choice = idx
        return choice

    def update_results(self, idx, reward):
        self.n += 1
        self.k_n[idx] += 1
        # Update total reward
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n

        # Update results for mean reward at the a_th k arm
        self.k_reward[idx] = self.k_reward[idx] + (reward - self.k_reward[idx]) / self.k_n[idx]

    def reset(self):
        self.__init__(self.k, self.iters)

    def update_best_count(self, idx, best):
        if(idx == best):
            self.bestcount += 1

    def update_step(self, idx):
        self.reward[idx] = self.mean_reward