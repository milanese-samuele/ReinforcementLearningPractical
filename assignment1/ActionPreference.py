import numpy as np
import math
import random

def draw(probs):
    rand = random.random()
    sum_prob = 0.0

    for i in range(len(probs)):
        prob = probs[i]
        sum_prob += prob

        if sum_prob > rand:
            return i
    return len(probs) - 1

class Action_Preference():
    def __init__(self, k, alpha, iters):
        self.label = "ActionPreference"
        self.k = k
        self.alpha = alpha
        self.k_n = np.zeros(k)
        self.k_reward = np.zeros(k)
        self.bestcount = np.zeros(iters)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.iters = iters
        self.max_upper_bound = 0
        self.sum_reward = np.zeros(k)
        self.n = 0
        self.H = np.zeros(k)
        self.probs = 0



    def choose_bandit(self):
        temp = sum([math.exp(x) for x in self.H])
        self.probs = [math.exp(x) / temp for x in self.H]

        return draw(self.probs)

    def update_results(self, idx, reward):
        self.n += 1
        self.k_n[idx] += 1

        # This line below does not seem to work
        self.H[idx] = self.H[idx] + self.alpha * (reward - self.mean_reward) * (1-self.probs[idx])

        # Update total reward
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n

        # Update results for mean reward at the a_th k arm
        self.k_reward[idx] = self.k_reward[idx] + (reward - self.k_reward[idx]) / self.k_n[idx]




    def reset(self):
        self.__init__(self.k, self.alpha, self.iters)

    def update_best_count(self, step, idx, best):
        if(idx == best):
            self.bestcount[step] = 1

    def update_bernoulli_count(self, step, reward):
        if (reward == 1):
            self.bestcount[step] = 1

    def update_step(self, idx):
        self.reward[idx] = self.mean_reward
