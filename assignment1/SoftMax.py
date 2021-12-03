#!/usr/bin/env python3


import random
import numpy as np
import math


def draw(probs):
    rand = random.random()
    sum_prob = 0.0

    for i in range(len(probs)):
        prob = probs[i]
        sum_prob += prob

        if sum_prob > rand:
            return i
    return len(probs) - 1


class Softmax:
    def __init__(self, k, tau, iters):
        self.label = "Softmax"
        self.tau = tau
        self.k = k
        self.k_n = np.zeros(k)
        self.k_reward = np.zeros(k)
        self.bestcount = 0
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.iters = iters

    def choose_bandit(self):
        # Calculate Softmax probabilities based on each round
        temp = sum([math.exp(x / self.tau) for x in self.k_reward])
        probs = [math.exp(x / self.tau) / temp for x in self.k_reward]

        # pick arm with draw
        return draw(probs)

    # Choose to update chosen arm and reward
    def update_results(self, idx, reward):
        # update counts pulled for chosen arm
        self.k_n[idx] += 1
        n = self.k_n[idx]

        # Update total reward
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / n

        # Update mean reward for the chosen arm
        """""
        k_reward = self.k_reward[idx]
        new_k_reward = ((n - 1) / float(n)) * k_reward + (1 / float(n)) * reward
        self.k_reward[idx] = new_k_reward
        """""
        self.k_reward[idx] = self.k_reward[idx] + (reward - self.k_reward[idx]) / n

    def reset(self):
        self.__init__(self.k, self.tau, self.iters)

    def update_best_count(self, idx, best):
        if(idx == best):
            self.bestcount += 1

    def update_step(self, idx):
        self.reward[idx] = self.mean_reward

