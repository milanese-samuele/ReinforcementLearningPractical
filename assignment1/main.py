#!/usr/bin/env python3

from bandit import *
from agent import *
import random
import matplotlib.pyplot as plt
import numpy as np


## PARAMETERs FOR EXPERIMENT
N = 500
K = 4

SEED = 42

alpha = 0.01
epsilon = 0.1

algorithms = [ EpsylonGreedyAgent(K, epsilon, N) ]

def plotResults(eps_rewards, Greed_rewards):
    plt.figure(figsize=(11,9))
    plt.plot(eps_rewards, color="green", label="$\epsilon$ Greedy")
    plt.plot(Greed_rewards, color="red", label="greedy")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Rewards after " + str(N)
              + " iterations")
    plt.show()


def main():

    bandits = [Bandit(random.random() * SEED+idx) for idx in range(K)]
    a = RandomAgent(bandits)
    eps_rewards = np.zeros(N)
    Greed_rewards = np.zeros(N)
    for idx in range(N):
        # print(a.act())

        # For Greedy
        greed = GreedyAgent(K, N)
        greed.run()
        Greed_rewards = Greed_rewards + (greed.reward - Greed_rewards) / (idx + 1)

        # For Epsilon greedy
        eps = EpsylonGreedyAgent(K, epsilon, N)
        eps.run()
        eps_rewards = eps_rewards + (eps.reward - eps_rewards) / (idx + 1)
    # Plot the results
    plotResults(eps_rewards, Greed_rewards)

def altmain():
    bandits = [Bandit(idx) for idx in range(K)]
    eps_rewards = np.zeros(N)
    for step in range(N):
        for i in range(N):
            rewards = np.array([bandits[idx].get_reward() for idx in range(K)])
            best_bidx = np.argmax(rewards)
            for alg in algorithms:
                choice = alg.choose_bandit()
                alg.update_best_count(choice, best_bidx)
                alg.update_results(choice, rewards[choice])
                alg.update_step(i)

        eps_rewards = eps_rewards + (algorithms[0].reward - eps_rewards) / (step + 1)
    print("{}/{} = {}%".format(algorithms[0].bestcount,
                               N*N,
                              (algorithms[0].bestcount*100)/(N*N)))
    plt.figure(figsize=(11,9))
    plt.plot(eps_rewards, color="green", label="$\epsilon$ Greedy")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Rewards after " + str(N)
              + " iterations")
    plt.show()




if __name__ == '__main__':
    altmain()
