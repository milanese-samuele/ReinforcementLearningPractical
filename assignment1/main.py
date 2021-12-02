#!/usr/bin/env python3

from bandit import *
from agent import *
import random
import matplotlib.pyplot as plt

## PARAMETERs FOR EXPERIMENT
N = 1000
K = 20

SEED = 42

alpha = 0.01
epsilon = 0.1

def plotResults(eps_rewards, Greed_rewards):
    plt.figure(figsize=(11,9))
    plt.plot(eps_rewards, color="green", label="$\epsilon$ Greedy")
    plt.plot(Greed_rewards, color="red", label="$\epsilon$")
    plt.legend(bbox_to_anchor=(1.4, 0.6))
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


if __name__ == '__main__':
    main()
