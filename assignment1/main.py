#!/usr/bin/env python3

from bandit import *
from agent import *
from greedy import *
import random
import matplotlib.pyplot as plt
import numpy as np
from SoftMax import *
from UCB import *
from ActionPreference import *


## PARAMETERs FOR EXPERIMENT
N = 1000
K = 5

SEED = 42
# BER or GAUSS
MODE = "GAUSS"
# MODE = "BER"

epsilon = 0.1
tau = 0.9
Qa = 50.0
c = 3.0
alpha = 0.1

ALGOS = [
          Greedy(K, N),
          EpsilonGreedy(K, epsilon, N),
          Softmax(K, tau, N),
          Optimistic(K, Qa, N),
          UCB(K, c, N),
          Action_Preference(K, alpha, N)
         ]

def plotResults(results, optimals):
    plt.figure(figsize=(11,9))
    fig, ax = plt.subplots(1,2)
    for i, p in enumerate(results):
        ax[0].plot(p, label=ALGOS[i].label)
    for i, p in enumerate(optimals):
        ax[1].plot(p, label=ALGOS[i].label)
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


def run_models(bandits, algorithms):
    for i in range(N):
        rewards = [bandits[idx].get_reward() for idx in range(K)]
        best_bidx = np.argmax(rewards)
        for alg in algorithms:
            choice = alg.choose_bandit()
            alg.update_best_count(i, choice, best_bidx)
            alg.update_results(choice, rewards[choice])
            alg.update_step(i)

def altmain():
    bandits = [Bandit(idx) for idx in range(K)]
    algorithms = ALGOS
    reward_records = [np.zeros(N) for _ in range(len(algorithms))]
    optimal_records = [np.zeros(N) for _ in range(len(algorithms))]
    # repetitions of experiment
    for rep in range(N):
        for bandit in bandits:
            bandit.reset()
        for alg in algorithms:
            alg.reset()
        run_models(bandits, algorithms)
        for idx, _ in enumerate(algorithms):
            reward_records[idx] = reward_records[idx] + (algorithms[idx].reward - reward_records[idx]) / (rep + 1)
            optimal_records[idx] = optimal_records[idx] + (algorithms[idx].bestcount - optimal_records[idx]) / (rep + 1)
    plotResults(reward_records, optimal_records)



if __name__ == '__main__':
    altmain()
