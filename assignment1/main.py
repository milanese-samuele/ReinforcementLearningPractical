#!/usr/bin/env python3

from bandit import *
from greedy import *
import matplotlib.pyplot as plt
from SoftMax import *
from UCB import *
from ActionPreference import *


## PARAMETERs FOR EXPERIMENT
N = 1000
K = 3

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
    plt.subplot(1,2,1)
    for i, p in enumerate(results):
        plt.plot(p, label=ALGOS[i].label)
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Rewards after " + str(N)
              + " iterations")
    plt.subplot(1,2,2)
    for i, p in enumerate(optimals):
        plt.plot(p, label=ALGOS[i].label)
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Average solution %")
    plt.title("Optimal solution % per each step")
    plt.show()

def get_rewards(bandits):
    if (MODE=="GAUSS"):
        return [bandits[idx].get_reward() for idx in range(K)]
    else :
        return [bandits[idx].get_reward_bernoulli() for idx in range(K)]

def run_models(bandits, algorithms):
    for i in range(N):
        rewards = get_rewards(bandits)
        best_bidx = np.argmax(rewards)
        for alg in algorithms:
            choice = alg.choose_bandit()
            if (MODE == "BER"):
                alg.update_bernoulli_count(i, rewards[choice])
            else:
                alg.update_best_count(i, choice, best_bidx)
            alg.update_results(choice, rewards[choice])
            alg.update_step(i)

def main():
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
    main()
