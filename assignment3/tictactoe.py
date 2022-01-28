#!/usr/bin/env python3
import numpy as np
import sys
from game import Game
from random import shuffle
from agent import *
import matplotlib.pyplot as plt

# function used for plotting results from experiments
def plots(players, winratios, drawsavg):
    plt.plot(winratios[players[0].name], label=players[0].name)
    plt.plot(winratios[players[1].name], label = players[1].name)
    plt.plot(drawsavg, label = "draws")
    plt.legend()
    plt.show()

# setting of the experiment that runs each type of agent against each other
def all_vs_all():
    players = [epsilonQAgent(1, 0.6, 0.1),
                UCBQAgent(1, 0.6, 4),
                greedyQAgent(1, 0.6),
                optimalQAgent(0.4, 0.5)]

    for idx, a in enumerate(players):
        for jdx, o in enumerate(players):
            if (idx == jdx):
                continue
            wins, draws = experiment([a, o], reps=1500, epochs=175)    
            plt.plot(wins[a.name], label=a.name)
            plt.plot(wins[o.name], label=o.name)
            plt.plot(draws, label="draws")
            title = a.name + " against " + o.name
            plt.title(title)
            plt.xlabel("epochs")
            plt.ylabel("wins/games")
            plt.legend()
            plt.show()
    
    

# setting of the experiment that runs each type of agent against the control
# random agent
def all_vs_control():
    
    players = [epsilonQAgent(1, 0.6, 0.1),
                UCBQAgent(1, 0.6, 4),
                greedyQAgent(1, 0.6),
                optimalQAgent(0.4, 0.5)]
    
    control = randomAgent()

    for a in players:
        wins, draws = experiment([a, control], reps=1500, epochs=175)    
        plt.plot(wins[a.name], label=a.name)
    plt.title("Exploration methods against Control Group (Random)")
    plt.xlabel("epochs")
    plt.ylabel("wins/games")
    plt.legend()
    plt.show()

# function used to find the best hyper-parameters
def hp_tuning():

    alphas = [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    gammas = [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

    control = randomAgent()
    greedys = list()

    for a in alphas:
        for g in gammas:
            greedys.append(optimalQAgent(a,g))
    
    for agent in greedys:
        wins, draws = experiment([agent, control], reps=100, epochs=150)
        print(agent.alpha, agent.gamma, ": ", np.average(wins[agent.name]), np.average(draws))
    
## core function that runs an experiment given two players, a number of repetitions
# epochs and the dimension of the grid
def experiment(players, reps = 500, epochs = 100, dim = 3):

    g = Game(dim)

    winratios = {players[0].name : np.zeros(shape=epochs),
                players[1].name : np.zeros(shape=epochs)}
    drawsavg = np.zeros(shape=epochs)

    for rep in range(reps):
        drawscnt = np.empty(shape= 0)
        for p in players:
            p.reset()

        for i in range(epochs):
            shuffle(players)            
            draw = g.play(players)
            drawscnt = np.append(drawscnt, draw)

        for p in players:
            winratios[p.name] = winratios[p.name] + (p.wincounter - winratios[p.name]) / (rep + 1)
        drawsavg = drawsavg + (drawscnt - drawsavg) / (rep + 1)
            
    return winratios, drawsavg

def main():
    # players = [epsilonQAgent(1.0, 0.6, 1), greedyQAgent(1.0, 0.6)]
    # wins, draws = experiment(players, reps=1000, epochs=175)
    # plots(players, wins, draws)
    if (sys.argv[1] == "ALLVSALL") :
        all_vs_all()
    elif (sys.argv[1] == "CONTROL") :
        all_vs_control()
    else :
        print ("wrong input, try again!")

if __name__ == "__main__" :
    main()