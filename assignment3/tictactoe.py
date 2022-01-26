#!/usr/bin/env python3
import numpy as np
from game import Game
from random import shuffle
from agent import *
import matplotlib.pyplot as plt

EPS = epsilonQAgent(0.5, 0.9, 0.1)
UCB = UCBQAgent(0.5, 0.9, 3)
GRD = greedyQAgent(0.5, 0.9)
OPT = optimalQAgent(0.5, 0.9)
RND = randomAgent()

firstplayer = EPS
secondplayer = RND

def plots(players, winratios, drawsavg):

    plt.plot(winratios[players[0].name], label=players[0].name)
    plt.plot(winratios[players[1].name], label = players[1].name)
    plt.plot(drawsavg, label = "draws")
    plt.legend()
    plt.show()

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
    all_vs_control()

if __name__ == "__main__" :
    main()