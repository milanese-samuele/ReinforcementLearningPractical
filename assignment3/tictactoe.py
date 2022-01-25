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
EPS2 = epsilonQAgent(0.5, 0.9, 0.1)
UCB2 = UCBQAgent(0.5, 0.9, 3)
GRD2 = greedyQAgent(0.5, 0.9)
OPT2 = optimalQAgent(0.5, 0.9)
RND2 = randomAgent()

firstplayer = EPS
secondplayer = RND

def plots(players, winratios, drawsavg):

    plt.plot(winratios[firstplayer.name], label=firstplayer.name)
    plt.plot(winratios[secondplayer.name], label = secondplayer.name)
    plt.plot(drawsavg, label = "draws")
    plt.legend()
    plt.show()

def main():
    reps = 500
    epochs = 100
    g = Game(3)
    players = [firstplayer, secondplayer]
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
            
    plots(players, winratios, drawsavg)

if __name__ == "__main__" :
    main()