#!/usr/bin/env python3
import numpy as np
from game import Game
from agent import *
import matplotlibpy.pyplot as plt

def main():
    g = Game(10)
    players = [randomAgent(), QAgent(0.5, 0.9)]
    winners = []
    reps = 100
    epochs = 1000
    wins = 0
    ratios = list()
    for e in range(epochs):
        for i in range(reps):
            wins += g.play(players)
        ratios.append(wins/reps)
    
    plt.plot(ratios)
    plt.show()
    

if __name__ == "__main__" :
    main()