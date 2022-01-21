#!/usr/bin/env python3
import numpy as np
from game import Game
from agent import *
import matplotlib.pyplot as plt

def main():
    g = Game(3)
    players = [QAgent(0.5, 0.9), randomAgent()]
    winners = []
    reps = 100
    epochs = 1000
    ratios = list()
    for e in range(epochs):
        wins = 0
        for i in range(reps):
            result = g.play(players)
            if result is not None:
                wins += result
        ratios.append(wins/reps)

    print(ratios)
    plt.plot(ratios)
    plt.show()
    

if __name__ == "__main__" :
    main()