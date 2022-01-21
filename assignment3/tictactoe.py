#!/usr/bin/env python3
import numpy as np
from game import Game
from agent import *

def main():
    g = Game(3)
    players = [QAgent(0.4, 0.4), QAgent(0.4, 0.4)]
    g.play(players)

if __name__ == "__main__" :
    main()