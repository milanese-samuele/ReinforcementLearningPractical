#!/usr/bin/env python3

from bandit import *
from agent import *
import random

## PARAMETERs FOR EXPERIMENT
N = 1000
K = 20

SEED = 42

def main():
    bandits = [Bandit(random.random() * SEED+idx) for idx in range(K)]
    a = RandomAgent(bandits)
    for _ in range(N):
        print(a.act())

if __name__ == '__main__':
    main()
