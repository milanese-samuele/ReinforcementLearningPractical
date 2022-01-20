#!/usr/bin/env python3
import numpy as np
from game import Game

def main():
    g = Game((3,3))
    print(g.available_moves())    

if __name__ == "__main__" :
    main()