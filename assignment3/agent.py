import numpy as np
import random

class randomAgent:

    def make_move(available_moves):
        return random.randint(0, len(available_moves))