import numpy as np
import random
from collections import defaultdict

class randomAgent:

    def make_move(self, available_moves):
        return random.choice(available_moves)

    def get_reward(self, reward):
        pass

class QAgent:

    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma # discounting factor
        self.qtable = dict()
        self.current_state = None
        self.last_action = None

    def make_move(self, available_moves):
        self.last_action = self.get_best_move(available_moves)
        return self.last_action

    def get_best_move(self, available_moves):
        state = str(available_moves)
        if state in self.qtable.keys():
            actions = self.qtable[state]
            best_action = None
            best_value = -1 
            for a in actions :
                if a.value() > best_value:
                    best_action = a.key()
        else:
            print("before randomchioes", available_moves)
            best_action = random.choice(available_moves)
            print("best action", best_action)
            self.qtable[state] = {best_action : 0.0}
        return best_action
        
    def lookahed_step(self, new_state):
        if new_state in self.qtable.keys():
            actions = self.qtable[new_state]
            best_action = None
            best_value = -1 
            for a in actions :
                if a.value() > best_value:
                    best_value = a.value()
            return best_value
        else:
            return 0.0

    def update(self, new_state, reward):
        print("currentstate", self.current_state)
        best_new_action = self.lookahed_step(new_state)
        self.qtable[self.current_state][self.last_action] = self.qtable[self.current_state][self.last_action] + self.alpha * (reward + self.gamma* best_new_action - self.qtable[self.current_state][self.last_action])
        
    def show_table(self):
        print(self.qtable)