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
        self.qtable = defaultdict(lambda : defaultdict(lambda : 0.0)) #states are keys, actions as value        
        self.current_state = None
        self.last_action = None

    def make_move(self, available_moves):
        self.last_action = self.get_best_move(available_moves)
        return self.last_action

    def get_best_move(self, available_moves):
        state = str(available_moves)
        actions = self.qtable[state].values()
        if len(actions) is not 0 :
            best_action = max(actions, key=actions.get)
        else:
            best_action = random.choice(available_moves)
            self.qtable[state] = {best_action : 0.0}
        return best_action
        

    def update(self, new_state, reward):
        best_new_action = self.get_best_move(new_state)
        self.qtable[self.current_state][self.last_action] = self.qtable[self.current_state][self.last_action] + self.alpha * (reward + self.gamma* self.qtable[new_state][best_new_action] - self.qtable[self.current_state][self.last_action])
        
    def show_table(self):
        print(self.qtable)