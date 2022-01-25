import numpy as np
import secrets
import random
import math
from collections import defaultdict

class randomAgent:

    def __init__(self):
        self.wincounter = np.empty(shape=0, dtype=int)
        self.name = "Random"

    def make_move(self, available_moves):
        return secrets.choice(available_moves)

    def update(self, new_state, reward):
        pass

    def reset(self):
        self.wincounter = np.empty(shape=0, dtype=int)

    def update_stats(self, result):
        self.wincounter = np.append(self.wincounter, result)

# Q Learning strategy with greedy exploration method
class greedyQAgent:

    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma # discounting factor
        self.qtable = dict()
        self.current_state = None
        self.last_action = None
        self.name = "Greedy Q-learning"
        self.wincounter = np.empty(shape=0, dtype=int)
    
    def reset(self):
        self.__init__(self.alpha, self.gamma)

    def make_move(self, available_moves):
        self.last_action = self.get_best_move(available_moves)
        return self.last_action

    def get_best_move(self, available_moves):
        state = str(available_moves)
        if state in self.qtable.keys():
            actions = self.qtable[state]
            best_action = None
            best_value = -math.inf 
            for a, v in actions.items() :
                if v > best_value:
                    best_action = a
                    best_value = v
        else:
            best_action = secrets.choice(available_moves)
            self.qtable[state] = {}
            for a in available_moves:
                self.qtable[state][a] = 0.0
        return best_action
        
    def lookahed_step(self, new_state):
        if new_state in self.qtable.keys():
            actions = self.qtable[new_state]
            best_action = None
            best_value = -math.inf 
            for a, v in actions.items() :
                if v > best_value:
                    best_action = a
                    best_value = v
            return best_value
        else:
            return 0.0

    def update(self, new_state, reward):
        best_new_action = self.lookahed_step(new_state)
        self.qtable[self.current_state][self.last_action] = self.qtable[self.current_state][self.last_action] + self.alpha * (reward + self.gamma* best_new_action - self.qtable[self.current_state][self.last_action])
        
    def show_table(self):
        print(self.qtable)

    def update_stats(self, result):
        self.wincounter = np.append(self.wincounter, result)

# Q Learning strategy with epsilon-greedy exploration method
class epsilonQAgent:

    def __init__(self, alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma # discounting factor
        self.epsilon = epsilon
        self.qtable = dict()
        self.current_state = None
        self.last_action = None
        self.name = "Epsilon Q-Learning"
        self.wincounter = np.empty(shape=0, dtype=int)
    
    def reset(self):
        self.__init__(self.alpha, self.gamma, self.epsilon)

    def make_move(self, available_moves):
        state = str(available_moves)
        if (random.uniform(0.0, 1.0) > self.epsilon):
            #exploit
            self.last_action = self._get_best_move(available_moves, state)
        else :
            #explore
            self.last_action = self._get_random_move(available_moves, state)
        return self.last_action

    def _get_random_move(self, available_moves, state):
        best_action = secrets.choice(available_moves)
        if state not in self.qtable.keys():
            self.qtable[state] = {}
            for a in available_moves:
                self.qtable[state][a] = 0.0
        return best_action

    def _get_best_move(self, available_moves, state):
        if state in self.qtable.keys():
            actions = self.qtable[state]
            best_action = None
            best_value = -math.inf 
            for a, v in actions.items() :
                if v > best_value:
                    best_action = a
                    best_value = v
        else:
            best_action = self._get_random_move(available_moves, state)
        return best_action
        
    def lookahed_step(self, new_state):
        if new_state in self.qtable.keys():
            actions = self.qtable[new_state]
            best_action = None
            best_value = -math.inf 
            for a, v in actions.items() :
                if v > best_value:
                    best_action = a
                    best_value = v
            return best_value
        else:
            return 0.0

    def update(self, new_state, reward):
        best_new_action = self.lookahed_step(new_state)
        self.qtable[self.current_state][self.last_action] = self.qtable[self.current_state][self.last_action] + self.alpha * (reward + self.gamma* best_new_action - self.qtable[self.current_state][self.last_action])
        
    def show_table(self):
        print(self.qtable)

    def update_stats(self, result):
        self.wincounter = np.append(self.wincounter, result)

# Q Learning strategy with greedy exploration method and optimal initialization
class optimalQAgent:

    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma # discounting factor
        self.qtable = dict()
        self.current_state = None
        self.last_action = None
        self.name = "Optimal Q-Learning"
        self.wincounter = np.empty(shape=0, dtype=int)
        self._optimal_board_init()

    def _optimal_board_init(self):
        # first move
        state = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        self.qtable[str(state)] = {}
        # hardcoding best move
        self.qtable[str(state)][(0,0)] = 1000000.0

    def reset(self):
        self.__init__(self.alpha, self.gamma)

    def make_move(self, available_moves):
        self.last_action = self.get_best_move(available_moves)
        return self.last_action

    def get_best_move(self, available_moves):
        state = str(available_moves)
        if state in self.qtable.keys():
            actions = self.qtable[state]
            best_action = None
            best_value = -math.inf 
            for a, v in actions.items() :
                if v > best_value:
                    best_action = a
                    best_value = v
        else:
            best_action = secrets.choice(available_moves)
            self.qtable[state] = {}
            for a in available_moves:
                self.qtable[state][a] = 0.0
        return best_action
        
    def lookahed_step(self, new_state):
        if new_state in self.qtable.keys():
            actions = self.qtable[new_state]
            best_action = None
            best_value = -math.inf 
            for a, v in actions.items() :
                if v > best_value:
                    best_action = a
                    best_value = v
            return best_value
        else:
            return 0.0

    def update(self, new_state, reward):
        best_new_action = self.lookahed_step(new_state)
        self.qtable[self.current_state][self.last_action] = self.qtable[self.current_state][self.last_action] + self.alpha * (reward + self.gamma* best_new_action - self.qtable[self.current_state][self.last_action])
        
    def show_table(self):
        print(self.qtable)

    def update_stats(self, result):
        self.wincounter = np.append(self.wincounter, result)

# Q Learning strategy with Upper Confidence bound exploration method
class UCBQAgent:

    def __init__(self, alpha, gamma, c):
        self.alpha = alpha
        self.gamma = gamma # discounting factor
        self.c = c
        self.stepcount = 0
        self.qtable = dict()
        self.current_state = None
        self.last_action = None
        self.name = "UCB Q-Learning"
        self.wincounter = np.empty(shape=0, dtype=int)
    
    def reset(self):
        self.__init__(self.alpha, self.gamma, self.c)

    def make_move(self, available_moves):
        state = str(available_moves)
        self.last_action = self._get_best_move(available_moves, state)
        return self.last_action

    def _get_random_move(self, available_moves, state):
        best_action = secrets.choice(available_moves)
        if state not in self.qtable.keys():
            self.qtable[state] = {}
            for a in available_moves:
                self.qtable[state][a] = {"value" : 0.0, "Nt" : 0}
            self.qtable[state][best_action]["Nt"] += 1
        return best_action

    def _get_best_move(self, available_moves, state):
        if state in self.qtable.keys():
            actions = self.qtable[state]
            best_action = None
            best_value = -math.inf 
            for a, v in actions.items() :
                if v["value"] > best_value:
                    best_action = a
                    best_value = v["value"]
            self.qtable[state][best_action]["Nt"] += 1
        else:
            best_action = self._get_random_move(available_moves, state)
        return best_action
        
    def lookahed_step(self, new_state):
        if new_state in self.qtable.keys():
            actions = self.qtable[new_state]
            best_action = None
            best_value = -math.inf 
            for a, v in actions.items() :
                if v["value"] > best_value:
                    best_action = a
                    best_value = v["value"]
            return best_value
        else:
            return 0.0

    def update(self, new_state, reward):
        self.stepcount += 1
        best_new_action = self.lookahed_step(new_state)
        self.qtable[self.current_state][self.last_action]["value"] = self.qtable[self.current_state][self.last_action]["value"] + self.alpha * (reward + self.gamma* best_new_action - self.qtable[self.current_state][self.last_action]["value"])
        self.qtable[self.current_state][self.last_action]["value"] += self.c*math.sqrt((math.log(self.stepcount)/self.qtable[self.current_state][self.last_action]["Nt"]))
        
    def show_table(self):
        print(self.qtable)

    def update_stats(self, result):
        self.wincounter = np.append(self.wincounter, result)
