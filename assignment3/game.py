import numpy as np
import math

class Game:
    
    def __init__(self, dim):
        self.board = np.zeros((dim,dim),dtype='int')
        self.dim = dim
    
    ## display spots on the board that have not been marked yet
    def available_moves(self):
        return np.asarray([x for x in self.board.ravel() if x == 0])
        # return np.argwhere(self.board == 0)


    ## main game loop 
    def play(self, players):

        next_player = 0
        while (not self.game_is_over()):
            self.board[players[next_player%2].make_move(available_moves())] = 1 if next_player%2 == 0 else 2
            next_player += 1

    def winner(self):

        # assuming that o is -1 and x is 1 or vica versa
        # horrizontaly check
        sum = 0
        for rows in range(self.dim):
            for columns in range(self.dim):
                sum += self.board[rows,columns]
            if abs(sum) == self.dim:
                return #something
            else:
                sum = 0

        # vertical check
        sum = 0
        for rows in range(self.dim):
            for columns in range(self.dim):
                sum += self.board[rows, columns]
            if abs(sum) == self.dim:
                return #something
            else:
                sum = 0

        # diagonal check
        if np.trace(self.board) == self.dim or np.trace(np.fliplr(self.board)) == self.dim:
            return # somehting


    def end_of_the_game(self):
        if not np.argwhere(self.board == 0):
            return True
        else:
            return False