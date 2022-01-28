import numpy as np
import math

class Game:
    
    def __init__(self, dim):
        self.board = np.zeros((dim,dim),dtype='int')
        self.dim = dim
    
    ## display spots on the board that have not been marked yet
    def available_moves(self):
        moves = []
        for rows in range(self.dim):
            for columns in range(self.dim):
                if self.board[rows, columns] == 0:
                    moves.append((rows, columns))
        return moves


    def winner(self):

        # assuming that o is -1 and x is 1 or vica versa
        # horrizontaly check
        sum = 0
        for rows in range(self.dim):
            for columns in range(self.dim):
                sum += self.board[rows,columns]
            if abs(sum) == self.dim:
                return 0 if sum < 0 else 1
            sum = 0

        # vertical check
        sum = 0
        for rows in range(self.dim):
            for columns in range(self.dim):
                sum += self.board[columns, rows]
            if abs(sum) == self.dim:
                return 0 if sum < 0 else 1
            else:
                sum = 0

        # diagonal check
        sum = 0
        sum = np.trace(self.board)
        if abs(sum) == self.dim:
            return 0 if sum < 0 else 1
        sum = np.trace(np.fliplr(self.board))
        if abs(sum) == self.dim:
            return 0 if sum < 0 else 1

        return False


    def end_of_the_game(self):
        if len(self.available_moves()) == 0:
            return True
        else:
            return False

    ## main game loop 
    def play(self, players):

        self.board = np.zeros((self.dim,self.dim),dtype='int')

        w = False

        while (not self.end_of_the_game()):

            for _, p in enumerate(players):
                if (self.end_of_the_game()):
                    break
                p.current_state = str(self.available_moves())
                x, y = p.make_move(self.available_moves())
                self.board[x, y] = -1 if _ == 0 else 1
            w = self.winner() # index of the tuple 0 or 1
            for p in players:
                p.update(str(self.available_moves()), 0)
            if (w is not False): # if theres a winner
                players[w].update(str(self.available_moves()), 100)
                players[abs(w - 1)].update(str(self.available_moves()), -100)

                players[w].update_stats(1)
                players[abs(w - 1)].update_stats(0)
                break
        
        if (w is False):
            for p in players:
                p.update_stats(0)
            return 1 # drawn
        else :
            return 0 # no draw
            

