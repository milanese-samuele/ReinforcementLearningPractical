import numpy as np

class Game:
    
    def __init__(self, dim):
        self.board = np.zeros(dim)
    
    ## display spots on the board that have not been marked yet
    def available_moves(self):
        return np.asarray([x for x in self.board.ravel() if x == 0])

    ## main game loop 
    def play(self, players):

        next_player = 0
        while (not self.game_is_over()):
            self.board[players[next_player%2].make_move(available_moves())] = 1 if next_player%2 == 0 else 2
            next_player += 1


