# QLEARNING TICTACTOE

For our final project we decided to look into TicTacToe. This is a really popular and well known game
but we will give a brief introduction to the rules. There are two players who play the game. The board
which the game is played on is a 3x3 grid (in our implementation the size can vary). Each player
chooses one square to put either an ’X’ (player 1) or an ’O’ (player 2). The essence of TicTacToe is
to have three of the same signs (’X’ or ’O’) next to each other horizontally, vertically or diagonally.
The first player to achieve this wins.
In our experiment, we implemented two agents that play the game with different strategies: Q-
learning and Random. Random agent was used as baseline for comparison and learning of other
agents and chooses a random action. On the other hand, Q-learning tries to learn the best strategy
throughout the games. We combined Q learning with different exploration strategies from assignment
one. The implementation contains: Greedy, Epsilon-Greedy, Upper Confidence Bound and Optimal
initialization. We will explain them in more details in the following sections.

## files

**agent.py** contains the implementations of the learning agents; **game.py** contains the framework that implements the actual game of TicTacToe; **tictactoe.py** contains the main function and the experiments setup as well as some utilities for plotting.

## usage 

This code was implemented for obtaining data from some experiments on reinforcement algorithms, for this reason it might not result very user-friendly. However, it can be used without touching any code by running it with some command line arguments. Namely, the user can replicate our two main experiments by giving the program one string as argument: it must be either "ALLVSALL" or "CONTROL".
"ALLVSALL" will replicate the experiment where each algorithm plays against each of the other algorithms and outputs a plot for each of the games. "CONTROL" will replicate the experiment where each algorithm plays against the random agent. It outputs a single plot containing the win ratios of each algorithm.
Note that running the algorithms will take some time.

### example
`$ python ./tictactoe.py ALLVSALL`

`$ python ./tictactoe.py CONTROL`
