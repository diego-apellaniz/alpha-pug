import logging
import math

from tqdm import tqdm

log = logging.getLogger(__name__)

### this class is an arena to compare two agents for single-player MDP environments (unlike othello, tic-tac-toe and all other games of the original repository)
class Arena1P():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, mcts1, mcts2, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.mcts1 = mcts1 # daq
        self.mcts2 = mcts2 # daq

    def playGame(self, player, mcts, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """

        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                #probs = mcts.getActionProb(board, temp=1)
                #print(board)
                #print(probs)
                #self.display(board)
            action = player(self.game.getCanonicalForm(board, curPlayer))
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            print(board)
            #self.display(board)
        return curPlayer * self.game.getGameEnded(board, curPlayer)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        max_one = -math.inf
        max_two = -math.inf
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame(self.player1, self.mcts1, verbose=verbose)
            if gameResult > max_one:
                max_one = gameResult


        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame(self.player2, self.mcts2, verbose=verbose)
            if gameResult > max_two:
                max_two = gameResult

        oneWon = max_one>max_two
        twoWon = max_one<max_two
        draws = max_one==max_two

        return oneWon, twoWon, draws