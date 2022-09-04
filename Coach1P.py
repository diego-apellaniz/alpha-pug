import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

#from Arena import Arena
from Arena1P import Arena1P as Arena
from MCTS1P import MCTS1P as MCTS

log = logging.getLogger(__name__)


class Coach1P():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.tempfactor = 10 #daq
        self.boards = {} # so we don't repeat states AND WITH DIFFERENT REWARDS in the training data
        

    def executeEpisode(self, numEps):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            #daq - exploration decay
            if numEps < self.args.tempThreshold:
                temp = self.tempfactor - numEps/self.args.tempThreshold*(self.tempfactor-1) # so it has a lineal variation between self.tempfactor and 1
            else:
                temp = 0

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, _ in sym:
                trainExamples.append(b)
            
            sum_test = np.sum(pi)
            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
            
            r = self.game.getGameEnded(board, self.curPlayer)
            
            if r != 0:
                new_training_data = []
                for b in trainExamples:
                    board_string = self.game.stringRepresentation(b)
                    if board_string not in self.boards or self.boards[board_string] < r:
                        self.boards[board_string] = r
                        new_training_data.append((b,r))
                return new_training_data #board, reward

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for i in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode(i)

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)            

            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)
            # daq -> normalize values
            trainExamples = self.normalizeTrainedExamples(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game, pmcts, nmcts, display = lambda x: self.game.stringRepresentationReadable(x))
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare, verbose = True)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins + draws == 0 or float(nwins+draws) / (pwins + nwins + draws) < self.args.updateThreshold: # we also update with draw, because NN is more updated
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

    # daq -> normalize rewards so they are between 0 and 1
    def normalizeTrainedExamples(self, trainexamples):
        input_boards, target_vs = list(zip(*trainexamples))
        MCTS.min_reward = min(MCTS.min_reward,np.min(target_vs))
        MCTS.max_reward = max(MCTS.max_reward,np.max(target_vs))
        normalized_vs = []
        if MCTS.min_reward == MCTS.max_reward:
            if MCTS.min_reward == 0:
                print("Check Vs values in training data")
            for v in target_vs:
                value = v/MCTS.min_reward
                normalized_vs.appens(np.array(value))
        else:
            dif = MCTS.max_reward - MCTS.min_reward
            for i in range(len(input_boards)):
                value = (np.subtract(target_vs[i],MCTS.min_reward)/dif)[0]
                normalized_vs.append(np.array(value))
        normalized_trained_examples = list(zip(*[input_boards, normalized_vs]))
        return normalized_trained_examples