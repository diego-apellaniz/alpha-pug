
#import gym
#from gym import error, spaces, utils
#from gym.utils import seeding
import numpy as np
import sys
import matplotlib.pyplot as plt
from truss.TrussGame import TrussGame as Game
from truss.pytorch.NNet import NNetWrapper as nn
from MCTS import MCTS
from utils import *

#episodes = 100
#np.set_printoptions(threshold=sys.maxsize)
#env = TrussEnv(True)
#for ep in range(episodes):
#    done = False
#    ep_reward=0
#    while not done:
#        _, reward, done, _ = env.step(env.action_space.sample())
#        ep_reward += reward
#        #env.render()
#    #if ep_reward>0:
#    #    frame = env.render()
#        #figure = plt.figure(figsize=(frame.shape[1] / 72.0, frame.shape[0] / 72.0), dpi=72)
#        #figure.savefig('truss.png')    
#    env.render()   
#    env.reset()    
#    print('episode ', ep, 'score %.1f' % ep_reward)

#env = TrussEnv(optimize = True)
#env.step([9.144,0.5])
#env.step([12.799,3.996+3])
#env.step([18.288,0.5])
#env.render()

args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

np.set_printoptions(threshold=sys.maxsize)
truss = Game()
initial_board = truss.getInitBoard()
print(truss.stringRepresentationReadable(initial_board))

trainExamples = []
board = truss.getInitBoard()
curPlayer = 1
episodeStep = 0
nnet = nn(truss)
mcts =  MCTS(truss, nnet, args)

#while True:
episodeStep += 1
canonicalBoard =truss.getCanonicalForm(board, curPlayer) # just reverse it if we're the other player
temp = int(episodeStep < args.tempThreshold) # 0 if espisodestep>15 -> select best action. Otherwise 1 -> action schosen with normal distribution of probs

pi = mcts.getActionProb(canonicalBoard, temp=temp)
action = np.random.choice(len(pi), p=pi)
board, curPlayer = truss.getNextState(board, curPlayer, action)
pi = mcts.getActionProb(board, temp=temp)
action = np.random.choice(len(pi), p=pi)
r = truss.getGameEnded(board, curPlayer)
if r != 0:
    pass
else:
    pi = mcts.getActionProb(board, temp=temp)
    action = np.random.choice(len(pi), p=pi)
    board, curPlayer = truss.getNextState(board, curPlayer, action)

    r = truss.getGameEnded(board, curPlayer)
    if r != 0:
        pass
    else:
        pi = mcts.getActionProb(board, temp=temp)
        action = np.random.choice(len(pi), p=pi)
        board, curPlayer = truss.getNextState(board, curPlayer, action)
        r = truss.getGameEnded(board, curPlayer)
        if r != 0:
            pass
        else:
            pi = mcts.getActionProb(board, temp=temp)
            action = np.random.choice(len(pi), p=pi)
            board, curPlayer = truss.getNextState(board, curPlayer, action)
            r = truss.getGameEnded(board, curPlayer)
            if r != 0:
                pass
            else:
                pi = mcts.getActionProb(board, temp=temp)
                action = np.random.choice(len(pi), p=pi)
                board, curPlayer = truss.getNextState(board, curPlayer, action)

print(pi)