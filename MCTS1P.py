import logging
import math

import numpy as np
from numpy.core.fromnumeric import nonzero
from torch.functional import norm

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS1P():
    """
    This class handles the MCTS tree.
    """
    
    Es = {}  # stores game.getGameEnded ended for board s
    Vs = {}  # stores game.getValidMoves for board s
    min_reward = float('inf')
    max_reward = -float('inf')

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        #self.Ps = {}  # stores initial policy (returned by neural net) - not any more! just Q values are returned from the NN      

        #daq
        self.boardsaux = []
        self.boards = []
        self.qs2 = []
        self.Qsn = {}
        self.qs_for_new_nodes = 0

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        # the probabilities will be based just on the Q values, not on the number of visists
        # the number of visits just penalize the nodes that have been explored several times for new explorations
        # for 1P environments, the number of times that it a nodes leads to victory is irrelevant, the goal is to find the
        # node that leads to the best reward
        #counts_prev  = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        counts = [self.Qsa[(s, a)] if (s, a) in self.Qsa else 0 for a in range(self.game.getActionSize())] 

        ## normalize vector
        max_value = np.max(counts)
        min_value = np.min(counts)
        normalized_counts = []
        if max_value != 0:
            dif = max_value - min_value            
            for x in counts:
                if x == 0:
                    normalized_counts.append(0)
                else:
                    value = (np.subtract(x,min_value)/dif)
                    normalized_counts.append(value)            
        else:
            for x in counts:
                if x == 0:
                    normalized_counts.append(0)
                else:
                    value = (x/min_value)
                    normalized_counts.append(value)
        counts = normalized_counts

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs) # choose randomly one of the actions with greater number of visits
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = np.array([x ** (1. / temp) for x in counts]).astype('float64')
        #counts = np.array([x ** (1. / temp) for x in counts])
        counts_sum = float(sum(counts))
        if counts_sum == 0:
            stp = "here"
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found (or a terminal node). The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state (just a value v for 1P). This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in MCTS1P.Es:
            MCTS1P.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
            MCTS1P.max_reward = max(MCTS1P.max_reward, MCTS1P.Es[s])
            MCTS1P.min_reward = min(MCTS1P.min_reward, MCTS1P.Es[s])
        if MCTS1P.Es[s] != 0:
            # terminal node            
            return True, MCTS1P.Es[s] #1P - True because terminal node

        if s not in self.Ns:
            # leaf node
            v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)

            if s not in MCTS1P.Vs:
                MCTS1P.Vs[s] = valids
            self.Ns[s] = 0
            return False, v #1P - False, because not terminal node

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound

        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u_1 = self.Qsa[(s, a)]
                    u_2 = self.args.cpuct * 1 * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                    u = u_1 + u_2

                else:
                    u_1 = self.qs_for_new_nodes
                    u_2 = self.args.cpuct * 1 * math.sqrt(self.Ns[s] + EPS)
                    u = u_1 + u_2

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        terminal, v = self.search(next_s)

        # normalize reward - if several rewards have been determined in this episode, it might not be accurate for the previous ones, but next episode will be better
        if terminal:
            if MCTS1P.max_reward == MCTS1P.min_reward:
                v = (v/MCTS1P.min_reward)[0]
            else:        
                dif = np.subtract(MCTS1P.max_reward, MCTS1P.min_reward)
                v = (np.subtract(v,MCTS1P.min_reward)/dif)[0]

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = max(self.Qsa[(s, a)], v) # the value of the node is the value of the best possible result downstream
            self.Nsa[(s, a)] += 1

            ##daq
            #index = self.boardsaux.index(self.game.stringRepresentation(next_s))
            #self.qs2[index] = self.Qsa[(s, a)]

        else:
            self.Qsa[(s, a)] = v + EPS
            self.Nsa[(s, a)] = 1
            #self.normalizeQs()

            ##daq
            #self.boardsaux.append(self.game.stringRepresentation(next_s))
            #self.boards.append(self.game.stringRepresentationReadable(next_s))
            #self.qs2.append(self.Qsa[(s, a)])

        self.Ns[s] += 1
        return False, v #1P

    #def normalizeQs(self):
    #    qs = self.Qsa.values()
    #    max_q = max(qs)
    #    min_q = min(qs)
    #    # reset dictionary and update values
    #    self.Qsn = {}
    #    if max_q == min_q:
    #        for s_a_pair in self.Qsa:
    #            self.Qsn[s_a_pair] = (self.Qsa[s_a_pair]/min_q)[0]
    #    else:        
    #        dif = np.subtract(max_q, min_q)
    #        for s_a_pair in self.Qsa:
    #            self.Qsn[s_a_pair] = (np.subtract(self.Qsa[s_a_pair],min_q)/dif)[0]
        