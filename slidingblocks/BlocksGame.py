import numpy as np
import math
import time

from .TrussLogic import Board

class TrussGame():
    node_content = {
    +0: "•",
    +1: "x",
    }
    max_spacing = 6 # horizontal spacing between consecutive nodes
    boards_record_test = []

    #def __init__(self, l = 18.288, h =4.572, nx = 6, ny = 4, optimize = False, normalize_mass = 50, min_mass = 40):
    def __init__(self, l = 18, h =7, nx = 12, ny = 8, optimize = False, normalize_mass = 50, min_mass = 40): # minimumm n = 4, otherwise error with neural network        
        self.l = l
        self.h = h
        self.nx = nx
        self.ny = ny
        self.optimize = optimize
        self.normalize_mass = normalize_mass
        self.min_mass = min_mass

    def getInitBoard(self):
        b = Board(self.l, self.h, self.nx, self.ny, self.optimize)
        return np.array(b.grid_nodes)

    def stringRepresentation(self, board):
        return board.tostring()
        #return ','.join(str(x) for x in board)

    def stringRepresentationReadable(self, board):
        board_s = []
        for row in board:
            new_row = []
            for node in row:
                new_row.append(TrussGame.node_content[node])
            board_s.append(new_row)
        return np.array(board_s)
        #if np.sum(board)>0:
        #    _, _, node_location = Board.get_populated_nodes(board, self.nx)
        #    truss, _ = Board.generate_truss(self.nx, self.ny, self.l, self.h, node_location);
        #    truss.plot();
        #else:
        #    return "No populated nodes found"


    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return board

    def getBoardSize(self):
        # (a,b) tuple
        return (self.nx+1, self.ny+1)

    def getActionSize(self):
        # return number of actions
        return (self.nx+1)*(self.ny+1)

    def getGameEnded(self, board, player):
        # return 0 if not ended, otherwise reward depending on steel mass of truss structure
        # game if ended if all supports and loads are populated
        # reward has to be negative, because MCTS considers a change of player to interpret the reward
        for support in Board.supports:
            if not board[support[1]][support[0]]:
                return 0
        for load in Board.loads:
            if not board[load[1]][load[0]]:
                return 0
        #calculate reward
        #check that no consecutive nodes have same height - it can happens with consecutive supports or loads
        _, _, node_location = Board.get_populated_nodes(board, self.nx)
        # Calculate reward based on mass of trucc structure
        steel_mass, _ = Board.get_steel_mass(self.nx, self.ny, self.l, self.h, node_location, self.optimize)
        reward = round(10* (1 -(steel_mass-self.min_mass)/self.normalize_mass),3)
        if reward == 0:
            reward = 1e-8 # so it doesn't seem like the game is not finished
        if reward>8:
            breakpoint = "here"
        return np.array([reward]) # so all v values are numpy arrays in the MCTS

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        new_board = np.copy(board).reshape(-1)
        new_board[action] = 1
        new_board = new_board.reshape(self.ny+1, self.nx+1)
        return(new_board, 1)

    def getValidMoves(self, board, player): #board should be a numpy array
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        current_column, last_column, node_location = Board.get_populated_nodes(board, self.nx)
        next_node_top_or_bottom = Board.next_node_top_or_bottom(node_location)
        target_row, target_columns = Board.get_targets(current_column, TrussGame.max_spacing, self.nx, next_node_top_or_bottom)
        orientation = Board.get_orientation(current_column, last_column, node_location)
        # Get valide moves - all nodes in next columns with different row than last node.
        # But they have to be on the same side than the node two steps ago
        for j in target_columns:
            if j == target_columns[-1] and target_row != -1:
                valids[target_row*(self.nx+1)+ j] = 1
            else:
                if orientation >= 0:
                    # select all nodes of that columns above the last one
                    for i in range(node_location[last_column]+1, self.ny+1):                      
                            valids[i*(self.nx+1) + j] = 1
                if orientation <= 0:
                    for i in range(0, node_location[last_column]): # it goes until node_location[last_column]                  
                            valids[i*(self.nx+1) + j] = 1
        valids_test = np.array(valids).reshape(self.ny+1, self.nx+1)
        TrussGame.boards_record_test.append(board)
        return np.array(valids)

    def getSymmetries(self, board, pi):
        ## generally no symmetries
        l = [(board, np.array(pi))]
        return l