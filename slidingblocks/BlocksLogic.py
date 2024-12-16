from numpy.lib.function_base import i0
import truss.StructPy.cross_sections as xs
import truss.StructPy.structural_classes as sc
import truss.StructPy.materials as ma
import truss.StructPy.Truss as Truss
import numpy as np

class Board():
    
    boundary_conditions_generated = False;
    crosec_list = []
    supports = []
    loads = []
    load_value = -100
    target_columns = []
    pre_target_columns = []
    bars = []
    fail_reward = -100

    def __init__(self, l = 18.288, h =4.572, nx = 4, ny = 4, optimize = False):   
        
        "Set up initial board configuration."

        self.l = l
        self.h = h
        self.nx = nx
        self.ny = ny
        self.optimize = optimize

        # Create the empty board array -> 0
        self.grid_nodes = [None]*(self.ny+1)
        for i in range(self.ny+1):
            self.grid_nodes[i] = [0]*(self.nx+1)        

        # Check if first run
        if not Board.boundary_conditions_generated:
            Board.boundary_conditions_generated = True
            # create list of cross sections for optimization
            Board.crosec_list = Board._create_crosecs_for_optimization()
            # Set up the supports
            Board.supports = [(0,0),(int(self.nx),0)] # index of supports            
            # Set up the loads
            Board.loads = [(int(self.nx/2),0)] # index of the noads where loads are applied
            # check loads and supports
            assert not any(s[1] for s in Board.supports), "all supports should have z=0"
            assert not any(l[1] for l in Board.loads), "all loads should have z=0"
            # Add target columns -> to set the valid nodes in order to avoid instabilities
            for s in Board.supports:
                Board.target_columns.append(s[0])
                if s[0] >0:
                    Board.pre_target_columns.append(s[0]-1)
            for l in Board.loads:
                Board.target_columns.append(l[0])
                if s[0] >0:
                    Board.pre_target_columns.append(l[0]-1)

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.grid_nodes[index]

    def _create_crosecs_for_optimization():
        crosecs = []
        delta_d = 2.5 #cm
        factor_t = 0.04
        for d in [10 + x * delta_d for x in range(47)]:
            crosecs.append(xs.hollowCircle(d/2, d/2*(1-2*factor_t)))
        return crosecs

    def get_steel_mass(nx, ny, l, h, nodes, optimize):
        steel_mass = 0.0 #kg       
        truss, loads = Board.generate_truss(nx, ny, l, h, nodes)
        #truss.plot()
        # calculate structure
        truss.directStiffness(np.array(loads))
        for i in range(len(truss.members)):
            #Board.bars[i].force = truss.members[i].axial
            if not optimize:
                steel_mass += abs(truss.members[i].axial) / truss.members[i].material.Fy * truss.members[i].length * 0.785
            else:
                truss.members[i].optimize(Board.crosec_list)
                if truss.members[i].cross == None:
                    print("Optimization failed. Bigger cross sections required.")
                    #Board.bars[i].diameter = 0
                    return Board.fail_reward, False
                else:
                    #Board.bars[i].diameter = 2*truss.members[i].cross.ro
                    steel_mass += truss.members[i].cross.A * truss.members[i].length * truss.members[i].material.density
        #print("Steel mass = {}".format(steel_mass))
        return steel_mass, True

    def generate_truss(nx, ny, l, h, nodes):
        Aluminium_6063_T5 = ma.Custom(E=6830, fy=17.2, rho = 0.27)
        loads = [] # load vector -> p1x, p1y, p2x, p2y...
        # create truss
        truss = Truss.Truss(cross=Board.crosec_list[0], material=Aluminium_6063_T5, optimize=True)
        # add nodes -> transform from discrete to continuous coordinates
        second_support=False
        for i in range(nx+1):
            if nodes[i]>=0:
                x = i
                y = nodes[i]              
                if (x,y) in Board.supports:
                    if second_support:
                        truss.addNode(x/nx*l, y/ny*h, fixity='roller')
                    else:
                        truss.addNode(x/nx*l, y/ny*h, fixity='pin')
                        second_support = True
                else:
                    truss.addNode(x/nx*l, y/ny*h, fixity='free')
                if (x,y) in Board.loads:
                    loads.append(0)
                    loads.append(Board.load_value)
                else:
                    loads.append(0)
                    loads.append(0)
        # generate bars through triangulation
        for i in range(len(truss.nodes)-2):
            truss.addMember(i, i+1)
            truss.addMember(i, i+2)
        truss.addMember(len(truss.nodes)-2, len(truss.nodes)-1)
        return truss, loads

    def find_last_column(node_locations):
        for x in range(len(node_locations)):
            if node_locations[-x-1]>=0:
                return len(node_locations)-x-1;


    def get_populated_nodes(board, nx):
        # get current column (after last populated one)
        node_location = []
        if np.all((board == 0)):
            last_column = -1
        else:            
            for i in range(nx+1):
                column = board[:,i]
                if np.max(column)>0:
                    loc = np.argmax(column)
                    node_location.append(loc)
                else:
                    node_location.append(-1)
            last_column = Board.find_last_column(node_location)
        current_column = 0
        if last_column>=0:
            current_column = last_column+1
        return current_column, last_column, node_location

    def get_targets(current_column, max_spacing, nx, next_node_top_or_bottom):
        target_columns = []
        j = current_column
        target_row = -1
        if j>nx:
            return target_row, target_columns        
        while True:
            target_columns.append(j)
            if j in Board.target_columns: # columns with loads or supports
                if next_node_top_or_bottom: # if next node ist to be on the top chord, a column with loads or supports can not be considered
                    target_columns.pop()
                    return target_row, target_columns
                target_row = 0
                return target_row, target_columns       
            if j in Board.pre_target_columns and not next_node_top_or_bottom: # if next node ist to be on the bottom chord, a column previous to one with loads or supports can not be considered
                target_columns.pop() # but we don't return yet, because we need to consider target column that comes next
            if j == nx or j - current_column +1 == max_spacing:
                return target_row, target_columns
            j = j+1

    def get_orientation(current_column, last_column, node_location):
        # Get side of two previous nodes
        # previous node above -> 1; previous node below ->-1
        if last_column>0:
            j = last_column
            while True:
                j = j-1
                if node_location[j] >= 0:
                    if node_location[j]>node_location[last_column]:
                        return 1
                    elif node_location[j]<node_location[last_column]:
                        return -1
                    else:
                        return 0
        else:
            return 0

    # will return 0 if the last populated node belongs to the bottom chord and 1 if it belongs to the top chord of the truss
    # basically the first node is a support in the bottom chord and then each new nodes is located in the opposite chord
    def next_node_top_or_bottom(nodes):
        number_of_nodes = np.sum(np.array(nodes) >= 0)
        location = number_of_nodes % 2 
        return location # 0 -> next node bottom; 1 -> next node top
