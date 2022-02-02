# MAE 413 - TFCD
# Homework 4
# Timothy Mayer
# 1/30/2022

import numpy as np

# Module contains classes for encapsulating data around pipe-network boundary conditions

class BoundaryCondition():
    '''Defines the known conditions at some point within a piping network'''

    def __init__(self, node, value, bc_type):
        self.node = node # [index] : location node this boundary condition applies to
        self.value = value # [psf] : known value at this location TODO other units and BC types
        self.bc_type = bc_type # [str] : the kind of bc this is
        assert bc_type in ["pressure", "mass_flowrate"], f"Boundary condition '{bc_type}' is not supported!"

        self.num_nodes = 1
        self.nodes = (node,)

    def compute(self, p_n, fluid, N):
        '''alias redirect for the compute call
        
        voids the p_n and fluid inputs, as they are unneeded, and forwards N to apply_boundary_condition
        N: total number of nodes, indates 1/2 number of eqs ie size of matrix
        '''
        return self.apply_boundary_condition(N)

    # TODO when we get to implementing more boundary condition types later in the homework schedule, it may be helpful to do a proper inheritance structure here?

    def apply_boundary_condition(self, N):
        '''Returns the linear algebra matricies to solve for the next iteration with this boundary condition

        N : total number of nodes, indicates 1/2 number of eqs ie size of matrix

        returns (M,b) to be appended to a full 2Nx2N matrix s.t. M*p_n+1 = b
        '''
        M = np.zeros((1,2*N))

        if self.bc_type == "pressure":
            M[0,self.node] = 1
        elif self.bc_type == "mass_flowrate":
            M[0,self.node+N] = 1

        b = np.array([[self.value]])

        return M,b
