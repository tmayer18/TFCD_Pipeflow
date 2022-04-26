# MAE 413 - TFCD
# Homework 4
# Timothy Mayer
# 1/30/2022

import numpy as np
from unum_units import units2 as u

# Module contains classes for encapsulating data around pipe-network boundary conditions

class BoundaryCondition():
    '''Defines the known conditions at some point within a piping network'''

    def __init__(self, node, value, bc_type):
        self.node = node # [index] : location node this boundary condition applies to
        self.value = value # [some-units] : known value at this location
        self.bc_type = bc_type # [str] : the kind of bc this is
        assert bc_type in ["pressure", "mass_flowrate", "temperature"], f"Boundary condition '{bc_type}' is not supported!"

        self.num_eqs = 1
        self.nodes = (node,)

    # pylint: disable=unused-argument
    def compute(self, p_n, N, NUM_STATES=2):
        '''alias redirect for the compute call
        
        voids the p_n unneeded input, and forwards N to apply_boundary_condition
        N: total number of nodes, indates 1/2 number of eqs ie size of matrix
        NUM_STATES : number of fluid properties tracked ie. pressure & massflow = 2
        '''
        return self.apply_boundary_condition(N, NUM_STATES)

    def apply_boundary_condition(self, N, NUM_STATES):
        '''Returns the linear algebra matricies to solve for the next iteration with this boundary condition

        N : total number of nodes, indicates 1/2 number of eqs ie size of matrix
        NUM_STATES : number of fluid properties tracked ie. pressure & massflow = 2

        returns (M,b) to be appended to a full 2Nx2N matrix s.t. M*p_n+1 = b
        '''
        M = np.zeros((1,N*NUM_STATES), dtype=object)

        if self.bc_type == "pressure":
            M[0,self.node] = 1*u.ul     # due to numpy vectorization weirdness, this needs to be a Unum object
        elif self.bc_type == "mass_flowrate":
            M[0,self.node+N] = 1*u.ul
        elif self.bc_type == "temperature":
            M[0,self.node+2*N] = 1*u.ul

        b = np.array([[self.value]])

        return M,b
