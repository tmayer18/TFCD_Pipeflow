# MAE 413 - TFCD
# Homework 3
# Timothy Mayer
# 1/21/2022

import numpy as np
π = np.pi
g = 32.174 # [ft/s^2] : Acceleration due to gravity

from colebrook_friction_factor import iterative_solve_colebrook

# Module contains classes for encapsulating data around various pipe-flow components

class Pipe():
    '''Defines a pipe object, containing dimensions and nodal connections'''

    def __init__(self, L, D, inlet_node, outlet_node, ϵD, Δz):
        self.L = L # [ft] : Length of pipe
        self.D = D # [ft] : Diameter of pipe
        self.inlet_node = inlet_node # [index] : location node of inlet of pipe
        self.outlet_node = outlet_node # [index] : location node of outlet of pipe
        self.ϵD  = ϵD # [ul] : Relative Roughness, ϵ/D
        self.Δz = Δz # [ft] : Elevation change, z_out-z_in

    # def friction_factor():
    #     '''Returns the friction factor of the pipe, either from ...'''
    #     # do I even want this? Should I support ṁ and V inputs for this?

    def compute_pipe(self, p_n, fluid, bc):
        '''Returns the linear algebra matricies to solve for the next iteration in a pipe

        pipe : Pipe object, contains dimensional info about the pipe
        p_n : solution column vector [p1, p2, ṁ1, ṁ2], at current iteration n
        fluid : Dict of fluid properties {ρ:val, μ:val}
        bc : Boundary condition values [p_in, p_out] # TODO this may be replaced later

        returns (M,b) s.t. M*p_n+1 = b
        '''
        # translate inputs to easier-to-read form
        p1, p2, ṁ1, ṁ2 = np.array(p_n).flatten() # flatten column vector into iteratable
        ρ = fluid["ρ"]
        μ = fluid["μ"]
        γ = ρ*g
        
        Re = 4*ṁ1/(π*μ*self.D)
        f = iterative_solve_colebrook(self.ϵD, Re)
        COE_coef = -16*f*self.L/(ρ*π**2*self.D**5)

        # form coefficient matrix
        M = np.mat([[1, -1, 0, COE_coef*ṁ2],
                    [0, 0, 1, -1],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0]])
        
        b = np.mat([[γ*self.Δz + COE_coef/2*ṁ2**2, 0, bc[0], bc[1]]]).T
        # TODO this may have to be rewritten to support different boundary conditions for p1 p2

        return M,b

class Tee():
    '''Defines a tee object, containing dimensions and nodal connections'''

    def __init__(self, D, inlet_nodes, outlet_nodes, run_nodes, ϵD, K_run=20, K_branch=60):
        self.D = D # [ft] : Tee Diameter (only supports constant diameter tees)
        # a Tee here must only have 3 legs, either 1_in->2_out or 2_in->1_out
        if type(inlet_nodes) == int: # (idx_1, idx_2) : Nodes of inlets, supports up to 2
            self.inlet_nodes = (inlet_nodes,)
        else:
            self.inlet_nodes = tuple(inlet_nodes)

        if type(outlet_nodes) == int: # (idx_1, idx_2) : Nodes of outlets, supports up to 2
            self.outlet_nodes = (outlet_nodes,)
        else:
            self.outlet_nodes = tuple(outlet_nodes)

        self.run_nodes = tuple(run_nodes) # (idx1, idx2) : Nodes of straight run section of tee
    
        # input validation
        assert len(self.inlet_nodes) + len(self.outlet_nodes) <= 3, "Only 3-port tees are supported"
        assert all([n in self.inlet_nodes+self.outlet_nodes for n in self.run_nodes]), "run_nodes are not inlets or outlets"

if __name__ == "__main__":
    my_pipe = Pipe(6, 0.936/12, 0, 1, 2e-4/(0.936/12), 1)
    A, b = my_pipe.compute_pipe(np.mat([[2304, 2304, 0.1, 0.1]]).T, {"ρ":1.94, "μ":2.34e-5}, (16*144, 14.7*144))
    print(A)
    print(b)
    p = np.linalg.solve(A, b) # solve Ax = b
    print(p)

    A, b = my_pipe.compute_pipe(p, {"ρ":1.94, "μ":2.34e-5}, (16*144, 14.7*144))
    print(A)
    print(b)
    p = np.linalg.solve(A, b) # solve Ax = b
    print(p)
