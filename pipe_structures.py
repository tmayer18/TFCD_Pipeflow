# MAE 413 - TFCD
# Homework 4
# Timothy Mayer
# 1/31/2022

import numpy as np
π = np.pi
g = 9.82 # [m/s^2] : Acceleration due to gravity

from colebrook_friction_factor import fully_turbulent_f, iterative_solve_colebrook

# Module contains classes for encapsulating data around various pipe-flow components

class FluidFlow():
    '''Parent class for pipes/elbows/annulus ect...
    Contains momentum and mass conservation equations solving for pressure and massflow'''

    def __init__(self, Di_in, Do_in, Di_out, Do_out, inlet_node, outlet_node, loss, L, ϵD, K, Δz):
        '''initialize an instance of FlowFlow
        Di_in [m] : Inlet annulus inner diameter
        Do_in [m] : Inlet annulus outer diameter
        Di_out [m] : Outlet annulus inner diameter
        Do_out [m] : Outlet annulus outer diameter
        inlet_node [idx] : location node of the inlet
        outlet_noew [idx] : location node of the outlet
        loss [str] : type of losses - 'major' or 'minor'
        L [m] : Length of pipe - used in major loss calculation
        ϵD [ul] : Relative roughness - used in major loss calculation
        K [ul] : Loss Coefficient, typically K=c*ft
        Δz [m] : elevation change, z_out-z_in'''
        
        self.Di_in = Di_in # [m] : diameter
        self.Do_in = Do_in # [m]
        self.Di_out = Di_out # [m]
        self.Do_out = Do_out # [m]

        self.inlet_node = inlet_node # [index]
        self.outlet_node = outlet_node # [index]
        self.num_nodes = 2 # number of nodes, ∴ number of eqs
        self.nodes = (inlet_node, outlet_node)

        assert loss in ["major", "minor"], "loss-type must be 'major' or 'minor'"
        self.loss = loss

        self.L = L # [m] : length
        self.ϵD = ϵD # [ul] : relative roughness
        self.K = K # [ul] : loss coefficient, tpically K=ft*C
        self.Δz = Δz # [m] : elevation change

        self.compute = self.compute_flow # alias redirect for the compute call

    def compute_flow(self, p_n, fluid, N):
        '''Returns the linear algebra matricies to solve for the next iteration in a pipe

        p_n : solution column vector [p0, p1, p2, ..., ṁ0, ṁ1, ṁ2, ...], at current iteration n, for each node index
        fluid : Dict of fluid properties {ρ:val, μ:val}
        N : total number of nodes, indicates 1/2 number of eqs ie size of matrix

        returns (M,b) to be appended to a full 2Nx2N matrix s.t. M*p_n+1 = b
        '''
        # translate inputs to easier-to-reference form
        p_n = np.array(p_n).flatten() # flatten column vector into single list

        p1 = p_n[self.inlet_node]
        p2 = p_n[self.outlet_node]
        ṁ1 = p_n[self.inlet_node + N]
        ṁ2 = p_n[self.outlet_node + N]

        # fluid properties extracted from the dict
        ρ = fluid["ρ"]
        μ = fluid["μ"]
        γ = ρ*g
        
        Dh = self.Do_in - self.Di_in # hydraulic diameter - calculated at the inlet side

        # calculating the head loss
        if self.loss == "major":
            # calculating the friction factor
            Re = 4*ṁ1/(π*μ*Dh) # TODO Re and ϵD is based on hydraulic diameter here
            f = iterative_solve_colebrook(self.ϵD, Re)
            loss_coef = f*self.L/Dh
        elif self.loss == "minor":
            loss_coef = self.K

        # coefficient terms
        A_in = 16/(ρ*π**2*(self.Do_in**2 - self.Di_in**2)**2)
        A_out = 16/(ρ*π**2*(self.Do_out**2 - self.Di_out**2)**2)

        # form coefficient matrix
        M = np.array([
            [1,   -1,     A_in*ṁ1,    -(loss_coef +1)*A_out*ṁ2], # Cons-of-Energy
            [0, 0, 1, -1]])          # Cons-of-Mass
        b = np.array([
            [γ*self.Δz + A_in/2*ṁ1**2 -(loss_coef +1)*A_out/2*ṁ2**2], # COE
            [0]])   # COM

        # expand the columns according to what nodes the pipe has
        M = matrix_expander(M, (2,N*2), (0,1), (self.inlet_node, self.outlet_node, N+self.inlet_node, N+self.outlet_node))
        return M,b

class Pipe(FluidFlow):
    '''Defines and solves flow in a pipe object, containing dimensions and nodal connections'''

    def __init__(self, L, D, inlet_node, outlet_node, ϵD, Δz):
        '''initialize an instance of Pipe()
        L [m] : Length of Pipe
        D [m] : Diameter of Pipe
        inlet_node [index] : location node of pipe inlet
        outlet_node [index] : location node of pipe outlet
        ϵD [ul] : Relative Roughness, ϵ/D
        Δz [m] : Elevation change, z_out-z_in'''

        # in a pipe, outer diameter is constant, annular inner diameter is zero
        super().__init__(0, D, 0, D, inlet_node, outlet_node, "major", L, ϵD, 0, Δz)
        
class Annulus(FluidFlow):
    '''Defines and solves flow in an annular pipe pbject, containing dimensions and nodal connections'''

    def __init__(self, L, Di, Do, inlet_node, outlet_node, ϵD, Δz):
        '''initialize an instance of Annulus()
        L [m] : Length of annular pipe
        Di [m] : Inner diameter of annulus
        Do [m] : Outer diameter of annulus
        inlet_node [index] : location node of the annulus output
        outlet_node [index] : location node of annular output
        ϵD [ul] : Relative Roughness, ϵ/D
        Δz [m] : Elevation change, z_out-z_in'''

        super().__init__(Di, Do, Di, Do, inlet_node, outlet_node, "major", L, ϵD, 0, Δz)

class Minor(FluidFlow):
    '''Defines a minor-loss object, (ex elbow, nozzle, ect...), containing dimensions and nodal connections'''

    def __init__(self, Di, Do, inlet_node, outlet_node, K):
        '''initialize an instance of Minor()
        Di [m] : Inlet Diameter
        Do [m] : Outlet Diameter
        inlet_node [index] : location node of inlet
        outlet_node [index] : location node of outlet
        K [ul] : Loss Coefficient, typically K=c*ft'''

        super().__init__(0, Di, 0, Do, inlet_node, outlet_node, "minor", 0, 0, K, 0)

class Tee():
    '''Defines a tee object, containing dimensions and nodal connections'''

    def __init__(self, D, inlet_nodes, outlet_nodes, run_nodes, ϵD, C_run=20, C_branch=60):
        '''initialize an instance of Tee
        D [m] : Tee Diameter (only constant diameter tees supported)
        inlet_nodes (idx, idx) : Up to 2 inlet node locations
        outlet_nodes (idx, idx) : Up to 2 outlet node locations
        run_nodes (idx, idx) : Which 2 nodes form the run of the tee
        ϵD [ul] : Relative roughness'''
        
        self.D = D # [ft] : Tee Diameter (only supports constant diameter tees)
        ft = fully_turbulent_f(ϵD)
        K_run = C_run*ft
        K_branch = C_branch*ft

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
        if len(self.outlet_nodes) == 2:
            self.configuration = "dual_outlet"
        else:
            self.configuration = "dual_inlet"

        # input validation
        assert len(self.inlet_nodes) + len(self.outlet_nodes) <= 3, "Only 3-port tees are supported"
        assert all([n in self.inlet_nodes+self.outlet_nodes for n in self.run_nodes]), "run_nodes are not inlets or outlets"

        # determine which loss coefficients go where in compute
        if (self.inlet_nodes[0] in run_nodes) and (self.outlet_nodes[0] in run_nodes):
            # if the first inlet and first outlet are in a run
            self.K1 = K_run
        else:
            self.K1 = K_branch

        if (self.inlet_nodes[-1] in run_nodes) and (self.outlet_nodes[-1] in run_nodes):
            # if the second inlet and second outlet are in a run (if there is no second, use the first)
            self.K1 = K_run
        else:
            self.K2 = K_branch

        self.num_nodes = 3 # number of nodes, ∴ number of eqs
        self.nodes = self.inlet_nodes+self.outlet_nodes

        self.compute = self.compute_tee # redirect alias for compute -> compute_tee

    def compute_tee(self, p_n, fluid, N):
        '''Returns the linear algebra matricies to solve for the next iteration in a minor-loss component

        p_n : solution column vector [p0, p1, p2, ..., ṁ0, ṁ1, ṁ2, ...], at current iteration n, for each node index
        fluid : Dict of fluid properties {ρ:val, μ:val}
        N : total number of nodes, indicates 1/2 number of eqs ie size of matrix

        returns (M,b) to be appended to a full 2Nx2N matrix s.t. M*p_n+1 = b
        '''
        # translate inputs to easier-to-read form
        p_n = np.array(p_n).flatten() # flatten column vector into single list

        # p_in1 = p_n[self.inlet_nodes[0]] # pressure terms unused
        # p_in2 = p_n[self.inlet_nodes[-1]]
        # p_out1 = p_n[self.outlet_nodes[0]]
        # p_out2 = p_n[self.outlet_nodes[-1]]
        ṁ_in1 = p_n[self.inlet_nodes[0] + N]
        ṁ_in2 = p_n[self.inlet_nodes[-1] + N]
        ṁ_out1 = p_n[self.outlet_nodes[0] + N]
        ṁ_out2 = p_n[self.outlet_nodes[-1] + N]
        # in1 and in2 will be the same for single-inlet systems, and vice versa for single-outlet
        
        ṁ_in = ṁ_in1 # ease of access alias
        ṁ_out = ṁ_out1

        ρ = fluid["ρ"]
        K1 = self.K1 # ease of access
        K2 = self.K2

        # coefficient constant
        C = 8/(ρ*π**2*self.D**4)

        # form coefficient matricies
        if self.configuration == "dual_outlet":
            M = np.array([
                [1, -1,  0,   2*C*ṁ_in,   -2*C*(K1+1)*ṁ_out1,           0], # COE inlet->outlet1
                [1,  0, -1,   2*C*ṁ_in,         0,            -2*C*(K2+1)*ṁ_out2], # COE inlet->outlet2
                [0, 0, 0, 1, -1, -1] # COM
            ])

            b = np.array([
                [C*(ṁ_in**2-(K1+1)*ṁ_out1**2)],
                [C*(ṁ_in**2-(K2+1)*ṁ_out2**2)],
                [0]
            ])

        elif self.configuration == "dual_inlet":
            M = np.array([
                [1,  0, -1,   2*C*(1-K1)*ṁ_in1,         0,           -2*C*ṁ_out], # COE inlet1->outlet
                [0,  1, -1,          0,         2*C*(1-K2)*ṁ_in2,    -2*C*ṁ_out], # COE inlet2->outlet
                [0, 0, 0, 1, 1, -1] # COM
            ])

            b = np.array([
                [C*((1-K1)*ṁ_in1**2-ṁ_out**2)],
                [C*((1-K2)*ṁ_in2**2-ṁ_out**2)],
                [0]
            ])
        else:
            raise Exception("Tee Object failure! Was neither dual_inlet nor dual_outlet")

        # expand matrix to full NxN size
        all_nodes = self.nodes
        M = matrix_expander(M, (3,N*2), (0,1,2), all_nodes+tuple(map(lambda x:x+N, all_nodes)))

        return M,b


def matrix_expander(A, NxM, row:tuple, col:tuple=(0,)):
    '''Expands matrix so that it's original elements lie within the rows and columns of a NxN matrix
    
    A : Numpy matrix or ndarray to expand
    NxM : Desired NxM size of result, tuple (N,M)
    row : tuple of row indices to fill
    col : tuple of col indices to fill
    
    ex: [[A,B],[C,D]], (4,3), (1,3), (1,2) -> [[0,0,0],[0,A,B],[0,0,0],[0,C,D]]'''

    A_row = len(A)
    A_col = len(A[0])
    N,M = NxM # unpack size

    I_row_expander = np.zeros((N,A_row))
    I_col_expander = np.zeros((A_col,M))

    # form modified identity matrices
    for (c,r) in zip(range(A_row), row):
        I_row_expander[r,c] = 1
    for (r,c) in zip(range(A_col), col):
        I_col_expander[r,c] = 1

    # matrix-multiply (with numpy @ operator) the matrices together
    return I_row_expander @ A @ I_col_expander


if __name__ == "__main__":
    # print(matrix_expander([[1,2],[3,4]], (4,3), (1,3), (1,2)))

    # my_pipe = Pipe(6, 0.936/12, 1, 2, 2e-4/(0.936/12), 1)
    # p_n = np.array([[2304, 2304, 2304, 2304, 0.1, 0.1, 0.1, 0.1]]).T
    # A, b = my_pipe.compute(p_n, {"ρ":1.94, "μ":2.34e-5}, 4)
    # print(A)
    # print(b)
    # p = np.linalg.solve(A, b) # solve Ax = b
    # print(p)

    # A, b = my_pipe.compute_pipe(p, {"ρ":1.94, "μ":2.34e-5}, (16*144, 14.7*144))
    # print(A)
    # print(b)
    # p = np.linalg.solve(A, b) # solve Ax = b
    # print(p)

    # my_tee = Tee(6, (0,1), 2, (0,2), 1)
    pass
