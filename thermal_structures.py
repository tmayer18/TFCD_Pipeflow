# MAE 413 - TFCD
# Project 2
# Timothy Mayer
# 3/25/2022

import numpy as np
from unum_units import units2 as u
import pipe_structures as pipes
from pipe_structures import matrix_expander
from CoolProp.CoolProp import PropsSI

π = np.pi
g = 9.81*u.m/(u.s**2) # [m/s^2]

# Provides a "wrapper" around the pipe_structures to additionally solve for heat transfer

class ThermallyConnected():
    '''Wrapper class for pipes in thermal-contact'''

    def __init__(self, pipeA: pipes.FluidFlow, pipeB: pipes.FluidFlow, wall, same_side_nodes):
        '''initialize an instance of Thermally Connected pipe-likes
        pipeA/B [FluidFlow() obj] : A pipe object that transfers heat to another
        wall [Wall() obj] : a wall object describing the conductive boundary between the two pipes
        same_side_nodes [idx] : tuple of nodes on the same side of the connection'''

        self.pipeA = pipeA
        self.pipeB = pipeB
        self.wall = wall
        self.same_side_nodes = same_side_nodes # TODO split this into the two coupled sides?

        self.inlet_node_a, self.inlet_node_b = same_side_nodes # these aren't actually required to be inlets, the math is reversable
        self.nodes = pipeA.nodes + pipeB.nodes
        other_side_nodes = set(self.nodes) - set(same_side_nodes)
        for n in other_side_nodes:
            if n in pipeA.nodes:
                self.outlet_node_a = n
            elif n in pipeB.nodes:
                self.outlet_node_b = n
            else:
                raise ValueError("Node index values don't match")

    def compute(self, p_n, fluid, N, NUM_STATES=3):
        '''Returns the linear algebra matricies to solve for the next iteration in a pipe

        p_n : solution column vector [p0, p1, p2, ..., ṁ0, ṁ1, ṁ2, ..., T1, T2, ...], at current iteration n, for each node index
        fluid : Dict of fluid properties {name:water, T_ref:215K, p_ref:atm} to pass to CoolProp
        N : total number of nodes, indicates 1/3 number of eqs ie size of matrix
        NUM_STATES : number of fluid properties tracked ie. pressure & massflow & temp = 3

        returns (M,b) to be appended to a full 3Nx3N matrix s.t. M*p_n+1 = b
        '''
        # pass the fluid-flow calculations to the child pipe class
        # TODO fluid properties determined by states
        M_a, b_a = self.pipeA.compute(p_n, fluid, N, NUM_STATES=NUM_STATES)
        M_b, b_b = self.pipeB.compute(p_n, fluid, N, NUM_STATES=NUM_STATES)

        # # pad out missing Temperature entries
        # M_a = matrix_expander(M_a, (2,N*3), (0,1), range(0,N*2))
        # M_b = matrix_expander(M_b, (2,N*3), (0,1), range(0,N*2))

        # generate temperature-solving matrix
        # extract states in readable form
        p_n = np.array(p_n).flatten() # flatten column vector into single list

        p1_a = p_n[self.inlet_node_a]
        p2_a = p_n[self.outlet_node_a]
        p1_b = p_n[self.inlet_node_b]
        p2_b = p_n[self.outlet_node_b]
        ṁ1_a = p_n[self.inlet_node_a + N]
        ṁ2_a = p_n[self.outlet_node_a + N]
        ṁ1_b = p_n[self.inlet_node_b + N]
        ṁ2_b = p_n[self.outlet_node_b + N]
        T1_a = p_n[self.inlet_node_a + 2*N]
        T2_a = p_n[self.outlet_node_a + 2*N]
        T1_b = p_n[self.inlet_node_b + 2*N]
        T2_b = p_n[self.outlet_node_b + 2*N]

        # average quantities
        p̄_a = (p1_a+p2_a)/2
        p̄_b = (p1_b+p2_b)/2
        T̄_a = (T1_a+T2_a)/2
        T̄_b = (T1_b+T2_b)/2

        # ρ = fluid["ρ"]
        ρ_a = PropsSI('DMASS', 'P', p̄_a, 'T', T̄_a, 'water') # [kg/m^3] # TODO pass in fluid name dynamically
        ρ_b = PropsSI('DMASS', 'P', p̄_b, 'T', T̄_b, 'water') # [kg/m^3] # TODO pass in fluid name dynamically
        # TODO units integration to this 
        
        # calculate thermal resistance
        R_a = 0 #1/(h*A)
        R_b = 0#1/(h*A)
        R_wall = 0#scylindrical_thermal_resistance() #wall.resistance
        # TODO handle straight walls too...

        R_tol = R_a + R_b + R_wall

        # calculate heat transfer between pipes, Q
        ΔT_lm = (T1_a - T2_a + T2_b - T1_b)/(np.log((T2_a - T2_b)/(T1_a - T1_b)))
        Q = ΔT_lm / R_tol
        # TODO

        # calculate heat capacity from properties
        Cp = PropsSI('CPMASS', 'P', p̄_a, 'T', T̄_a, 'water') # [J/kg/K]

        # temperature matrices (abstracted into function since we do it twice)
        M_Ta, b_Ta = temp_matrix_assemble(self.pipeA, ρ_a, Cp, ṁ1_a, ṁ2_a, T1_a, T2_a, Q, N)
        M_Tb, b_Tb = temp_matrix_assemble(self.pipeB, ρ_b,Cp, ṁ1_b, ṁ2_b, T1_b, T2_b, Q, N)

        # assemble all matrices together
        M = np.concatenate((M_a, M_Ta, M_b, M_Tb))
        b = np.concatenate((b_a, b_Ta, b_b, b_Tb))

        return M,b

class AdiabaticPipe(): # TODO this name is Pipe?
    '''Wrapper class for insulated pipe'''

    def __init__(self, pipe: pipes.FluidFlow):
        '''initialize an instance of AdiabaticPipe
        pipe [FluidFlow() obj] : a pipe object'''
        
        self.pipe = pipe
        self.nodes = pipe.nodes

        self.inlet_node = pipe.inlet_node
        self.outlet_node = pipe.outlet_node
        self.num_nodes = 2 # TODO inherit this - how do we deal with adibatic tees
    
    def compute(self, p_n, fluid, N, NUM_STATES=2):
        '''Returns the linear algebra matricies to solve for the next iteration in a pipe

        p_n : solution column vector [p0, p1, p2, ..., ṁ0, ṁ1, ṁ2, ..., T1, T2, ...], at current iteration n, for each node index
        fluid : Dict of fluid properties {name:water, T_ref:215K, p_ref:atm} to pass to CoolProp
        N : total number of nodes, indicates 1/3 number of eqs ie size of matrix
        NUM_STATES : number of fluid properties tracked ie. pressure & massflow & temp = 3

        returns (M,b) to be appended to a full 3Nx3N matrix s.t. M*p_n+1 = b
        '''
        # pass the fluid-flow calculations to the child pipe
        M_f, b_f = self.pipe.compute(p_n, fluid, N, NUM_STATES=NUM_STATES)
        # M_f = matrix_expander(M_f, (2,N*3), (0,1), range(0,N*2))

        # generate temperature-solving matrix
        p_n = np.array(p_n).flatten() # flatten column vector to single list

        p1 = p_n[self.inlet_node]
        p2 = p_n[self.outlet_node]
        ṁ1 = p_n[self.inlet_node + N]
        ṁ2 = p_n[self.outlet_node + N]
        T1 = p_n[self.outlet_node + 2*N]
        T2 = p_n[self.outlet_node + 2*N]

        p̄ = (p1+p2)/2 # average pressure for property lookup
        T̄ = (T1+T2)/2

        # TODO T property into fluid array for florwate compute
        ρ = PropsSI('DMASS', 'P', (p̄+fluid['p_ref']).asNumber(u.Pa), 'T', T̄.asNumber(u.K), fluid['name']) * u.kg/(u.m**3) # [kg/m^3] : fluid density
        Cp = PropsSI('CPMASS', 'P', (p̄+fluid['p_ref']).asNumber(u.Pa), 'T', T̄.asNumber(u.K), fluid['name']) * u.J/(u.kg*u.K) # [J/kg*K] : fluid mass specific heat

        # temperature matrices
        M_T, b_T = temp_matrix_assemble(self.pipe, ρ, Cp, ṁ1, ṁ2, T1, T2, 0*u.W, N)

        # assemble matrices together
        M = np.concatenate((M_f, M_T))
        b = np.concatenate((b_f, b_T))

        return M,b


def temp_matrix_assemble(pipe, ρ, Cp, ṁ1, ṁ2, T1, T2, Q, N):
    '''assemble the matrices for temp diff of a single pipe - put into function because we need it twice'''
    A_in = 8/(ρ**2*π**2*(pipe.Do_in**2 - pipe.Di_in**2)**2)
    A_out = 8/(ρ**2*π**2*(pipe.Do_out**2 - pipe.Di_out**2)**2)

    M_T = np.array([
        [Cp*T1 + 3*A_in*ṁ1**2, -(Cp*T2 + 3*A_out*ṁ2**2), Cp*ṁ1, -Cp*ṁ2]
    ])
    b_T = np.array([
        [Q + Cp*T1*ṁ1 - Cp*T2*ṁ2 + 2*A_in*ṁ1**3 - 2*A_out*ṁ2**3 + ṁ1*g*pipe.Δz]
    ])
    
    #expand matrix to columns of [ṁ1, ṁ2, T1, T2]
    M_T = matrix_expander(M_T, (1,3*N), (0,), (pipe.inlet_node+N, pipe.outlet_node+N, pipe.inlet_node+2*N, pipe.outlet_node+2*N) )
    return M_T, b_T

# class ThermalWall():
#     '''parent class for conductive thermal walls through which heat-transfer occurs'''

#     def __init__(self):
#         self.resistance = 0 # [W/m^2]

# ported from schuyler's getR() function
def cylindrical_thermal_resistance(d_outer,d_inner,L,k):
    '''returns the thermal conductive resistance through a cylindrical wall
    d_outer [m] : diameter of outside of wall (larger dimension)
    d_inner [m] : diameter of inside of wall (smaller dimension
    L [m] : Length of of the pipes connected
    k [W/m/K] : Thermal conductivity resistance of wall'''

    r_outer = d_outer/2
    r_inner = d_inner/2 # these two halves would cancel eachother out, but radius is "more correct"

    R = np.log(r_inner/r_outer)/(2*π*L*k) #[K/W]
    return R
    

if __name__ == "__main__":
    # my_tp = ThermallyConnected(
    #     pipes.Pipe(5, 1, 0,1, 0, 0),
    #     pipes.Pipe(5, 2, 2,3, 0, 0),
    #     4, (0,2)
    # )
    my_tp = AdiabaticPipe(
        pipes.Pipe(1*u.m, 2*u.inch, 0,1, 0, 0*u.m)
    )
    p = np.array([1*u.Pa, 1*u.Pa, 1*u.kg/u.s, 1*u.kg/u.s, 300*u.K, 300*u.K])
    print(my_tp.compute(p, {'name':'water', 'T_ref':300*u.K, 'p_ref':u.atm}, 2))