# MAE 413 - TFCD
# Project 2
# Timothy Mayer
# 3/25/2022

import math
import numpy as np
from unum_units import Unum2
from unum_units import units2 as u
import pipe_structures as pipes
from pipe_structures import matrix_expander
from nusselt_correlations import convection_coefficient_lookup
from CoolProp.CoolProp import PropsSI

# DEBUG
import logging
logger = logging.getLogger(__name__)

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

        self.num_eqs = 6 # number of equations provided, 4 from child pipe flow, 2 from thermal equations

        self.wall.send_dimensions(pipeA, pipeB) # update the wall with the dimensions of the two pipes

    def compute(self, p_n, N, NUM_STATES=3):
        '''Returns the linear algebra matricies to solve for the next iteration in a pipe

        p_n : solution column vector [p0, p1, p2, ..., ṁ0, ṁ1, ṁ2, ..., T1, T2, ...], at current iteration n, for each node index
        N : total number of nodes, indicates 1/3 number of eqs ie size of matrix
        NUM_STATES : number of fluid properties tracked ie. pressure & massflow & temp = 3

        returns (M,b) to be appended to a full 3Nx3N matrix s.t. M*p_n+1 = b
        '''
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

        # pass the fluid-flow calculations to the child pipe class
        self.pipeA.fluid['T_ref'] = T̄_a # replace the placeholder 'reference Temp' with the fluid's actual temp
        self.pipeB.fluid['T_ref'] = T̄_b

        M_a, b_a = self.pipeA.compute(p_n, N, NUM_STATES=NUM_STATES)
        M_b, b_b = self.pipeB.compute(p_n, N, NUM_STATES=NUM_STATES)

        # fluid properties, using coolprop if enabled
        p̄_a_val = (p̄_a+self.pipeA.fluid['p_ref']).asNumber(u.Pa) # we use these unitless-versions in Coolprop lookups a few times. Lets convert to the proper units only once for speed
        p̄_b_val = (p̄_b+self.pipeB.fluid['p_ref']).asNumber(u.Pa)
        T̄_a_val = T̄_a.asNumber(u.K)
        T̄_b_val = T̄_b.asNumber(u.K)

        if self.pipeA.fluid['use_coolprop']:
            ρ_a = PropsSI('DMASS', 'P', p̄_a_val, 'T', T̄_a_val, self.pipeA.fluid['name']) * u.kg/(u.m**3) # [kg/m^3] : fluid density
            μ_a = PropsSI("VISCOSITY", "P", p̄_a_val, 'T', T̄_a_val, self.pipeA.fluid["name"]) * u.Pa*u.s # [Pa*s] : fluid viscosity
            k_a = PropsSI("CONDUCTIVITY", "P", p̄_a_val, 'T', T̄_a_val, self.pipeA.fluid["name"]) * u.W/u.m/u.K # [W/m/K] : fluid conductivity
            Pr_a = PropsSI('PRANDTL', 'P', p̄_a_val, 'T', T̄_a_val, self.pipeA.fluid['name']) # [ul] : Prandtl Number
            Cp_a = PropsSI('CPMASS', 'P', p̄_a_val, 'T', T̄_a_val, self.pipeA.fluid['name']) * u.J/u.kg/u.K # [J/kg/K] : Specific Heat Capacity
        else:
            ρ_a = self.pipeA.fluid['ρ']
            μ_a = self.pipeA.fluid['μ']
            k_a = self.pipeA.fluid['k']
            Pr_a = self.pipeA.fluid['Pr']
            Cp_a = self.pipeA.fluid['Cp']

        if self.pipeB.fluid['use_coolprop']:
            ρ_b = PropsSI('DMASS', 'P', p̄_b_val, 'T', T̄_b_val, self.pipeB.fluid['name']) * u.kg/(u.m**3) # [kg/m^3] : fluid density
            μ_b = PropsSI("VISCOSITY", "P", p̄_b_val, 'T', T̄_b_val, self.pipeB.fluid["name"]) * u.Pa*u.s # [Pa*s] : fluid viscosity
            k_b = PropsSI("CONDUCTIVITY", "P", p̄_b_val, 'T', T̄_b_val, self.pipeB.fluid["name"]) * u.W/u.m/u.K # [W/m/K] : fluid conductivity
            Pr_b = PropsSI('PRANDTL', 'P', p̄_b_val, 'T', T̄_b_val, self.pipeB.fluid['name']) # [ul] : Prandtl Number
            Cp_b = PropsSI('CPMASS', 'P', p̄_b_val, 'T', T̄_b_val, self.pipeB.fluid['name']) * u.J/u.kg/u.K # [J/kg/K] : Specific Heat Capacity
        else:
            ρ_b = self.pipeB.fluid['ρ']
            μ_b = self.pipeB.fluid['μ']
            k_b = self.pipeB.fluid['k']
            Pr_b = self.pipeB.fluid['Pr']
            Cp_b = self.pipeB.fluid['Cp']

        # Reynolds Number
        Dh_a = self.pipeA.Do_in - self.pipeA.Di_in
        Dh_b = self.pipeB.Do_in - self.pipeB.Di_in
        Re_a = abs(4*ṁ1_a/(π*μ_a*Dh_a)).asUnit(u.ul)
        Re_b = abs(4*ṁ1_b/(π*μ_b*Dh_b)).asUnit(u.ul)

        # lookup convection coefficients
        h_a = convection_coefficient_lookup(self.pipeA, Pr_a, Re_a, T̄_b, T̄_a, k_a)
        h_b = convection_coefficient_lookup(self.pipeB, Pr_b, Re_b, T̄_a, T̄_b, k_b)
        # NOTE here we assume the surf. temp of one fluid is the mean temp of the other... not strictly true

        # calculate thermal resistance
        R_a = 1/(h_a*self.wall.areaA)
        R_b = 1/(h_b*self.wall.areaB)
        R_wall = self.wall.resistance

        R_tol = R_a + R_b + R_wall

        # calculate heat transfer between pipes, Q
        try:
            ΔT_lm = (T1_a - T2_a + T2_b - T1_b)/(math.log((T2_a - T2_b)/(T1_a - T1_b)))
        except ZeroDivisionError: # catch vanilla python division error. Using numpy64 floats prevents this usualy                # TODO put zerodivision catches elsewhere in the code
            ΔT_lm = 0*u.K
        if T1_a==T2_a and T1_b==T2_b:
            ΔT_lm = T1_a-T1_b # the uniform T case.
        
        Q = ΔT_lm / R_tol # positive Q is heat out of pipeA -> pipeB
        logger.debug("Heat Transfer Q=%s at ΔT_lm=%s", Q, ΔT_lm)

        # temperature matrices (abstracted into function since we do it twice)
        M_Ta, b_Ta = temp_matrix_assemble(self.pipeA, ρ_a, Cp_a, ṁ1_a, ṁ2_a, T1_a, T2_a, -Q, N)
        M_Tb, b_Tb = temp_matrix_assemble(self.pipeB, ρ_b, Cp_b, ṁ1_b, ṁ2_b, T1_b, T2_b, Q, N)

        # assemble all matrices together
        M = np.concatenate((M_a, M_Ta, M_b, M_Tb))
        b = np.concatenate((b_a, b_Ta, b_b, b_Tb))

        return M,b

class AdiabaticPipe(): # TODO this name is Pipe? does this only work for pipes?
    '''Wrapper class for insulated pipe'''

    def __init__(self, pipe: pipes.FluidFlow):
        '''initialize an instance of AdiabaticPipe
        pipe [FluidFlow() obj] : a pipe object'''
        
        self.pipe = pipe
        self.nodes = pipe.nodes

        self.inlet_node = pipe.inlet_node
        self.outlet_node = pipe.outlet_node
        self.num_eqs = 3 # number of equations provided, 2 from child pipe flow, one from thermal
    
    def compute(self, p_n, N, NUM_STATES=2):
        '''Returns the linear algebra matricies to solve for the next iteration in a pipe

        p_n : solution column vector [p0, p1, p2, ..., ṁ0, ṁ1, ṁ2, ..., T1, T2, ...], at current iteration n, for each node index
        N : total number of nodes, indicates 1/3 number of eqs ie size of matrix
        NUM_STATES : number of fluid properties tracked ie. pressure & massflow & temp = 3

        returns (M,b) to be appended to a full 3Nx3N matrix s.t. M*p_n+1 = b
        '''
        p_n = np.array(p_n).flatten() # flatten column vector to single list

        p1 = p_n[self.inlet_node]
        p2 = p_n[self.outlet_node]
        ṁ1 = p_n[self.inlet_node + N]
        ṁ2 = p_n[self.outlet_node + N]
        T1 = p_n[self.outlet_node + 2*N]
        T2 = p_n[self.outlet_node + 2*N]

        p̄ = (p1+p2)/2 # average pressure for property lookup
        T̄ = (T1+T2)/2

        # pass the fluid-flow calculations to the child pipe
        self.pipe.fluid['T_ref'] = T̄ # set fluid ref-temp to actual temp for property lookup
        M_f, b_f = self.pipe.compute(p_n, N, NUM_STATES=NUM_STATES)

        if self.pipe.fluid['use_coolprop']:
            ρ = PropsSI('DMASS', 'P', (p̄+self.pipe.fluid['p_ref']).asNumber(u.Pa), 'T', T̄.asNumber(u.K), self.pipe.fluid['name']) * u.kg/(u.m**3) # [kg/m^3] : fluid density
            Cp = PropsSI('CPMASS', 'P', (p̄+self.pipe.fluid['p_ref']).asNumber(u.Pa), 'T', T̄.asNumber(u.K), self.pipe.fluid['name']) * u.J/(u.kg*u.K) # [J/kg*K] : fluid mass specific heat
        else:
            ρ = self.pipe.fluid['ρ']
            Cp = self.pipe.fluid['Cp']

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

class ThermalWall():
    '''parent class for conductive thermal walls through which heat-transfer occurs'''

    def __init__(self, k):
        self.k = Unum2.coerceToUnum(k).asUnit(u.W/u.m/u.K) # [W/m/K] : wall thermal conductivity
        self.resistance = 0 # [W/m^2] : conductive thermal resistance
        self.areaA = 0 # [m^2] : heat-transfer area from the pipeA-side-fluid
        self.areaB = 0 # [m^2] : heat-transfer area from the pipeB-side-fluid

    def send_dimensions(self, pipeA, pipeB):
        '''Called when a ThermallyConnected object is created, and passes the dimensional
        information from the two pipes to the wall object, to prevent repeated input data'''
        raise NotImplementedError # this is the base class, please extend this class
        # TODO maybe add some base functionality?

class NestedPipeWall(ThermalWall):
    '''thermal-wall for a pipe contained within another pipe (making it an annulus)'''

    def send_dimensions(self, pipeA, pipeB):
        if isinstance(pipeA, pipes.Annulus):
            annularPipe = pipeA
            rA = pipeA.Di_in/2 # radius of pipe A is inner dimension
            innerPipe = pipeB
            rB = pipeB.Do_in/2
        elif isinstance(pipeB, pipes.Annulus):
            annularPipe = pipeB
            rB = pipeB.Di_in/2
            innerPipe = pipeA
            rA = pipeA.Do_in/2
        assert isinstance(innerPipe, pipes.Pipe) and isinstance(annularPipe, pipes.Annulus), 'NestedPipeWall must contain one pipe and one annulus'

        # t = (annularPipe.Di_in - innerPipe.Do_in)/2 # [m] : thickness of wall
        
        assert annularPipe.L == innerPipe.L, 'The pipes must be of the same length'
        L = annularPipe.L

        self.areaA = 2*π*L*rA
        self.areaB = 2*π*L*rB

            # ported from Scheuyler's getR() function
        self.resistance = (math.log(innerPipe.Do_in/annularPipe.Di_in)/(2*π*L*self.k)).asUnit(u.K/u.W) # [K/W] : conductive resistance
    

if __name__ == "__main__":
    # my_tp = ThermallyConnected(
    #     pipes.Pipe(5, 1, 0,1, 0, 0),
    #     pipes.Pipe(5, 2, 2,3, 0, 0),
    #     4, (0,2)
    # )
    my_tp = AdiabaticPipe(
        pipes.Pipe(1*u.m, 2*u.inch, 0,1, 0, 0*u.m, {'name':'water', 'T_ref':300*u.K, 'p_ref':u.atm})
    )
    p = np.array([1*u.Pa, 1*u.Pa, 1*u.kg/u.s, 1*u.kg/u.s, 300*u.K, 300*u.K])
    print(my_tp.compute(p, 2, NUM_STATES=3))