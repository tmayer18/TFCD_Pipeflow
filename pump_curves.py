# MAE 413 - TFCD
# Homework 6
# Timothy Mayer
# 2/7/2022

import scipy.interpolate as intp
import numpy as np
from pipe_structures import matrix_expander

# Module contains classes for encapsuling data for pumps in pipe-flow networks

class CappedPCHIP(intp.PchipInterpolator):
    '''Extends pchip interpolator to keep the output within the defined input range'''

    def __init__(self, x, y, extrapolate=False):
        super().__init__(x, y, extrapolate=extrapolate) # define parent object

    def __call__(self, x):
        '''override for call to add edge-case handling'''
        y = super(CappedPCHIP, self).__call__(x) # evaluate the default pchip at x

        if np.isnan(y): # x was outside of known bounds for the pump curve. 
            min_x = min(self.x)
            max_x = max(self.x)
            if x < min_x:
                return self(min_x)
            elif x > max_x:
                return self(max_x)
            else:
                raise ValueError # something has gone wrong in pchip
        return y


class PumpCurve():
    '''Defines the pump-curve that governs a pump, and provides methods to traverse said curve mathematically'''

    def __init__(self, curve_flowrate, curve_pressure_rise, fluid=None, units="QH"):
        '''
        curve_flowrate : listlike of flowrate values
        curve_pressure_rise : listlike of pressure rise values, corresponding to flowrate points
        fluid : dict of fluid properties for the curve {'ρ':#, 'μ':#}
        units : Which units inputs are in. 
            "QH": Volumetric Flowrate [gpm] and Pressure Head [ft]
            "mdp": Mass FLowrate [slug/s] and Pressure Rise [psf]
        '''
        assert units in ["QH", "mdp"], "Invalid units specified"
        if units == "QH":
            self.flowrate = curve_flowrate # TODO conversion factors
            self.head = curve_pressure_rise
        elif units == "mdp":
            self.flowrate = curve_flowrate # [slug/s] : Mass Flowrate
            self.head = curve_pressure_rise # [psf] : Pressure rise across pump

        self.fluid = fluid

        # create a piecewise-cubic-hermetic-interpolating-polynomial function (pchip) of the curve
        self.pchip_curve = CappedPCHIP(self.flowrate, self.head, extrapolate=False)
        self.deriv = self.pchip_curve.derivative()

    def __call__(self, *args): # make object callable as function, like pchip operates
        return self.pchip_curve(*args)


class Pump():
    '''Defines a pump object, containing nodal connectioins and the pump-curve it behaves by'''

    def __init__(self, inlet_node, outlet_node, pump_curve: PumpCurve):
        self.inlet_node = inlet_node # [index] : location node of pipe inlet
        self.outlet_node = outlet_node # [index] : location node of pipe outlet
        self.pump_curve = pump_curve # [PumpCurve object] : defines details of the pump's curve

        self.num_nodes = 2 # number of nodes, ∴ number of eqs
        self.nodes = (inlet_node, outlet_node)

        self.compute = self.compute_pump # redirect compute call to compute_pipe, as an alias

    def compute_pump(self, p_n, fluid, N):
        '''Returns the linear algebra matrices to solve for the next iteration in a pump component
        
        p_n : solution column vector [p0, p1, ..., ṁ0, ṁ1, ...], at current iteration n, for each node index
        fluid : Dict of fluid properties {ρ:val, μ:val}
        N : total number of nodes, indicates 1/2 number of eqs ie size of matrix

        returns (M,b) to be appended to a full 2Nx2N matrix s.t. M*p_n+1 = b
        '''
        # translate inputs to easier-to-read form
        p_n = np.array(p_n).flatten() # flatten column vector into a single list

        # check that the fluid is valid for this curve
        if fluid != None or fluid != self.pump_curve.fluid:
            print("WARNING! compute_pump is being called with a different fluid than the Pump's curve is valid for")

        # p1 = p_n[self.inlet_node]
        # p2 = p_n[self.outlet_node]
        ṁ1 = p_n[self.inlet_node + N]
        # ṁ2 = p_n[self.outlet_node + N]

        # form coefficient matrix
        M = np.array([
            [-1, 1, -self.pump_curve.deriv(ṁ1), 0], # Cons-of-Energy
            [0, 0, -1, 1] # Cons-of-Mass
        ])
        
        b = np.array([
            [self.pump_curve(ṁ1)-self.pump_curve.deriv(ṁ1)*ṁ1], # COE
            [0]
        ])   # COM

        # expand the columns according to what nodes the pipe has
        M = matrix_expander(M, (2,N*2), (0,1), (self.inlet_node, self.outlet_node, N+self.inlet_node, N+self.outlet_node))
        return M,b
    
# if __name__ == "__main__":
#     a = PumpCurve([1,2,3],[0,10,5])
#     print(a)
#     print(a(1))
#     print(a.deriv(1))
#     my_pump = Pump(0,1,a)
#     A,b = my_pump.compute(np.array([0.1,0.1,0.1,0.1]), None, 2)
#     print(A)
#     print(b)