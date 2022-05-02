# MAE 413 - TFCD
# Project 2
# Timothy Mayer
# 3/23/2022

import numpy as np
from unum_units import Unum2
from unum_units import units2 as u
import logging
from functools import lru_cache
from tabulate import tabulate

logger = logging.getLogger(__name__)


# module contains code that iteratites on a matrix-set to solve the state of a pipe-network

# TODO i suspect the unit conversions take a lot of time? Analyze that time and see if cacheing something could make calculations faster

def iterative_compute(pipe_network, desired_tolerance, max_iterations, p_0, N, NUM_STATES, relax=1):
    '''Iteratively solves for the state of a pipe-network
    pipe_network : List of pipe-structure objects
    desired_tolerance : consequetive change between iterations to stop iterating at
    max_iterations : cap of iterations to compute
    p_0 : initial solution vector [p1, p2, ... ṁ1, ṁ2, ... T1, T2, ...]
    N : Number of nodes in the network
    NUM_STATES : properties at each node to compute, ie pressure and massflow = 2
    relax : under-relaxation factor. Useful for unstable solution iteration'''

    #Iterate on the equations
    err = 10e2 # init error value
    i = 0 # iteration counter

    p_n = p_0.copy() # copy the initialized solution vector so we can modify it

    while abs(err) > desired_tolerance and i <= max_iterations:
        A = np.empty((0, N*NUM_STATES)) # init empty matrix
        b = np.empty((0,1))
        
        for elem in pipe_network:
            Ai, bi = elem.compute(p_n, N, NUM_STATES=NUM_STATES)
            A = np.append(A, Ai, axis=0) # append matrix-linearized equations to the matrix
            b = np.append(b, bi, axis=0)

        logger.debug("CHECK: the length of the matrix is %ix%i", len(A), len(A[0]))

        A = Unum2.apply_padded_units(A, b, p_n)
        Ainv = Unum2.unit_aware_inv(A)
        p_n1 = Ainv@b # solve linear equation Ax=b

        p_n1_ul, _ = Unum2.strip_units(Unum2.arr_as_base_unit(p_n1)) # to compare percent change, we don't care about units
        p_n_ul, _ = Unum2.strip_units(Unum2.arr_as_base_unit(p_n))
        err = max(abs( (p_n_ul-p_n1_ul)/(p_n_ul+1e-16) )) # largest percent change in any solution value

        logger.debug("Solution Vector at iteration %i: %s", i, Unum2.arr_as_unit(p_n1, iter_solution_log.get_logging_units(N, NUM_STATES)))
        logger.info("Error at iteration %i: %f", i, err)

        i+=1
        p_n = relax*p_n1 + (1-relax)*p_n # p_n = p_n+1, with underrelaxation

    if i >= max_iterations:
        logger.warning("The solution is not converged. Iterations terminated after iteration limit was reached")

    return p_n


# config for units on logged data
class iter_solution_log():
    '''encapsulates the data for what units to log each iteration's solution with. Not really the best use of a OOP, but avoids the global statement and allows caching'''
    # we'll never instantiate an object, so no __init__ is needed

    p_units = u.Pa
    ṁ_units = u.kg/u.s
    T_units = u.K

    @classmethod
    def config(cls, pressure_units=u.Pa, massflow_units=u.kg/u.s, temperature_units=u.K):
        '''sets the units used for the logging of each iteration's solution vector'''
        # pylint: disable=global-statement
        cls.p_units = pressure_units
        cls.ṁ_units = massflow_units
        cls.T_units = temperature_units
        iter_solution_log.get_logging_units.cache_clear()

    @staticmethod
    @lru_cache # cache the result so this doesn't slow iterations down
    def get_logging_units(N, NUM_STATES):
        ret_units = np.ones((N,1))*iter_solution_log.p_units
        ret_units = np.append(ret_units, np.ones((N,1))*iter_solution_log.ṁ_units, axis=0)
        if NUM_STATES >= 3: # if T is present # TODO i don't like this method of specifying units here
            ret_units = np.append(ret_units, np.ones((N,1))*iter_solution_log.T_units, axis=0)
        return ret_units

# results printing
def print_results_table(p, has_temp=False, ṁ_units=u.kg/u.s, p_units=u.Pa, T_units=u.K, rel_temp=False, node_conversions=None):
    table = []
    p = np.array(p).flatten()
    N = len(p)//(2+has_temp)


    for n in range(N): # collect a single node on the table
        entry = []
        if node_conversions:
            entry.append(node_conversions[n])
        else:
            entry.append(n) # node number
        entry.append(p[n+N].asNumber(ṁ_units))
        entry.append(p[n].asNumber(p_units))
        if has_temp:
            entry.append(p[n+2*N].asNumber(T_units))

            if rel_temp: # handle offset units as a special case at print-time
                if T_units == u.K:
                    entry[-1] -= 273.15 # convert K to °C
                elif T_units == u.Rk:
                    entry[-1] -= 459.67 # convert R to °F
                else:
                    raise NotImplementedError
        table.append(entry)

    if rel_temp: # handle offset units as a special case at print-time
        if T_units == u.K:
            class Celsius(): # tiny class to mimic the method of unum2
                def strUnit(self): return "[°C]"
            T_units = Celsius()
        elif T_units == u.Rk:
            class Fahrenheit(): # tiny class to mimic the method of unum2
                def strUnit(self): return "[°F]"
            T_units = Fahrenheit()

    headers = ["Node #", f"ṁ {ṁ_units.strUnit()}", f"p {p_units.strUnit()}"]
    if has_temp:
        headers.append(f"T {T_units.strUnit()}")

    print(tabulate(table, headers))
