# MAE 413 - TFCD
# Project 2
# Timothy Mayer
# 3/23/2022

import numpy as np
import pipe_structures as pipes
from unum_units import Unum2
from unum_units import units2 as u
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# module contains code that iteratites on a matrix-set to solve the state of a pipe-network

# TODO i suspect the unit conversions take a lot of time? Analyze that time and see if cacheing something could make calculations faster

def iterative_compute(pipe_network, fluid, desired_tolerance, max_iterations, NUM_STATES):
    '''Iteratively solves for the state of a pipe-network
    pipe_network : List of pipe-structure objects
    fluid : string of fluid to lookup properties in coolprop
    desired_tolerance : consequetive change between iterations to stop iterating at
    max_iterations : cap of iterations to compute
    NUM_STATES : properties at each node to compute, ie pressure and massflow = 2'''
    # print some stuff about number of piping elements, and size of M matrix
    N = 0 # number of nodes
    last_node = -1
    pipe_likes = 0
    for elem in pipe_network:
        N += elem.num_nodes
        last_node = max((last_node,)+elem.nodes)
        if type(elem) in [pipes.Minor, pipes.Pipe, pipes.Tee, pipes.Annulus]:
            pipe_likes += 1 # count the number of pipe-likes in the network

    N = int(N/2) # Each node in the network shows up exactly twice in the network, at the start and end of its pipe-like, or as a boundary condition
    assert last_node+1 == N, "There aren't enough equations!"
    logger.info("There are %i pipe-likes in the network, requiring a %ix%i matrix to solve\n", pipe_likes, N*2, N*2)

    #Iterate on the equations
    p_n = np.append(
            2304*np.ones((N,1))*u.psf,
            0.1*np.ones((N,1))*u.slug/u.s, # init column solution vector
            axis=0)
    # TODO adaptive init for temperatures
    # that will probably mean the initialization needs to go somewhere else? along with the N counting code?
    # Should the user get some input on the initialization values?

    err = 10e2 # init error value
    i = 0 # iteration counter

    while abs(err) > desired_tolerance and i <= max_iterations:
        A = np.empty((0, N*NUM_STATES)) # init empty matrix
        b = np.empty((0,1))
        
        for elem in pipe_network:
            Ai, bi = elem.compute(p_n, fluid, N)
            A = np.append(A, Ai, axis=0) # append matrix-linearized equations to the matrix
            b = np.append(b, bi, axis=0)

        logger.debug("CHECK: the length of the matrix is %ix%i", len(A), len(A[0]))

        A = Unum2.apply_padded_units(A, b, p_n)
        Ainv = Unum2.unit_aware_inv(A)
        p_n1 = Ainv@b # solve linear equation Ax=b

        p_n1_ul, _ = Unum2.strip_units(p_n1) # to compare percent change, we don't care about units
        p_n_ul, _ = Unum2.strip_units(p_n)
        err = max(abs( (p_n_ul-p_n1_ul)/(p_n_ul+1e-16) )) # largest percent change in any solution value

        logger.debug("Solution Vector at iteration %i: %s", i, Unum2.arr_as_unit(p_n1, iter_solution_log.get_logging_units(N, NUM_STATES))) # TODO logger configure for units
        logger.info("Error at iteration %i: %f", i, err)

        i+=1
        p_n = p_n1.copy() # p_n = p_n+1
        # copy is necessary cause pointers. Otherwise they will be the same object

    if i >= max_iterations:
        logger.warning("The solution is not converged. Iterations terminated after iteration limit was reached")

    return p_n, N


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
    @lru_cache
    def get_logging_units(N, NUM_STATES):
        ret_units = np.ones((N,1))*iter_solution_log.p_units
        ret_units = np.append(ret_units, np.ones((N,1))*iter_solution_log.ṁ_units, axis=0)
        if NUM_STATES >= 3: # if T is present # TODO i don't like this method of specifying units here
            ret_units = np.append(ret_units, np.ones((N,1))*iter_solution_log.T_units, axis=0)
        return ret_units

if __name__ == '__main__':
    print(iter_solution_log.get_logging_units(4, 2))