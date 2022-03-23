# MAE 413 - TFCD
# Project 2
# Timothy Mayer
# 3/23/2022

import numpy as np
import pipe_structures as pipes
import logging

logger = logging.getLogger(__name__)

# module contains code that iteratites on a matrix-set to solve the state of a pipe-network

def iterative_compute(pipe_network, fluid, desired_tolerance, max_iterations, NUM_STATES):
    '''Iteratively solves for the state of a pipe-network
    pipe_network : List of pipe-structure objects
    fluid : dict of fluid properties
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
    logger.info(f"There are {pipe_likes} pipe-likes in the network, requiring a {N*2}x{N*2} matrix to solve\n")

    #Iterate on the equations
    p_n = 10*np.ones((N*NUM_STATES,1)) # init column solution vector
    # TODO adaptive init for temperatures

    err = 10e2 # init error value
    i = 0 # iteration counter

    while abs(err) > desired_tolerance and i <= max_iterations:
        A = np.empty((0, N*NUM_STATES)) # init empty matrix
        b = np.empty((0,1))
        
        for elem in pipe_network:
            Ai, bi = elem.compute(p_n, fluid, N)
            A = np.append(A, Ai, axis=0) # append matrix-linearized equations to the matrix
            b = np.append(b, bi, axis=0)

        logger.debug(f"CHECK: the length of the matrix is {len(A)}x{len(A[0])}")

        p_n1 = np.linalg.solve(A,b) # solve linear equation Ax=b
        err = max(abs( (p_n-p_n1)/(p_n+1e-16) )) # largest percent change in any solution value

        logger.debug(f"Solution Vector at iteration {i}: {p_n1}")
        logger.info(f"Error at iteration {i}: {err}")

        i+=1
        p_n = p_n1.copy() # p_n = p_n+1
        # copy is necessary cause pointers. Otherwise they will be the same object

    if i >= max_iterations:
        logger.warning("The solution is not converged. Iterations terminated after iteration limit was reached")

    return p_n, N

