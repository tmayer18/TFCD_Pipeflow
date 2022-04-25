# MAE 413 - TFCD
# Project 2
# Timothy Mayer
# 3/25/2022

import numpy as np
import pipe_structures as pipes
import thermal_structures as thermal
from unum_units import units2 as u
import logging

logger = logging.getLogger(__name__)

# provides initialization of solution-vectors

# TODO we suspect that Temperature will need to be initialized based on their boundary conditions - a uniform starting temp will never change?
# this may be wrong though as at least one temperature in each fluid-loop will eventully propagate out of the specificed boundary conditions? We'll test that

def network_analysis(pipe_network, NUM_STATES):
    '''analyzed an input piping network, checking for appropiate equations '''
    # print some stuff about number of piping elements, and size of M matrix
    N = 0 # number of nodes
    last_node = -1
    pipe_likes = 0
    for elem in pipe_network:
        N += elem.num_nodes
        last_node = max((last_node,)+elem.nodes)
        if type(elem) in [pipes.Minor, pipes.Pipe, pipes.Tee, pipes.Annulus, thermal.AdiabaticPipe]:
            pipe_likes += 1 # count the number of pipe-likes in the network
        if type(elem) in [thermal.ThermallyConnected]:
            pipe_likes += 2

    N = int(N/2) # Each node in the network shows up exactly twice in the network, at the start and end of its pipe-like, or as a boundary condition # FIXME this isn't right anymore
    assert last_node+1 == N, "There aren't enough equations!"
    logger.info("There are %i pipe-likes in the network, requiring a %ix%i matrix to solve\n", pipe_likes, N*NUM_STATES, N*NUM_STATES)

    return N

def uniform_fluidflow(network, NUM_STATES, ṁ=0.1*u.kg/u.s, p=u.atm):
    '''initializes a uniform pressure and massflow field at each node'''
    assert NUM_STATES==2, 'fluid flow only initializes 2 states: p, ṁ'
    N = network_analysis(network, NUM_STATES)
    p_0 = np.concatenate((
        np.ones((N,1))*p,   # TODO these as base units for speed? We chould check that
        np.ones((N,1))*ṁ
    ))
    return p_0, N

def uniform_thermal_fluidflow(network, NUM_STATES, ṁ=0.1*u.kg/u.s, p=u.atm, T=300*u.K):
    '''initialized a uniform temperature, pressure and massflow field at each node'''
    assert NUM_STATES==3, 'thermal fluid flow initializes 3 states: p, ṁ, T'
    N = network_analysis(network, NUM_STATES)
    p_0 = np.concatenate((
        np.ones((N,1))*p,
        np.ones((N,1))*ṁ,
        np.ones((N,1))*T
    ))
    return p_0, N

