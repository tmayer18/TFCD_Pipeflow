# MAE 413 - TFCD
# Project 2
# Timothy Mayer
# 3/25/2022

import numpy as np
import pipe_structures as pipes
import thermal_structures as thermal
import boundary_conditions as bc
from unum_units import units2 as u
import logging

logger = logging.getLogger(__name__)

# provides initialization of solution-vectors

def network_analysis(pipe_network, NUM_STATES):
    '''analyzed an input piping network, checking for appropiate equations '''
    # print some stuff about number of piping elements, and size of M matrix
    N = 0 # number of nodes
    last_node = -1
    pipe_likes = 0
    for elem in pipe_network:
        N += elem.num_eqs
        last_node = max((last_node,)+elem.nodes)
        if type(elem) in [pipes.Minor, pipes.Pipe, pipes.Tee, pipes.Annulus, thermal.AdiabaticPipe]:
            pipe_likes += 1 # count the number of pipe-likes in the network
        if type(elem) in [thermal.ThermallyConnected]:
            pipe_likes += 2

    N = N//NUM_STATES # each state should have one equation per node
    if not last_node+1 == N:
        raise ValueError(f"There aren't enough equations! {N=} but {last_node+1=}")
    logger.info("There are %i pipe-likes in the network, requiring a %ix%i matrix to solve\n", pipe_likes, N*NUM_STATES, N*NUM_STATES)

    return N

def uniform_fluidflow(network, NUM_STATES, ṁ=0.1*u.kg/u.s, p=u.atm):
    '''initializes a uniform pressure and massflow field at each node'''
    if not NUM_STATES==2:
        raise ValueError(f'fluid flow only initializes 2 states: p, ṁ. {NUM_STATES=}')
    N = network_analysis(network, NUM_STATES)
    p_0 = np.concatenate((
        np.ones((N,1))*p,   # TODO these as base units for speed? We chould check that
        np.ones((N,1))*ṁ
    ))
    return p_0, N

def uniform_thermal_fluidflow(network, NUM_STATES, ṁ=0.1*u.kg/u.s, p=u.atm, T=300*u.K):
    '''initialized a uniform temperature, pressure and massflow field at each node'''
    if not NUM_STATES==3:
        raise ValueError(f'thermal fluid flow initializes 3 states: p, ṁ, T. {NUM_STATES=}')
    N = network_analysis(network, NUM_STATES)
    p_0 = np.concatenate((
        np.ones((N,1))*p,
        np.ones((N,1))*ṁ,
        np.ones((N,1))*T
        ))
    return p_0, N

def seq_thermal_unif_fluidflow(network, NUM_STATES, ṁ=0.1*u.kg/u.s, p=u.atm, T=300*u.K):
    '''initialized a uniform pressure and massflow field, and a sequential temperature field at each node'''
    if not NUM_STATES==3:
        raise ValueError(f'thermal fluid flow initializes 3 states: p, ṁ, T. {NUM_STATES=}')
    N = network_analysis(network, NUM_STATES)
    p_0 = np.concatenate((
        np.ones((N,1))*p,
        np.ones((N,1))*ṁ,
        np.ones((N,1))*T +np.arange(0,N).reshape((N,1))*u.K
    ))
    return p_0, N

def normalize_node_numbers(network):
    '''Modifies a network such that the nodes are properly enumerated. Useful for solving parts of large networks without manually renumbering'''

    src_nodes = set()
    for elem in network:
        for n in elem.nodes:
            src_nodes.add(n)
    old_nodes_lookup = dict(enumerate(src_nodes)) #old[2] -> 67
    new_nodes_lookup = dict((v,k) for k,v in old_nodes_lookup.items()) # invert dictionary, new[67] -> 2

    def update_nodes(elem, new_nodes_lookup):
        if isinstance(elem, pipes.FluidFlow):
            elem.inlet_node = new_nodes_lookup[elem.inlet_node]
            elem.outlet_node = new_nodes_lookup[elem.outlet_node]
            elem.nodes = (elem.inlet_node, elem.outlet_node)
        elif isinstance(elem, pipes.Tee):
            elem.inlet_nodes = tuple(new_nodes_lookup[n] for n in elem.inlet_nodes)
            elem.outlet_nodes = tuple(new_nodes_lookup[n] for n in elem.outlet_nodes)
            elem.run_nodes = tuple(new_nodes_lookup[n] for n in elem.run_nodes)
            elem.nodes = elem.inlet_nodes+elem.outlet_nodes
        elif isinstance(elem, thermal.ThermallyConnected):
            update_nodes(elem.pipeA, new_nodes_lookup)
            update_nodes(elem.pipeB, new_nodes_lookup)
            elem.nodes = elem.pipeA.nodes+elem.pipeB.nodes
            elem.same_side_nodes = tuple(new_nodes_lookup[n] for n in elem.same_side_nodes)
            elem.inlet_node_a, elem.inlet_node_b = elem.same_side_nodes
            other_side_nodes = set(elem.nodes) - set(elem.same_side_nodes)
            for n in other_side_nodes:
                if n in elem.pipeA.nodes:
                    elem.outlet_node_a = n
                elif n in elem.pipeB.nodes:
                    elem.outlet_node_b = n
        elif isinstance(elem, thermal.AdiabaticPipe):
            update_nodes(elem.pipe, new_nodes_lookup)
            elem.nodes = elem.pipe.nodes
            elem.inlet_node = elem.pipe.inlet_node
            elem.outlet_node = elem.pipe.outlet_node
        elif isinstance(elem, bc.BoundaryCondition):
            elem.node = new_nodes_lookup[elem.node]
            elem.nodes = (elem.node,)

    for elem in network:
        update_nodes(elem, new_nodes_lookup)

    return network, old_nodes_lookup