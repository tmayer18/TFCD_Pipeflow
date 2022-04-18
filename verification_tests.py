#%% MAE 413 - TFCD
# Timothy Mayer
# 4/14/2022

# Various verification tests for the pipe-flow solver code
#   constructed with juypter notebook notation for running individual tests

import numpy as np
import logging
from unum_units import Unum2
from unum_units import units2 as u
import pipe_structures as pipes
import boundary_conditions as bc
from iterative_solver import iterative_compute, iter_solution_log
from colebrook_friction_factor import fully_turbulent_f

logging.basicConfig(level=logging.DEBUG)

# %% Single Pipe - Homework 3
# NUM_STATES = 2 # pressure and massflow

# fluid = {
#     "name": "water",
#     "T_ref": (273+4)*u.K, # temperature at which to lookup properties
#     "p_ref": 0, # adjustment from network guage pressures to absolute
# }

# L = 6*u.ft
# D = 0.936*u.inch

# pipe_network = (
#     bc.BoundaryCondition(0, 16*u.psi, 'pressure'),
#     pipes.Pipe(L, D, 0, 1, 2e-4*u.ft/D, 1*u.ft),
#     bc.BoundaryCondition(1, 14.7*u.psi, 'pressure')
# )

# p_n, N = iterative_compute(pipe_network, fluid, 1e-8, 100, NUM_STATES)

# print(' SOLUTION ')
# p_n = Unum2.arr_as_unit(p_n, np.array([[u.psi, u.psi, u.slug/u.s, u.slug/u.s]]).T)
# print(p_n)
# print("Expected ṁ = 0.07119 [slug/s]")

# # %% Single Elbow - Homework 4a
# NUM_STATES = 2 # pressure and massflow

# fluid = {
#     "name": "water",
#     "T_ref": (273+4)*u.K, # temperature at which to lookup properties
#     "p_ref": u.atm, # adjustment from network guage pressures to absolute
# }

# D = 1.049*u.inch
# ft = fully_turbulent_f(2e-4*u.ft/D)

# pipe_network = (
#     bc.BoundaryCondition(0, 0*u.psi, 'pressure'),
#     pipes.Minor(D, D, 0,1, 30*ft),
#     bc.BoundaryCondition(1, -1.5*u.psi, 'pressure')
# )

# p_n, N = iterative_compute(pipe_network, fluid, 1e-8, 30, NUM_STATES)

# print(' SOLUTION ')
# p_n = Unum2.arr_as_unit(p_n, np.array([[u.psi, u.psi, u.slug/u.s, u.slug/u.s]]).T)
# print(p_n)
# print("Expected ṁ = 0.20357 [slug/s]")

# # %% Single Nozzle - Homework 4b
# NUM_STATES = 2 # pressure and massflow

# fluid = {
#     "name": "water",
#     "T_ref": (273+4)*u.K, # temperature at which to lookup properties
#     "p_ref": 0, # adjustment from network guage pressures to absolute
# }


# pipe_network = (
#     bc.BoundaryCondition(0, 14.7*u.psi, 'pressure'),
#     pipes.Minor(4*u.inch, 2*u.inch, 0,1, 0),
#     bc.BoundaryCondition(1, 8*u.psi, 'pressure')
# )

# p_n, N = iterative_compute(pipe_network, fluid, 1e-8, 30, NUM_STATES)

# print(' SOLUTION ')
# p_n = Unum2.arr_as_unit(p_n, np.array([[u.psi, u.psi, u.slug/u.s, u.slug/u.s]]).T)
# print(p_n)
# print("Expected ṁ = 1.3785 [slug/s]")

# %% Simple Network- Homework 4c
NUM_STATES = 2 # pressure and massflow

fluid = {
    "name": "water",
    "T_ref": (273+4)*u.K, # temperature at which to lookup properties
    "p_ref": u.atm, # adjustment from network guage pressures to absolute
}

D = 4*u.inch
ϵD = 4.47e-4*u.ul
ft = fully_turbulent_f(ϵD)

pipe_network = (
    bc.BoundaryCondition(0, 50*u.psi, 'pressure'),
    pipes.Pipe(50*u.ft, D, 0,1, ϵD, 0*u.ft),
    pipes.Minor(D, D, 1,2, 30*ft), # elbow
    pipes.Pipe(10*u.ft, D, 2,3, ϵD, 10*u.ft),
    pipes.Minor(D, D, 3,4, 30*ft), # elbow
    pipes.Minor(D, D, 4,5, 3*ft), # valve
    pipes.Pipe(50*u.ft, D, 5,6, ϵD, 0*u.ft),
    bc.BoundaryCondition(6, 0*u.psi, 'pressure')
)

iter_solution_log.config(pressure_units=u.psf, massflow_units=u.slug/u.s)
p_n, N = iterative_compute(pipe_network, fluid, 1e-8, 30, NUM_STATES)

print(' SOLUTION ')
# p_n = Unum2.arr_as_unit(p_n, np.array([[u.psi, u.psi, u.slug/u.s, u.slug/u.s]]).T)
print(p_n)
print("Expected ṁ = 1.3785 [slug/s]")
# %%
