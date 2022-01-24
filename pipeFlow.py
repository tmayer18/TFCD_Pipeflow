# MAE 413 - TFCD
# Homework 3
# Timothy Mayer
# 1/23/2022

# Will compute the pressure [psf] and flowrate [slug/s] at the ends of a pipe

import numpy as np
from pipe_structures import *

# %% Inputs
D = 0.936/12 # [ft] : Pipe Diameter
ϵ = 2e-4 # [ft] : Pipe Roughness
L = 6 # [ft] : Pipe Length
bc = [16*144, 14.7*144] # [psf] pressure at inlet and outlet -> TODO wil be moved to its own handler with more bc support

my_pipe = Pipe(L, D, 1, 2, ϵ/D, 1)
water = {
    "ρ": 1.94,  # [slugs/ft^3] : Density
    "μ": 2.34e-5, # [lbf-s/ft^2] : Viscosity
    }

max_iterations = 40 # maximum allowable iterations
desired_tolerance = 1e-8 # desired tolerance in solution. When to cease iteration

# %% Evaluate ans setup piping network
# TODO comments in the provided matlab code hint that this perhaps will involve combining different pipe elements into a larger M matrix for solving.

# print some stuff about number of piping elements, and size of M matrix (currently 1 element, 4x4 matrix)

# %% Iterate on the equations
p_n = 0.1*np.ones((4,1)) # init column solution vector

err = 10e2 # init error value
i = 0 # iteration counter

while abs(err) > desired_tolerance and i <= max_iterations:
    A,b = my_pipe.compute_pipe(p_n, water, bc)
    # this will get more advanced as we get to non-singular element networks

    p_n1 = np.linalg.solve(A,b) # solve linear equation Ax=b

    err = max(abs( (p_n-p_n1)/(p_n+1e-16) )) # largest change in any solution value

    print(f"Solution Vector at iteration {i}: {p_n1}")
    print(f"Error at iteration {i}: {err}")

    i+=1
    p_n = p_n1 # p_n = p_n+1

print(f"The solution is determined to be {p_n}")
if i >= max_iterations:
    print("ERROR: The solution is not converged. Iterations terminated after iteration limit was reached")