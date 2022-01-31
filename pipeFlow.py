# MAE 413 - TFCD
# Homework 4
# Timothy Mayer
# 1/31/2022

# Will compute the pressure [psf] and flowrate [slug/s] at the ends of a pipe

import numpy as np
import pipe_structures as pipes
import boundary_conditions as bc
from colebrook_friction_factor import fully_turbulent_f

# %% Inputs
NUM_STATES = 2 # number of states per node to solve for
# Currently pressure, p and mass flowrate, ṁ

D = 4.026/12 # [ft] : pipe diameter
ϵD = 4.47e-4 # [ul] : relative roughness, ϵ/D
ft = fully_turbulent_f(ϵD) # [ul] : fully turbulent friction factor
pipe_network = (
    pipes.Pipe(50, D, 0, 1, ϵD, 0),
    pipes.Minor(D, D, 1, 2, 30*ft), # elbow
    pipes.Pipe(10, D, 2, 3, ϵD, 10), 
    pipes.Minor(D, D, 3, 4, 30*ft), # elbow
    pipes.Minor(D, D, 4, 5, 3*ft), # ball valve
    pipes.Pipe(50, D, 5, 6, ϵD, 0),
    bc.BoundaryCondition(0, (50+14.7)*144, "pressure"), 
    bc.BoundaryCondition(6, 14.7*144, "pressure")
)

water = {
    "ρ": 1.94,  # [slugs/ft^3] : Density
    "μ": 2.34e-5, # [lbf-s/ft^2] : Viscosity
    }

max_iterations = 40 # maximum allowable iterations
desired_tolerance = 1e-8 # desired tolerance in solution. When to cease iteration

# %% Pre-printing information
# print some stuff about number of piping elements, and size of M matrix
N = 0 # number of nodes
nodes = []
pipe_likes = 0
for elem in pipe_network:
    N += elem.num_nodes
    if type(elem) in [pipes.Minor, pipes.Pipe]:
        pipe_likes += 1 # count the number of pipe-likes in the network

N = int(N/2) # Each node in the network shows up exactly twice in the network, at the start and end of its pipe-like, or as a boundary condition
# TODO more error-checking on the pipe network to ensure that is is valid
print(f"There are {pipe_likes} pipe-likes in the network, requiring a {N*2}x{N*2} matrix to solve\n")


# %% Iterate on the equations
p_n = 0.1*np.ones((N*NUM_STATES,1)) # init column solution vector

err = 10e2 # init error value
i = 0 # iteration counter

while abs(err) > desired_tolerance and i <= max_iterations:
    A = np.empty((0, N*NUM_STATES)) # init empty matrix
    b = np.empty((0,1))
    
    for elem in pipe_network:
        Ai, bi = elem.compute(p_n, water, N)
        A = np.append(A, Ai, axis=0) # append matrix-linearized equations to the matrix
        b = np.append(b, bi, axis=0)
    
    # print(f"CHECK: the length of the matrix is {len(A)}x{len(A[0])}")

    p_n1 = np.linalg.solve(A,b) # solve linear equation Ax=b

    err = max(abs( (p_n-p_n1)/(p_n+1e-16) )) # largest percent change in any solution value

    # print(f"Solution Vector at iteration {i}: {p_n1}")
    # print(f"Error at iteration {i}: {err}")

    i+=1
    p_n = p_n1.copy() # p_n = p_n+1
    # copy is necessary cause pointers. Otherwise they will be the same object

# %% Print the results
if i >= max_iterations:
    print("ERROR: The solution is not converged. Iterations terminated after iteration limit was reached")

print(" SOLUTION ")
print("Node     ṁ          Q         p           ρ          μ")
print("      [slug/s]    [gpm]     [psia]    [slug/cf]  [lbf*s/ft^2]")
for i in range(N):
    print(f"{i}    {float(p_n[i+N]):0.7f}    {float(p_n[i+N]/water['ρ']/0.13368*60):0.2f}      {float(p_n[i]/144):4.2f}      {water['ρ']}      {water['μ']}")

input("\nPress any key to exit >> ")