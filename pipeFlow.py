# MAE 413 - TFCD
# Homework 6
# Timothy Mayer
# 2/8/2022

# Will compute the pressure [psf] and flowrate [slug/s] at the ends of a pipe network

import numpy as np
import matplotlib.pyplot as plt
import pipe_structures as pipes
import boundary_conditions as bc
import pump_curves as pumps
from colebrook_friction_factor import fully_turbulent_f

# %% Inputs
NUM_STATES = 2 # number of states per node to solve for
# Currently pressure, p and mass flowrate, ṁ

D = 4.026/12 # [ft] : pipe diameter
ϵ = 1.8e-3/12 # [ft] : roughness
ϵD = ϵ/D # [ul] : relative roughness, ϵ/D
ft = fully_turbulent_f(ϵD) # [ul] : fully turbulent friction factor
K_elb = 30*ft
K_entr = 0.5
K_filt = 5.5

water = {
    "ρ": 1.94,  # [slugs/ft^3] : Density
    "μ": 2.34e-5, # [lbf-s/ft^2] : Viscosity
    }

goulds_curve_head = np.array([108.936, 108.936, 108.085, 107.66, 106.383, 104.255, 100.851, 97.0213, 93.1915, 88.0851, 81.7021, 75.3191, 69.3617, 65.9574, 0])
goulds_curve_flowrate = np.array([0, 100.127, 200.413, 279.361, 375.378, 462.859, 565.274, 665.555, 748.766, 827.707, 919.45, 996.256, 1064.53, 1098.66, 1446.37])
Goulds3196 = pumps.PumpCurve(goulds_curve_flowrate, goulds_curve_head, fluid=water, units='QH')

pipe_network = (
    bc.BoundaryCondition(0, 0, "pressure"), # p_atm
    pipes.Pipe(5.3, 4, 0,1, ϵD, -5.3), # full tank
    # pipes.Pipe(0, 4, 0,1, ϵD, 0), # empty tank
    pipes.Minor(D,D, 1,2, K_entr),
    pipes.Pipe(10, D, 2,3, ϵD, 0),
    pumps.Pump(3,4, Goulds3196),
    pipes.Pipe(20, D, 4,5, ϵD, 0),
    pipes.Minor(D,D, 5,6, K_filt),
    pipes.Pipe(20, D, 6,7, ϵD, 0),
    pipes.Minor(D,D, 7,8, K_elb),
    pipes.Pipe(6, D, 8,9, ϵD, 6),
    pipes.Minor(D,D, 9,10, K_elb),
    pipes.Pipe(15, D, 10,11, ϵD, 0),
    bc.BoundaryCondition(11, 0, "pressure") # p_atm
)

max_iterations = 40 # maximum allowable iterations
desired_tolerance = 1e-8 # desired tolerance in solution. When to cease iteration

# %% Pre-printing information
# print some stuff about number of piping elements, and size of M matrix
N = 0 # number of nodes
nodes = []
last_node = -1
pipe_likes = 0
for elem in pipe_network:
    N += elem.num_nodes
    last_node = max((last_node,)+elem.nodes)
    if type(elem) in [pipes.Minor, pipes.Pipe, pipes.Tee]:
        pipe_likes += 1 # count the number of pipe-likes in the network

N = int(N/2) # Each node in the network shows up exactly twice in the network, at the start and end of its pipe-like, or as a boundary condition
assert last_node+1 == N, "There aren't enough equations!"
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
print("      [slug/s]    [gpm]     [psig]    [slug/cf]  [lbf*s/ft^2]")
for i in range(N):
    print(f"{i}    {float(p_n[i+N]):0.7f}    {float(p_n[i+N]/water['ρ']/0.13368*60):0.2f}      {float(p_n[i]/144):4.2f}      {water['ρ']}      {water['μ']}")

# %plotting results
fig, ax = plt.subplots(figsize=(8.5,5))
Goulds3196.plot()
plt.plot(p_n[3+N], p_n[4]-p_n[3], '.r')
plt.title("Full Tank Case")
plt.legend(("Pumpcurve", "Solution"))
plt.ylabel("Pressure Rise over Pump [psf]")
plt.xlabel("Flowrate [slug/s]")
plt.grid()
plt.show()

input("\nPress any key to exit >> ")
# %%
