# MAE 413 - TFCD
# Homework 7
# Timothy Mayer
# 2/14/2022

# Verifies the selection of a fan for a looped segment of pipe

import numpy as np
import matplotlib.pyplot as plt
import pipe_structures as pipes
import boundary_conditions as bc
import pump_curves as pumps
from colebrook_friction_factor import fully_turbulent_f

# %% Inputs
NUM_STATES = 2 # number of states per node to solve for
# Currently pressure, p and mass flowrate, ṁ

D = 3 # [ft] : pipe diameter
K_elb = .30 # Loss coefficient for smooth elbow

water = {
    "ρ": 1.94,  # [slugs/ft^3] : Density
    "μ": 2.34e-5, # [lbf-s/ft^2] : Viscosity
    }

air = {
    "ρ": 0.075/32.174, # [slug/ft^3] : Density
    "μ": 3.74e-7, #[lbf-s/ft^2] : Viscosity
}

vad_fan_head = np.array([4.91439,4.31408,3.79041,3.39776,2.96155,2.52539,2.04565,1.62047]) # [in wg]   ##TODO y crossing at 6.4269,
vad_fan_flowrate = np.array([38034.6,45148.7,50162.8,53427.5,56691.8,59722.7,62870.1,65434.3]) # cfm

vad_fan_head *= 5.20 # [in wg] -> [psf]
vad_fan_flowrate *= air["ρ"]/60 # [cfm] -> [slug/s]

VAD48H21 = pumps.PumpCurve(vad_fan_flowrate, vad_fan_head, fluid=air, units='mdp')

pipe_network = (
    bc.BoundaryCondition(0,0,"pressure"), # reference 0 point, after fan
    pipes.Pipe(20, D, 0,1, 0, 0),
    pipes.Minor(D,D, 1,2, K_elb),
    pipes.Pipe(20, D, 2,3, 0, 0),
    pipes.Minor(D,D, 3,4, K_elb),
    pipes.Pipe(40, D, 4,5, 0, 0),
    pipes.Minor(D,D, 5,6, K_elb),
    pipes.Pipe(20, D, 6,7, 0, 0),
    pipes.Minor(D,D, 7,8, K_elb),
    pipes.Pipe(20, D, 8,9, 0, 0),
    pumps.Pump(9,10, VAD48H21),
    bc.BoundaryCondition(10,0,"pressure")
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
p_n = 10*np.ones((N*NUM_STATES,1)) # init column solution vector

err = 10e2 # init error value
i = 0 # iteration counter

while abs(err) > desired_tolerance and i <= max_iterations:
    A = np.empty((0, N*NUM_STATES)) # init empty matrix
    b = np.empty((0,1))
    
    for elem in pipe_network:
        Ai, bi = elem.compute(p_n, air, N)
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
print("      [slug/s]    [cfm]     [psig]    [slug/cf]  [lbf*s/ft^2]")
for i in range(N):
    print(f"{i}    {float(p_n[i+N]):0.7f}    {float(p_n[i+N]/air['ρ']*60):0.2f}      {float(p_n[i]/144):4.2f}      {air['ρ']:4.4f}      {air['μ']}")

# %plotting results
fig, ax = plt.subplots(figsize=(8.5,5))
VAD48H21.plot()
plt.plot(p_n[9+N], p_n[10]-p_n[9], '.r')
plt.title("Operational Point of VAD48H21 Fan in Wind Tunnel")
plt.legend(("Pumpcurve", "Solution"))
plt.ylabel("Pressure Rise over Fan [psf]")
plt.xlabel("Flowrate [slug/s]")
plt.grid()
plt.show()

input("\nPress any key to exit >> ")
# %%
