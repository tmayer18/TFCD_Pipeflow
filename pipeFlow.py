# MAE 413 - TFCD
# Homework 7
# Timothy Mayer
# 2/14/2022

# Verifies the selection of a fan for a looped segment of pipe

import numpy as np
import matplotlib.pyplot as plt
import logging
import pipe_structures as pipes
import boundary_conditions as bc
import pump_curves as pumps
from iterative_solver import iterative_compute

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING) # matplotlib generates lots of debug logs

# %% Inputs
NUM_STATES = 2 # number of states per node to solve for
# Currently pressure, p and mass flowrate, ṁ

D = 3*0.3048 # [ft] -> [m] : pipe diameter
K_elb = .30 # Loss coefficient for smooth elbow

water = {
    "ρ": 1.94,  # [slugs/ft^3] : Density
    "μ": 2.34e-5, # [lbf-s/ft^2] : Viscosity
    }

air = {
    "ρ": 1.1790290265088545, # [kg/m^3] : Density  ## OLD FIELD
    "μ": 1.8537365427712757e-05, #[?SI] : Viscosity ## OLD FIELD
    "name": "air",
    "T_ref": 315 # K
}

vad_fan_head = np.array([4.91439,4.31408,3.79041,3.39776,2.96155,2.52539,2.04565,1.62047]) # [in wg]   ##TODO y crossing at 6.4269,
vad_fan_flowrate = np.array([38034.6,45148.7,50162.8,53427.5,56691.8,59722.7,62870.1,65434.3]) # cfm

vad_fan_head *= 5.20 * 47.88# [in wg] -> [psf] -> [Pa]
vad_fan_flowrate *= air["ρ"]/60 * (0.3048)**3 # [cfm] -> [m^3/s] -> [kg/s]

VAD48H21 = pumps.PumpCurve(vad_fan_flowrate, vad_fan_head, fluid=air, units='mdp')

# NOTE research into the most recommended unit conversion libraries
# units - convoluted syntax, straightforward usage. 2 [m] -> unit('m')(2) 
            # works in numpy fine, but unitless operations like log or sin fail
            # I also don't like the syntax, it feels bulky
# pint - the most popular option by far. Easier to use inline. 2 [m] -> 2*ureg.m
            # this has proper support for unitless operations (log, sin ect..)
            # defualt formatting for printing is ugly (2 meter), but can be changed
# unum - a pint alternative. Even easier inline support 2 [m] -> 2*m or 2*u.m
            # more memory efficient integration with numpy arrays, 
            # but fails on lots of common operations (log, sin...)


p_ref = 14.7*144*47.88 # [psf] -> [Pa]
pipe_network = (
    bc.BoundaryCondition(0,0+p_ref,"pressure"), # reference 0 point, after fan
    pipes.Pipe(20*0.3048, D, 0,1, 0, 0),
    pipes.Minor(D,D, 1,2, K_elb),
    pipes.Pipe(20*0.3048, D, 2,3, 0, 0),
    pipes.Minor(D,D, 3,4, K_elb),
    pipes.Pipe(40*0.3048, D, 4,5, 0, 0),
    pipes.Minor(D,D, 5,6, K_elb),
    pipes.Pipe(20*0.3048, D, 6,7, 0, 0),
    pipes.Minor(D,D, 7,8, K_elb),
    pipes.Pipe(20*0.3048, D, 8,9, 0, 0),
    pumps.Pump(9,10, VAD48H21),
    bc.BoundaryCondition(10,0+p_ref,"pressure")
)

# %% Iterate on the solution
p_n, N = iterative_compute(pipe_network, air, 1e-8, 40, NUM_STATES)

# %% Print the results
print(" SOLUTION ")
print("Node     ṁ          Q         p           ρ          μ")
print("      [kg/s]    [cfm?]     [Pa]    [slug/cf?]  [lbf*s/ft^2?]")
for i in range(N):
    print(f"{i}    {float(p_n[i+N]):0.7f}    {float(p_n[i+N]/air['ρ']*60):0.2f}      {float(p_n[i]):4.2f}      {air['ρ']:4.4f}      {air['μ']}")

# %plotting results
fig, ax = plt.subplots(figsize=(8.5,5))
VAD48H21.plot()
plt.plot(p_n[9+N], p_n[10]-p_n[9], '.r')
plt.title("Operational Point of VAD48H21 Fan in Wind Tunnel")
plt.legend(("Pumpcurve", "Solution"))
plt.ylabel("Pressure Rise over Fan [Pa]")
plt.xlabel("Flowrate [kg/s]")
plt.grid()
plt.show()

input("\nPress any key to exit >> ")
# %%
