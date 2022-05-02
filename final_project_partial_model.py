# TCFD Final Project - Modelling of the Copper Heat Exchanger
# 4-27-2022
# Timothy Mayer, Scheuyler McNaughton


import logging
from unum_units import units2 as u
import pipe_structures as pipes
import thermal_structures as thermal
import boundary_conditions as bc
import solution_initialization as init
import iterative_solver as solver

logging.basicConfig(level=logging.INFO)

NUM_STATES = 3 # track 3 quantities, m,p,T

water = {
    "name":"water",
    "T_ref":(273.15+20)*u.K,
    "p_ref":0*u.Pa,
    "use_coolprop":True
}

ϵ = 0.61*u.mm # roughness of copper pipe
OD_large = 1.125*u.inch
ID_large = OD_large-2*(0.0465*u.inch)
OD_mid = 0.872*u.inch
ID_mid = OD_mid-2*(0.0475*u.inch)
OD_small = 0.627*u.inch
ID_small = OD_small-2*(0.0375*u.inch)

L = 1.00965*u.m

ϵD_inner = ϵ/ID_small
ϵD_outer = ϵ/(ID_large-OD_small)
ϵD_mid = ϵ/ID_mid

K_90_small = 0.72
K_90_mid = 0.66
K_junct = 0.66

k_cu = 401*u.W/u.m/u.K # copper conduction

# parallel flow
pipe_network = ( # NOTE using 0-indexed nodes, matching the spreadsheet not the diagram
    thermal.ThermallyConnected(
        pipes.Pipe(L, ID_small, 9,10, ϵD_inner, 0*u.m, water), #pipe10
        pipes.Annulus(L, OD_small, ID_large, 56,57, ϵD_outer, 0*u.m, water), #pipe64
        thermal.NestedPipeWall(k_cu),
        (9,56)
    ),
    thermal.AdiabaticPipe(pipes.Pipe(3.5*u.inch, ID_small, 10,11, εD_inner, 0*u.m, water)), #pipe11
    thermal.AdiabaticPipe(pipes.Minor(ID_mid, ID_mid, 57,58, K_junct, water)), #junction-split pipe12 - WEIRD MINOR LOSS - PRETENDING THE DIAMTER DOESNT LIKE... TOTALLY CHANGE HERE
    thermal.AdiabaticPipe(pipes.Minor(ID_small, ID_small, 11,12, K_90_small, water)), #90 pipe13
    thermal.AdiabaticPipe(pipes.Pipe(5*u.inch, ID_small, 12,13, εD_inner, -5*u.inch, water)), # pipe14
    thermal.AdiabaticPipe(pipes.Minor(ID_small, ID_small, 13,14, K_90_small, water)), #90 pipe15
    thermal.AdiabaticPipe(pipes.Pipe(5*u.inch, ID_mid, 58,59, ϵD_mid, -5*u.inch, water)), # pipe65
    thermal.AdiabaticPipe(pipes.Pipe(3.5*u.inch, ID_small, 14,15, εD_inner, 0*u.m, water)), # pipe16
    thermal.AdiabaticPipe(pipes.Minor(ID_mid, ID_mid, 59,60, K_junct, water)), #junction-union pipe17
    thermal.ThermallyConnected(
        pipes.Pipe(L, ID_small, 15,16, ϵD_inner, 0*u.m, water), #pipe18
        pipes.Annulus(L, OD_small, ID_large, 60,61, ϵD_outer, 0*u.m, water), #pipe66
        thermal.NestedPipeWall(k_cu),
        (15,60)
    ),
    thermal.AdiabaticPipe(pipes.Pipe(3.5*u.inch, ID_small, 16,17, εD_inner, 0*u.m, water)), #pipe19
    thermal.AdiabaticPipe(pipes.Minor(ID_mid, ID_mid, 61,62, K_junct, water)), #junction-split pipe20
    thermal.AdiabaticPipe(pipes.Minor(ID_small, ID_small, 17,18, K_90_small, water)), #90 pipe21
    thermal.AdiabaticPipe(pipes.Pipe(5*u.inch, ID_small, 18,19, εD_inner, -5*u.inch, water)), #pipe22
    thermal.AdiabaticPipe(pipes.Pipe(5*u.inch, ID_mid, 62,63, ϵD_mid, -5*u.inch, water)), #pipe67
    bc.BoundaryCondition(9, 0.2*u.lbm/u.s, "mass_flowrate"),
    bc.BoundaryCondition(56, 0.7*u.lbm/u.s, "mass_flowrate"),
    bc.BoundaryCondition(9, u.atm, "pressure"),
    bc.BoundaryCondition(56, u.atm, "pressure"),
    bc.BoundaryCondition(9, (120+459.67)*u.Rk, "temperature"),
    bc.BoundaryCondition(56, (50+459.67)*u.Rk, "temperature"),

)

pipe_network, old_nodes_table = init.normalize_node_numbers(pipe_network)

p_0, N = init.uniform_thermal_fluidflow(pipe_network, NUM_STATES)
p_n = solver.iterative_compute(pipe_network, 1e-8, 200, p_0, N, NUM_STATES)

print('======= SOLUTION =======')
solver.print_results_table(p_n, has_temp=True, ṁ_units=u.lbm/u.s, T_units=u.Rk, rel_temp=True, node_conversions=old_nodes_table)