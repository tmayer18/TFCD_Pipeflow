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
from colebrook_friction_factor import fully_turbulent_f

logging.basicConfig(level=logging.DEBUG)

NUM_STATES = 3 # track 3 quantities, m,p,T

water = {
    "name":"water",
    "T_ref":(273.15+20)*u.K,
    "p_ref":0*u.Pa
}

Do = 0.0262128*u.m
Di = 0.0159258*u.m
Dh = Do-Di
ϵ = 0.61*u.mm
L = 0.9906*u.m
Dp = 0.0140208*u.m

pipe_network = (
    thermal.ThermallyConnected(
        pipes.Annulus(L, Di, Do, 0,1, ϵ/Dh, 0*u.m, water), # cold
        pipes.Pipe(L, Dp, 2,3, ϵ/Dp, 0*u.m, water), # hot
        thermal.NestedPipeWall(401*u.W/u.m/u.K),
        (0,2)
    ),
    bc.BoundaryCondition(2, (121.4+459.67)*u.Rk, 'temperature'),
    bc.BoundaryCondition(1, (63.8+459.67)*u.Rk, 'temperature'),
    bc.BoundaryCondition(0, -0.656*u.lbm/u.s, 'mass_flowrate'),
    bc.BoundaryCondition(2, 0.174*u.lbm/u.s, 'mass_flowrate'),
    bc.BoundaryCondition(0, u.atm, 'pressure'),
    bc.BoundaryCondition(2, u.atm, 'pressure'),
)

p_0, N = init.uniform_thermal_fluidflow(pipe_network, NUM_STATES)
p_n = solver.iterative_compute(pipe_network, 1e-8, 200, p_0, N, NUM_STATES)

print('======= SOLUTION =======')
solver.print_results_table(p_n, has_temp=True, ṁ_units=u.lbm/u.s, T_units=u.Rk, rel_temp=True)