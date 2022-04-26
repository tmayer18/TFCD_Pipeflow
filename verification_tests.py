#%% MAE 413 - TFCD
# Timothy Mayer
# 4/14/2022

# Various verification tests for the pipe-flow solver code
#   constructed with juypter notebook notation for running individual tests

import logging
from unum_units import units2 as u
import pipe_structures as pipes
import thermal_structures as thermal
import boundary_conditions as bc
import solution_initialization as init
import iterative_solver as solver
from colebrook_friction_factor import fully_turbulent_f

logging.basicConfig(level=logging.INFO)

# %% Single Pipe - Homework 3     ===================================================
NUM_STATES = 2 # pressure and massflow

water = {
    "name": "water",
    "T_ref": (273+4)*u.K, # temperature at which to lookup properties
    "p_ref": 0, # adjustment from network guage pressures to absolute
}

L = 6*u.ft
D = 0.936*u.inch

pipe_network = (
    bc.BoundaryCondition(0, 16*u.psi, 'pressure'),
    pipes.Pipe(L, D, 0, 1, 2e-4*u.ft/D, 1*u.ft, water),
    bc.BoundaryCondition(1, 14.7*u.psi, 'pressure')
)

p_0, N = init.uniform_fluidflow(pipe_network, NUM_STATES)

solver.iter_solution_log.config(pressure_units=u.psf, massflow_units=u.slug/u.s)
p_n = solver.iterative_compute(pipe_network, 1e-8, 100, p_0, N, NUM_STATES)

print(' SOLUTION ')
solver.print_results_table(p_n, p_units=u.psi, ṁ_units=u.slug/u.s)
print("Expected ṁ = 0.07119 [slug/s]")

# %% Single Elbow - Homework 4a    ==========================================
NUM_STATES = 2 # pressure and massflow

water = {
    "name": "water",
    "T_ref": (273+4)*u.K, # temperature at which to lookup properties
    "p_ref": u.atm, # adjustment from network guage pressures to absolute
}

D = 1.049*u.inch
ft = fully_turbulent_f(2e-4*u.ft/D)

pipe_network = (
    bc.BoundaryCondition(0, 0.0001*u.psi, 'pressure'),
    pipes.Minor(D, D, 0,1, 30*ft, water),
    bc.BoundaryCondition(1, -1.5*u.psi, 'pressure')
)

p_0, N = init.uniform_fluidflow(pipe_network, NUM_STATES)

solver.iter_solution_log.config(pressure_units=u.psf, massflow_units=u.slug/u.s)
p_n = solver.iterative_compute(pipe_network, 1e-8, 30, p_0, N, NUM_STATES)

print(' SOLUTION ')
solver.print_results_table(p_n, p_units=u.psi, ṁ_units=u.slug/u.s)
print("Expected ṁ = 0.20357 [slug/s]")

# %% Single Nozzle - Homework 4b    ==================================================
NUM_STATES = 2 # pressure and massflow

water = {
    "name": "water",
    "T_ref": (273+4)*u.K, # temperature at which to lookup properties
    "p_ref": 0, # adjustment from network guage pressures to absolute
}

pipe_network = (
    bc.BoundaryCondition(0, 14.7*u.psi, 'pressure'),
    pipes.Minor(4*u.inch, 2*u.inch, 0,1, 0, water),
    bc.BoundaryCondition(1, 8*u.psi, 'pressure')
)

p_0, N = init.uniform_fluidflow(pipe_network, NUM_STATES)

solver.iter_solution_log.config(pressure_units=u.psf, massflow_units=u.slug/u.s)
p_n = solver.iterative_compute(pipe_network, 1e-8, 30, p_0, N, NUM_STATES)

print(' SOLUTION ')
solver.print_results_table(p_n, p_units=u.psf, ṁ_units=u.slug/u.s)
print("Expected ṁ = 1.3785 [slug/s]")

# %% Simple Network- Homework 4c    ============================================
NUM_STATES = 2 # pressure and massflow

water = {
    "name": "water",
    "T_ref": (273+4)*u.K, # temperature at which to lookup properties
    "p_ref": u.atm, # adjustment from network guage pressures to absolute
}

D = 4*u.inch
ϵD = 4.47e-4*u.ul
ft = fully_turbulent_f(ϵD)

pipe_network = (
    bc.BoundaryCondition(0, 50*u.psi, 'pressure'),
    pipes.Pipe(50*u.ft, D, 0,1, ϵD, 0*u.ft, water),
    pipes.Minor(D, D, 1,2, 30*ft, water), # elbow
    pipes.Pipe(10*u.ft, D, 2,3, ϵD, 10*u.ft, water),
    pipes.Minor(D, D, 3,4, 30*ft, water), # elbow
    pipes.Minor(D, D, 4,5, 3*ft, water), # valve
    pipes.Pipe(50*u.ft, D, 5,6, ϵD, 0*u.ft, water),
    bc.BoundaryCondition(6, 1e-10*u.psi, 'pressure')
)

p_0, N = init.uniform_fluidflow(pipe_network, NUM_STATES)

solver.iter_solution_log.config(pressure_units=u.psf, massflow_units=u.slug/u.s)
p_n = solver.iterative_compute(pipe_network, 1e-8, 30, p_0, N, NUM_STATES)

print(' SOLUTION ')
solver.print_results_table(p_n, ṁ_units=u.slug/u.s, p_units=u.psf)
print("Expected ṁ = 5.5107 [slug/s]")


# %% Simple Split Network- Homework 5    ============================================
NUM_STATES = 2 # pressure and massflow

water = {
    "name": "water",
    "T_ref": (273+4)*u.K, # temperature at which to lookup properties
    "p_ref": 0, # adjustment from network guage pressures to absolute
}

D = 1.610*u.inch
ϵ = 1.8e-3*u.inch
ϵD = ϵ/D
ft = fully_turbulent_f(ϵD)

# TODO dummy zero-index element
pipe_network = (
    pipes.Pipe(15*u.ft, D, 0,1, ϵD, 0*u.m, water),
    pipes.Tee(D, 1, (2,6), (2,6), ϵD, water),
    pipes.Pipe(8*u.ft, D, 2,3, ϵD, 8*u.ft, water),
    pipes.Minor(D, D, 3,4, 30*ft, water), # Elbow
    pipes.Pipe(85*u.ft, D, 4,5, ϵD, 0*u.m, water),
    bc.BoundaryCondition(5, u.atm, 'pressure'),
    bc.BoundaryCondition(5, 0.12*u.slug/u.s, 'mass_flowrate'),
    pipes.Pipe(6*u.ft, D, 6,7, ϵD, -6*u.ft, water),
    pipes.Tee(D, 7, (8,15), (7,8), ϵD, water),
    pipes.Pipe(2*u.ft, D, 8,9, ϵD, -2*u.ft, water),
    pipes.Minor(D, D, 9,10, 30*ft, water), # Elbow
    pipes.Pipe(10*u.ft, D, 10,11, ϵD, 0*u.m, water),
    pipes.Tee(D, (11,18), 12, (11,12), ϵD, water),
    pipes.Pipe(10*u.ft, D, 15,16, ϵD, 0*u.m, water),
    pipes.Minor(D,D, 16,17, 30*ft, water), # Elbow
    pipes.Pipe(2*u.ft, D, 17,18, ϵD, -2*u.ft, water),
    pipes.Minor(D, 1.049*u.inch, 12,13, 0.0214996, water),
    pipes.Pipe(10*u.ft, 1.049*u.inch, 13,14, ϵ/(1.049*u.inch), 0*u.m, water),
    bc.BoundaryCondition(14, u.atm, 'pressure')
)

p_0, N = init.uniform_fluidflow(pipe_network, NUM_STATES)

solver.iter_solution_log.config(pressure_units=u.psi, massflow_units=u.slug/u.s)
p_n = solver.iterative_compute(pipe_network, 1e-8, 30, p_0, N, NUM_STATES)

print(' SOLUTION ')
solver.print_results_table(p_n, ṁ_units=(u.slug/u.s), p_units=(u.psi))
print("\nExpected output ṁ_14 = 0.21469 [slug/s]")
print("Expected inlet pressure p_0 = 7.48 [psig]")

# %% Single Adiabatic Pipe - Project   ===============================================================
NUM_STATES = 3

water = {
    'name':'water',
    'p_ref':0,
    'T_ref':300*u.K
}

L = 6*u.ft
D = 0.936*u.inch

pipe_network = (
    bc.BoundaryCondition(0, 16*u.psi, 'pressure'),
    bc.BoundaryCondition(0, 315*u.K, 'temperature'),
    thermal.AdiabaticPipe(
        pipes.Pipe(L, D, 0, 1, 2e-4*u.ft/D, 1*u.ft, water)
    ),
    bc.BoundaryCondition(1, 14.7*u.psi, 'pressure')
)

p_0, N = init.uniform_thermal_fluidflow(pipe_network, NUM_STATES)

# iter_solution_log.config(pressure_units=u.psf, massflow_units=u.slug/u.s)
p_n = solver.iterative_compute(pipe_network, 1e-8, 100, p_0, N, NUM_STATES)

print(' SOLUTION ')
solver.print_results_table(p_n, ṁ_units=(u.slug/u.s), p_units=(u.psi), has_temp=True)

print("Expected ṁ = 0.07119 [slug/s]")
print("Expected T1=T2 = 315K")

# %% Simple pipe-in-pipe counterflow heat exchanger, Heat Transfer Textbook ex 11.1
NUM_STATES = 3

water = {
    "name":"water",
    "T_ref":300*u.K,
    "p_ref":0*u.Pa,
    "use_coolprop":False, # disable coolprop property lookup to match textbook properties
    "Cp":2131*u.J/u.kg/u.K,
    "μ":725e-6*u.N*u.s/(u.m**2),
    "ρ":1/(1.007e-3*u.m**3/u.kg),
    "k":0.625*u.W/u.m/u.K,
    "Pr":4.85
}
oil = {
    "name":"oil", # engine oil, not a coolprop fluid
    "T_ref":300*u.K,
    "p_ref":0*u.Pa,
    "use_coolprop":False, # disable coolprop, as its not even a fluid in coolprop
    "Cp":2131*u.J/u.kg/u.K,
    "μ":3.25e-2*u.N*u.s/(u.m**2),
    "ρ":853*u.kg/(u.m**3),
    "k":0.138*u.W/u.m/u.K,
    "Pr":None
}

pipe_network = (
    thermal.ThermallyConnected(
        pipes.Annulus(65.9*u.m, 25*u.mm, 45*u.mm, 0,1, 0,0*u.m, oil),
        pipes.Pipe(65.9*u.m, 25*u.mm, 2,3, 0,0*u.m, water),
        thermal.NestedPipeWall(1e-12*u.W/u.m/u.K), # neglecting conduction resistance, probably will cause div/by/zero error
        (0,2)
    ),
    bc.BoundaryCondition(0, 0.1*u.kg/u.s, 'mass_flowrate'),
    bc.BoundaryCondition(2, -0.2*u.kg/u.s, 'mass_flowrate'),
    bc.BoundaryCondition(0, u.atm, 'pressure'),
    bc.BoundaryCondition(2, u.atm, 'pressure'),
    bc.BoundaryCondition(0, (100+273.15)*u.K, 'temperature'),
    bc.BoundaryCondition(2, (30+273.15)*u.K, 'temperature'),
)

p_0, N = init.uniform_thermal_fluidflow(pipe_network, NUM_STATES)

p_n = solver.iterative_compute(pipe_network, 1e-8, 100, p_0, N, NUM_STATES)
solver.print_results_table(p_n, has_temp=True)