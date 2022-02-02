# MAE 413 - TFCD
# Homework 1
# Timothy Mayer
# 1/19/2022 - Ported to Python 1/21/2022

import math
from statistics import mean

def iterative_solve_colebrook(ϵD, Re):
    '''Iteratively solves the self-referential Colebrook-White Equation for the Darcy Friction Factor f:
    
    ϵD : Relative Roughness [unitless], ϵ/D
    Re : Reynolds Number [unitless], pipe-diameter based Re_D
    '''
    if Re < 2300:
        # Laminar Flow, friction factor derivable from Poiseuille Flow
        return 64/Re

    elif Re >= 2300:
        # Turbulent Flow, f requires iteration on Colebrook
        lb = 1e-10 # lower limit of solvable friction factors
        ub = 0.3 # upper limit of solvable friction factors - 3x higher than expected values
        desired_e = 1e-8 # desired level of accuracy (error) at which to end iteration
        current_e = 1e10 # current error - initalized to large number

        #error function definition. Rearrangement of Colebrook equation
        err = lambda f: -2.0*math.log10( (ϵD/3.7) + (2.51/(Re*f**0.5)) ) - f**(-0.5)

        while abs(current_e) > desired_e:
            cen = mean((lb,ub)) # bisector value
            current_e = err(cen)
            if err(lb)*err(cen) < 0:
                # zero-crossing exists between lb and cen;
                ub = cen # shrink bounds
            elif err(cen)*err(ub) < 0:
                # zero-crossing exists between cen and ub;
                lb = cen # shrink bounds

        return cen #take bisector value as the friction factor to return

    #else:
        # TODO error handling and defaults

def fully_turbulent_f(ϵD):
    '''Returns the fully-turbulent friction factor from the Colebrook Equation

    ϵD : Relative Roughness [unitless], ϵ/D
    '''
    return 1 / (-2.0*math.log10(ϵD/3.7))**2

# test case, ran as main
if __name__ == "__main__":
    # treat program as a calculator, and as for user input to solve for
    if input("Fully Turbulent Solver? y/n >> ").lower() == "y":
        f = fully_turbulent_f(float(input("Relative Roughness ϵ/D = ")))
        print(f"The fully-turbulent friction factor is {f}")
    else:
        f = iterative_solve_colebrook(float(input("Relative Roughness ϵ/D = ")), float(input("Reynolds Number Re = ")))
        print(f"The friction factor is {f}")

    input("\nPress any key to exit >> ")