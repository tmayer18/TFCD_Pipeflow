# Schuyler McNaughton's First Attempt at Python
# TBH, Matlab is better.

# This function contains a repository of h correlations that are dependent on Geometry,
# Temperature, Prandlt Number, and Reynolds Number

# Need to know: Temperature, Prandlt Number, Reynolds Number, Annulus/Pipe, Length, Diameter, Friction Factor,  Viscosity???
# Mean Temp, Surface Temp

import numpy as np # Cuz Tim did it

def geth(TempConfig,Pr,Re,Ts,Tm,k,D):
    
    # Finds Nusselt Number for an Annulus   
    if isinstance(None,Annulus):
        if Re < 1E4: # Laminar Correlation for an Annulus
            Nu = 5.385
        else: # Double linear interpolation dependant on Prandtl Number and Reynolds Number
            Pr_Turbulent = (0, 0.001, 0.003, 0.01, 0.03, 0.5, 0.7, 1.0, 3.0, 10, 30, 100, 1000)
            Re_Turbulent = (1E4, 3E4, 1E5, 3E5, 1E6)
            Nu_Turbulent = ((5.7, 5.7, 5.7, 5.8, 6.1, 22.5, 27.8, 35, 60.8, 101, 147, 210, 390),
            (5.78, 5.78, 5.8, 5.92, 6.9, 47.8, 61.2, 76.8, 142, 214, 367, 514, 997),
            (5.8, 5.8, 5.9, 6.7, 11, 120, 155, 197, 380, 680, 1030, 1520, 2880),
            (5.8, 5.88, 6.32, 9.8, 23, 290, 378, 486, 966, 1760, 2720, 4030, 7650),
            (5.8, 6.23, 8.62, 21.5, 61.2, 780, 1030, 1340, 2700, 5080, 8000, 12000, 23000))


        for n in range(len(Re_Turbulent)-1): # Reynolds Number Interpolation
            if Re >= Re_Turbulent[n] and Re < Re_Turbulent[n+1]:
                Nu_Single_Interp = Nu_Turbulent[all,n] + (Re - Re_Turbulent[n])/(Re_Turbulent[n+1]-Re_Turbulent[n])*(Nu_Turbulent[all,n+1]-Nu_Turbulent[all,n])
        
        for n in range(len(Pr_Turbulent)-1): # Prandtl Number Interpolation
            if Pr >= Pr_Turbulent[n] and Pr < Pr_Turbulent[n+1]:
                Nu = Nu_Single_Interp[n] + (Pr - Pr_Turbulent[n])/(Pr_Turbulent[n+1] - Pr_Turbulent[n])*(Nu_Single_Interp[n+1] - Nu_Single_Interp[n])
        

    # Finds Nusselt Number for a Pipe
    if isinstance(None,Pipe):
        # Dittus Bolter
        if Pr <= 160 and Pr >= 0.6 and Re >10000 and Ts > Tm : # Annulus water is hotter than Pipe Water
                n = 0.4
                Nu = 0.023*Re**0.8*Pr**n
        elif Pr <= 160 and Pr >= 0.6 and Re >10000 and Ts < Tm : # Pipe Water is hotter than Annulus Water
            n = 0.4
            Nu = 0.023*Re**0.8*Pr**n
        else: # For Laminar Flow
            Nu = 4.364

    h = Nu*k/D
    return h