# R you ready kids??? Aye Aye Captain
# Hehehe I'm hilarious - Also by Schuyler, but this probably doesn't need to be a function

from numpy import pi
import numpy as np

def getR(h_Pipe,h_Annulus,Pipe_Outer_Diameter,Annulus_Inner_Diameter,Length_Pipe,k):

    r_Pipe = Pipe_Outer_Diameter/2
    r_Annulus = Annulus_Inner_Diameter/2

    A_Annulus = 2*pi*r_Annulus*Length_Pipe
    A_Pipe = 2*pi*r_Pipe*Length_Pipe

    R = 1/(h_Annulus*A_Annulus) + np.log(r_Annulus/r_Pipe)/(2*pi*Length_Pipe*k) + 1/(h_Pipe*A_Pipe)

    return R