# %%
#   tests and compared the functionaloity of some various unit libraries

import numpy as np
import math

# %%pint 
print("\n\nPINT TIME")
import pint
ureg = pint.UnitRegistry()
a = 4*ureg.kg
b = 5*ureg.m
c = 6*ureg.s**2
print(a*b*c)
try:
    print(a+b)
except Exception as e:
    print(e)
a = np.array([1,2,3,4])*ureg.kg
print(a)
# try:

#     print(np.array([1*ureg.kg, 2*ureg.kg]))
#     print(np.array([1*ureg.kg, 2*ureg.m]))
#     # we can't put different Quantities inside a np array - 
#     # pint wants to entirely wrap the matrix whch won't work
#     # for this usecase
# except Exception as e:
#     print(e)

print(1*ureg.m + 2*ureg.cm)
repr(1*ureg.m + 2*ureg.cm)
repr(ureg.cm)
repr(1*ureg.cm + 4*ureg.m)

# %%unum
print("\n\nUNUM Time")
import unum.units as u
a = 1*u.kg
b = 2*u.m
c = 3*u.s**2
print(a*b*c)
a = np.array([1,2,3])*u.kg
a1 = u.kg*np.array([1,2,3])
b = np.array([1*u.kg, 2*u.m, 3*u.s])
# unum *does* let us put multiple units inside a matrix just fine
print(a)
print(a1)
print(b)
print(b*u.N)
try:
    print(math.log(a))
    print(np.log(a))
    # BUT unum doesn't support any of the math(trig) operations
except Exception as e:
    print(e)
print(f"a log operation: {math.log(3*u.m/(4*u.m))}") # but does on default math
#print(f"another log operation: {np.log(3*u.m/(4*u.m))}") # but does on default math
a = np.array([[1*u.kg/(u.s**2), 2/u.m]])
b = np.array([[6*u.m],[5*u.J]])
print(a@b)

A = np.array([[1,2],[3,4]])*u.kg
print(type(A))
try:
    print(np.linalg.inv(A))
    # we also can't inherently use the inverse
    # however we can make a custom wrapper that
    # strips the units, does the inverse, and then puts the
    # units back on
except Exception as e:
    print(e)

# how does unum handle conversions
#a = 4*u.m + 6*u.ft # well theres another problem.. no SAE units. We'll have to define them
a = 4*u.m + 6*u.cm
print(a)
repr(a)
print(a.asUnit(u.m))
b = 4*u.cm / (8*u.s**2)
print(b)
repr(u.cm)
print(u.cm.normalize())


# %% units
from units import unit
a = unit('m')(3)
b = unit('m')(4)
# I just don't like this syntax.. so we're not gonna use it
# From prior testing it seems to operate very similar to unum