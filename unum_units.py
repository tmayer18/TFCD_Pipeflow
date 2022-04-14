# MAE 413 - TFCD
# Project 2
# Timothy Mayer
# 4/10/2022

# An extension of the unum unit-handling package, implementing some missing features needed for this code

# We've chosen unum over more popular packages like pint because the linear-algebra we're doing to iteratively solve
# requires arrays with units in them. Pint does not allow for that, though it is the far nicer package. We instead will
# suffer with the less developed unum library with does let quantities with different units exist in the same matrix

# P.S. I have since realized that this library is very very poorly written - there is lots of bad code I now have to deal with here...
# almost not worth it to deal with but sunken-cost is too much. I've made my bed, time to lie in it

# P.S.S. This class contains alot of terrible python code and bad practices - please please don't use this as an example for how to
# do actual programming, nor should this really be used anywhere else. If you're looking for the same functionality, look into
# making a proper forked version of the original Unum package and do actual module packaging stuff

from unum import Unum, IncompatibleUnitsError
import unum.units as u
import numpy as np

class UnitError(TypeError):
    '''An operation has a problem with units'''
    # a reimplementation because none of the Unum exceptions let you pass an arbitrary string
    def __init__(self, *args):
        super().__init__(self, *args)

class Unum2(Unum):
    '''Our modified version of Unum, implementing required
    features missing from the original library'''

    _base_unit_table = {}
    # loops through existing Unum units looking for base units to register
    for u_key, u_val in Unum.getUnitTable().items():
        if u_val[0] is None: # if no conversion is present, its a base unit
            _base_unit_table[u_key] = getattr(u, u_key) # store a reference to the unit object

    def __init__(self, unit, value=1, conv=None, name=''):
        # pass init call to base class, registering it in their unit table
        if isinstance(conv, int) and conv == 0: # is it a base unit?
            unit_key = list(unit.keys())[0]
            Unum2._base_unit_table[unit_key] = self # register to the table
        super().__init__(unit, value, conv, name)

    @classmethod
    def get_base_unit_table(cls):
        return cls._base_unit_table.copy()

    def as_base_unit(self):
        '''Sends the Unum2 object to base units, using registered list of conversions'''
        target_unit = 1
        for unit_key, unit_pow in self._unit.items(): # for each component unit, if there are multiple
            if unit_key in Unum2._base_unit_table:
                # we're already a base unit
                target_unit *= Unum2._base_unit_table[unit_key]**unit_pow
            else:
                # otherwise recursively retrieve lineage of units, starting with the next findable in the unit table
                parent_unit = Unum2(Unum2.getUnitTable()[unit_key][0]._unit)**unit_pow
                target_unit *= parent_unit.as_base_unit() # if rec_expand is a base unit, recursive call will return target_unit=self
        
        if isinstance(target_unit, Unum): # if this isn't a unitless quantity
            target_unit._value = 1 # overwrite conversion factor, as_unit will handle the conversion factors
        return self.asUnit(target_unit)

    @staticmethod
    @np.vectorize # decorator that make this a per-element numpy operation
    def arr_as_base_unit(num): # since we can't register a method call nparr.as_base_unit to the numpy array, we write a static method arr_as_base_unit(arr)
        if isinstance(num, Unum):
            return num.as_base_unit()
        return num

    @staticmethod
    @np.vectorize
    def arr_as_unit(num, other):
        return Unum.coerceToUnum(num).asUnit(other)

    @staticmethod
    @np.vectorize # decorator that make this a per-element numpy operation
    def strip_units(num): 
        if isinstance(num, Unum):
            return num.asNumber(), Unum2(num._unit)
        return num, 1

    @staticmethod
    @np.vectorize
    def arr_normalize(num):
        if isinstance(num, Unum):
            return num.normalize()
        return num

    def check_match_units(self, other): # checks if two numbers have compatiable units
        try:
            self.matchUnits(Unum.coerceToUnum(other)) # if this succeeds, there is a conversion
            return True
        except IncompatibleUnitsError:
            return False
    
    @staticmethod
    @np.vectorize
    def arr_check_unit_match(num, other_num):
        if isinstance(num, Unum):
            return num.check_match_units(other_num)
        return True # unitless numbers all have the same, nonexistant units


    @staticmethod
    def apply_padded_units(A,b,x):
        '''Takes the matrices of a matrix linear-equation set Ax=b, and injects missing units into coefficient matrix A
        - assembling multiple matrices into one leaves '0' entries that won't know what unit they require to survive inversion,
            this function replaces 0 scalars with <0 [some-unit]> Unum objects'''
        A_val, A_units = Unum2.strip_units(A)
        _, b_units = Unum2.strip_units(b)
        _, x_units = Unum2.strip_units(x)

        v_isinstance = np.vectorize(isinstance)

        expected_units = b_units@np.reciprocal(x_units).T # expected units of A based on solution and constant vector
        corrected_units = np.where(v_isinstance(A_units, Unum), A_units, expected_units) # where the original matrix had a non-Unum 1, inject the expected unit

        # check if expected match unchanged units
        check_units = np.where(v_isinstance(A_units, Unum), expected_units, A_units)
        match_array = Unum2.arr_check_unit_match(check_units, A_units)
        if not np.all(match_array): # a prexisting unit was incorrect
            ii, jj = np.where(np.logical_not(match_array))
            errored_elements = [(i,j) for i, j in zip(ii,jj)]
            raise UnitError(f"apply_padded_units has discovered input units that will not produce the desired output units, in cells {errored_elements}")

        return A_val*corrected_units

    @staticmethod
    def unit_aware_inv(A):
        '''performs linear-algebra matrix inversion, with unit-aware operations
         * all elements must have appropiate units, including zeros, as matrix-inversion operations may change zeros to numbers'''
        A = Unum2.arr_as_base_unit(A) # drop to consistent base-units
        A_vals, A_units = Unum2.strip_units(A)

        Ainv_vals = np.linalg.inv(A_vals)
        Ainv_units = np.reciprocal(A_units).T # when all the units are consistent, the inverse matrix's units are the Transpose of each units recripocal

        return Ainv_vals * Ainv_units

    # now we wrap the methods of Unum1 that explicitly return a Unum1 object to now return Unum2 objects
    _upgrade_methods = ('__add__', '__sub__', '__mul__', '__div__', '__truediv__',
        '__floordiv__', '__pow__', '__abs__', '__radd__', '__rsub__', '__rmul__',
        '__rdiv__', '__rfloordiv__', '__rtruediv__', '__rpow__', '__getitem__', '__setitem__',
        '__pos__', '__neg__', '__mod__')
    ignores_zero = ('__add__', '__sub__', '__radd__', '__rsub__') # to use identity matrices, adding zero should not change/set any units (0kg + 4m = 4m)
    for method_str in _upgrade_methods:
        def wrapped_method(self, *args, method_str=method_str, ignores_zero=ignores_zero):
            # if method_str in ['__add__', '__mul__']:
            #     print(f"unum math {method_str} : {self}, {args}")
            if method_str in ignores_zero:
                if Unum2.coerceToUnum(args[0]).asNumber() == 0:
                    args = (Unum2(self._unit, value=0),)+args[1:] # set incoming 0 to have matching units
                if self.asNumber() == 0:
                    self = Unum2(Unum.coerceToUnum(args[0])._unit, value=0) # set self to match units of incoming unum

            ret_obj = getattr(super(), method_str)(*args) # get the __ method from the super class
            return Unum2(ret_obj._unit, ret_obj._value) # recast as a Unum2
        vars()[method_str] = wrapped_method # add to this classes attributes, dynamicaly by name

    # def __repr__(self):
    #     return f"<Unum2 object({self._value}, {self._unit})>"

class units2():
    '''upconvert all predefined Unum1 units to Unum2, but as a class instead of the
    import bullcrap Unum1 did'''
    # note, that since we're dynamically generating these attributes, the linter will fail to show them properly, so we'll manually set their names here to avoid
    # errors in the linting process
    m, Ym, Zm, Em, Pm, Tm, Gm, Mm, km, hm, dam, ym, zm, am, fm, pm, nm, um, mm, cm, dm, s, Ys, Zs, Es, Ps, Ts, Gs, Ms, ks, hs, das, ys, zs, fs, ps, ns, us, ms, cs, ds, A, YA, ZA, EA, PA, TA, GA, MA, kA, hA, daA, yA, zA, aA, fA, pA, nA, uA, mA, cA, dA, K, YK, ZK, EK, PK, TK, GK, MK, kK, hK, daK, yK, zK, aK, fK, pK, nK, uK, mK, cK, dK, mol, Ymol, Zmol, Emol, Pmol, Tmol, Gmol, Mmol, kmol, hmol, damol, ymol, zmol, amol, fmol, pmol, nmol, umol, mmol, cmol, dmol, cd, Ycd, Zcd, Ecd, Pcd, Tcd, Gcd, Mcd, kcd, hcd, dacd, ycd, zcd, acd, fcd, pcd, ncd, ucd, mcd, ccd, dcd, kg, Yg, Zg, Eg, Pg, Tg, Gg, Mg, hg, dag, yg, zg, ag, fg, pg, ng, ug, mg, cg, dg, g, rad, sr, Hz, N, Pa, J, W, C, V, F, ohm, S, Wb, T, H, celsius, lm, lx, Bq, Gy, Sv, kat, min, h, d, deg, L, t, Np, dB, eV, ua, mile, nmile, knot, a, ha, bar, angstrom, b, Ci, R, rem = [None]*189
    
    exceptions = {"deg C":"celsius", "nmi":"nmile"}
    ignores = ["u", "'", "''"]
    unum1_table = Unum.getUnitTable()
    Unum.reset() # now reset the base class table so we can reregister as Unum2
    for symbol, (conv, _, name) in unum1_table.items():
        if conv is None:
            conv = 0 # for some reason, 0 indicates base unit, not None
        if symbol in exceptions:
            symbol = exceptions[symbol] # handle naming inconsistencies
        if symbol not in ignores:
            # print(f"{symbol}", end=', ') # print out full list of units for above line
            vars()[symbol] = Unum2.unit(symbol, conv, name)

    # define new units that the source library does not have
    ul = Unum2({}) # unitless quantity


if __name__ == "__main__":
    print("\n====Main====\n")

    def repr_override(obj):
        return f"<Unum object({obj._value}, {obj._unit})>"
    Unum.__repr__ = repr_override

    def list_str(value): # recursive str caller
        if not isinstance(value, (list, tuple, np.ndarray)): return str(value)
        return [list_str(v) for v in value]

    def lprint(thing):
        #prints thing but with nested __str__ calls rather than __repr__ for pretty printing
        print([list_str(item) for item in thing])


    Teic = Unum2.unit('tei', 0, 'teichert')
    # lTeic = Unum2.unit('ltei', 1e-3*Teic, 'lil teichert')
    # hlTeic = Unum2.unit('hltei', 0.5*lTeic, 'half lil teichert')
    # hhlTeic = Unum2.unit('hhlTeic', 300*hlTeic, 'hip half lil teichert')
    # Can = Unum2.unit('can', 0, "canino")
    # gen = Unum2.unit('gen', 0.5*lTeic*1.2*Can, 'genius')

    u = units2

    test_A = np.array([
        [1*u.ul, -1*u.ul, 4.3e6/(u.m*u.s), -13e6/(u.m*u.s)],
        [0*u.m*u.s, 0*u.m*u.s, 1*u.ul, -1*u.ul],
        [1*u.ul, 0, 0, 0],
        [0, 0, 1*u.ul, 0]])
    test_b = np.array([[-439000*u.kg/(u.m*u.s**2), 0*u.kg/u.s, 2304*u.Pa, 0.1*u.kg/u.s]]).T
    test_x = np.array([[0.1*u.Pa, 0.1*u.Pa, 0.1*u.kg/u.s, 0.1*u.kg/u.s]]).T
    print(Unum2.apply_padded_units(test_A,test_b,test_x))


    # a = np.array([[2,3],[4,5]])
    # a1 = np.array([[u.m, u.m**2/(u.s**2)],[u.s**2/u.m, u.ul]])
    # A = a*a1
    # print(A)
    # print(A@np.eye(2,2))
    # print(Unum2.unit_aware_inv(A))
    # print(Unum2.unit_aware_inv(a))

    # A = np.array([
    #     [2*u.m**2/(u.s**2),     2*u.N,      3*u.N*u.m/u.s,      4*u.m],
    #     [5/(u.s**2*u.m),        6*u.N/(u.m**3), 7*u.kg/(u.s**3*u.m), 8/(u.m**2)],
    #     [9*u.m/(u.kg*u.s**2),   10/(u.s**2),    12*u.m/(u.s**3),    11/u.kg],
    #     [13/(u.kg*u.s),     14/(u.m*u.s),   15/(u.s**2),    16/(u.N*u.s)]
    # ])
    # b = np.array([[1*u.N*u.m,    2*u.N/(u.m**2),     3*u.m/(u.s**2),  4/u.s]]).T

    # print(A)
    # print(b)
    # Aul, _ = Unum2.strip_units(A)
    # print(Aul)
    # print(np.linalg.inv(Aul))
    # Ainv = Unum2.unit_aware_inv(A)
    # print(Unum2.arr_normalize(Ainv)) # normalize won't get kgm/s2 -> N
    # x = Ainv@b
    # print(x)
    
    

