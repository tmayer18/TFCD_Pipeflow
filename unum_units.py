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

class Repr2Str(str): # A version of string that always prints with the str method, written to print matrices in exception messages
    def __repr__(self): return str(self) # replace the repr with the str method

class UnitError(TypeError):
    '''An operation has a problem with units'''
    # a reimplementation because none of the Unum exceptions let you pass an arbitrary string
    def __init__(self, *args):
        args =tuple((Repr2Str(arg) for arg in args))
        super().__init__(self, *args)

# we'll need to np.vectorize a bunch of methods so they work on all elements of a matrix.
#   Since numpy likes enforcing data types (its secretly C), it assumes the data type of the first element applies to the whole matrix
#   Here we write a partially-bound version of the numpy.vectorize function, as a decorator, that forces the datatype to be floats
def unum_vectorize(otypes):
    def _wrapped_func(func):
        return np.vectorize(func, otypes=otypes)
    return _wrapped_func

# pylint: disable=protected-access,redefined-outer-name
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
    @unum_vectorize(otypes=[object]) # decorator that make this a per-element numpy operation
    def arr_as_base_unit(num): # since we can't register a method call nparr.as_base_unit to the numpy array, we write a static method arr_as_base_unit(arr)
        if isinstance(num, Unum):
            return num.as_base_unit()
        return num

    @staticmethod
    @unum_vectorize(otypes=[object])
    def arr_as_unit(num, other):
        return Unum.coerceToUnum(num).asUnit(other)

    @staticmethod
    @unum_vectorize(otypes=[float, object])
    def strip_units(num):
        if isinstance(num, Unum):
            return num._value, Unum2(num._unit)
        return num, 1

    @staticmethod
    @unum_vectorize(otypes=[object])
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
    @unum_vectorize(otypes=[bool])
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

        units_matrix = Unum2.arr_normalize(b_units@np.reciprocal(x_units).T) # expected units of A based on solution and constant vector
        #FIXME prior line takes 100x the time of the rest of this function
        expected_units = Unum2.strip_units(units_matrix)[1]
        corrected_units = np.where(v_isinstance(A_units, Unum), A_units, expected_units) # where the original matrix had a non-Unum 1, inject the expected unit

        # check if expected match unchanged units
        check_units = np.where(v_isinstance(A_units, Unum), expected_units, A_units)
        match_array = Unum2.arr_check_unit_match(check_units, A_units)
        if not np.all(match_array): # a prexisting unit was incorrect
            ii, jj = np.where(np.logical_not(match_array))
            errored_elements = [(i,j) for i, j in zip(ii,jj)]
            raise UnitError(f"apply_padded_units has discovered input units that will not produce the desired output units, in cells {errored_elements} of matrix\n{A_units}")

        return A_val*corrected_units

    @staticmethod
    def unit_aware_inv(A):
        '''performs linear-algebra matrix inversion, with unit-aware operations
         * all elements must have appropiate units, including zeros, as matrix-inversion operations may change zeros to numbers'''
        A = Unum2.arr_as_base_unit(A) # drop to consistent base-units
        #FIXME as_base_unit takes 100x longer to run than anything else in this function
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
            # if method_str in ['__add__', '__mul__', '__radd__']:
            #     print(f"unum math {method_str} : {self}, {args} --> ", end='')
            if method_str in ignores_zero and not self.check_match_units(args[0]): # operations that should ignore zeros with units when they don't match
                other_is_zero = Unum2.coerceToUnum(args[0]).asNumber() == 0
                self_is_zero = self.asNumber() == 0
                if self_is_zero and other_is_zero: return 0 # anniahalite the units, hoping somewhere else a correction will be applied
                elif other_is_zero: return self # otherwise, keep the units of the non-zero
                elif self_is_zero:  return args[0]
                
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
    inch =  Unum2.unit('inch', 2.54*cm, 'inch')
    ft =    Unum2.unit('ft', 12*inch, 'foot')
    gal =   Unum2.unit('gal', 231*inch**3, 'gallon')
    lbm =   Unum2.unit('lbm', 453.59237*g, 'pound-mass')
    slug =  Unum2.unit('slug', 32.17404*lbm, 'slug')
    lbf =   Unum2.unit('lbf', 4.448222*N, 'pound-force')
    Btu =   Unum2.unit('Btu', 778.17*ft*lbf, 'british-thermal-unit')
    hp =    Unum2.unit('hp', 550*ft*lbf/s, 'horsepower')
    psi =   Unum2.unit('psi', lbf/(inch**2), 'pounds-per-square-inch')
    psf =   Unum2.unit('psf', lbf/(ft**2), 'pound-per-square-foot')
    atm =   Unum2.unit('atm', 14.696*psi, 'atmospheric-pressure')
    Rk =    Unum2.unit('Rk', K/1.8, 'Rankine')


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
    # num, unit = Unum2.strip_units(test_A)
    # print(num)
    # print(unit)
    test_b = np.array([[-439000*u.kg/(u.m*u.s**2), 0*u.kg/u.s, 2304*u.Pa, 0.1*u.kg/u.s]]).T
    test_x = np.array([[0.1*u.Pa, 0.1*u.Pa, 0.1*u.kg/u.s, 0.1*u.kg/u.s]]).T
    print(Unum2.apply_padded_units(test_A,test_b,test_x))
    print(Unum2.strip_units(test_A))
    print(Unum2.arr_as_base_unit(test_A))


    # test_A_2 = np.array([
    #     [1, 0, 0, 0], # for some god-forsaken-reason, at least one element in each row must have units, otherwise... numpy starts making up data? it thinks (int(1)) nas units of kg/s AFTER being called asNumber
    #     [1, -1, 7000/u.kg/u.s, -21000/u.K/u.s],
    #     [0*u.m*u.s, 0*u.m*u.N, 1, 1],
    #     [0, 1*u.ul, 0, 0]
    # ])
    # num, unit = Unum2.strip_units(test_A_2)
    # print(num)
    # print(unit)

    
    

