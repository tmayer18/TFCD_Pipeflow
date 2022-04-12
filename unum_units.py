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

from unum import Unum
import unum.units as u

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
        
        target_unit._value = 1 # overwrite conversion factor, as_unit will handle the conversion factors
        return self.as_unit(target_unit)

    # now we wrap the methods of Unum1 that explicitly return a Unum1 object to now return Unum2 objects
    _upgrade_methods = ['__add__', '__sub__', '__mul__', '__div__',
        '__floordiv__', '__pow__', '__abs__', '__radd__', '__rsub__', '__rmul__',
        '__rdiv__', '__rfloordiv__', '__rpow__', '__getitem__', '__setitem__',
        '__pos__', '__neg__']
    for method_str in _upgrade_methods:
        def wrapped_method(self, *args, method_str=method_str):
            ret_obj = getattr(super(), method_str)(*args)
            return Unum2(ret_obj._unit, ret_obj._value)
        vars()[method_str] = wrapped_method

    def __repr__(self):
        return f"<Unum2 object({self._value}, {self._unit})>"

    # replace Unum camelCase with snake_case
    as_number = Unum.asNumber
    as_unit = Unum.asUnit

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


if __name__ == "__main__":
    print("\n====Main====\n")

    def repr_override(obj):
        return f"<Unum object({obj._value}, {obj._unit})>"
    Unum.__repr__ = repr_override

    Teic = Unum2.unit('tei', 0, 'teichert')
    lTeic = Unum2.unit('ltei', 1e-3*Teic, 'lil teichert')
    hlTeic = Unum2.unit('hltei', 0.5*lTeic, 'half lil teichert')
    hhlTeic = Unum2.unit('hhlTeic', 300*hlTeic, 'hip half lil teichert')
    Can = Unum2.unit('can', 0, "canino")
    gen = Unum2.unit('gen', 0.5*lTeic*1.2*Can, 'genius')

    print(u)
    print(units2)
    print(units2.celsius)
    
    a = hhlTeic * Can * units2.mm
    b = units2.N
    c = -b
    d = +c
    e = a*b
    print(a)
    print(b.as_base_unit())
    print(c, d, e)

    
    

