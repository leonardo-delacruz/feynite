"""constants.py
Symbols required to construct symbolic tensor structures
-------
    d         space-time dimension
    L         number of loops
    N_EDGES   number of propagators
    U_POL     placeholder for first Symanzik polynomial ufdata[0] (= det Mmatrix, factored)
    F_POL     placeholder for second Symanzik ufdata[1] (= fpol numerator)
    MU        bare Lorentz index carried by the Qvec entries

Heads / constructors
--------------------
    _kfun / kmom(i)     loop momentum k[i]
    _pfun / pmom(a)     external momentum P[a]
    _zfun / z(i)        Feynman parameter z[i]
    _adjfun             adjugate element adj[sigma_a, sigma_b]
    sigma(i)            symbolic propagator label (generate_tensor_structures)
    muidx(i)            positional Lorentz index mu[i]

Build numerators and kinematics ONLY with these constructors so every
module of the stack sees the SAME SymPy heads (two `class dL(sp.Function)`
definitions in different modules are DIFFERENT SymPy classes -- mixing
them silently breaks every substitution).
"""

import sympy as sp

# ---------------------------------------------------------------------------
# Global symbols and object constructors
# ---------------------------------------------------------------------------

d  = sp.Symbol('d')
L  = sp.Symbol('L')
N_EDGES = sp.Symbol('N_EDGES')
U_POL = sp.Symbol('U_POL')
F_POL = sp.Symbol('F_POL')
MU = sp.Symbol('mu')

_kfun   = sp.Function('k')
_pfun   = sp.Function('p')
_zfun   = sp.Function('z')
_adjfun = sp.Function('adj')


def kmom(i):
    """Loop momentum k[i]."""
    return _kfun(i)


def pmom(a):
    """External momentum P[a]."""
    return _pfun(a)


def z(i):
    """Feynman parameter z[i]."""
    return _zfun(i)


def sigma(i):
    """Symbolic propagator label used inside generate_tensor_structures. We need different indices"""
    return sp.Symbol('sigma%d' % i)


def muidx(i):
    """Positional Lorentz index mu[i] (i = 1..rank)."""
    return sp.Symbol('mu%d' % i)

__all__ = [
    'd', 'L', 'N_EDGES', 'U_POL', 'F_POL', 'MU',
    'kmom', 'pmom', 'z', 'sigma', 'muidx',
    '_kfun', '_pfun', '_zfun', '_adjfun'
]

