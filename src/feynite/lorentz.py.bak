# -*- coding: utf-8 -*-
"""lorentz.py
The Lorentz algebra: dL (dot product of Lorentz invariants), lV (lorentz Vectors),
 MT (metric), their contractions and the kinematics registry. dL and lV  do not
 follow Python naming conventions.

 Important: This is not a general purpose implementation of Lorentz algebra!

    "Kinematics registry"  (_KIN_REGISTRY, set_/clear_/get_kinematics)

    "Lorentz algebra"       (dL, lV, MT, dL_of, _contract_mul_once,
                                 _split_term_factors, _split_scalar_packable,
                                 _contract_factors, lorentz_simplify)

This is the Python analogue of the Mathematica TagSetDelayed rule base;
the Times-upvalue contraction rules are executed by `lorentz_simplify`.

IMPORTANT (single-class contract): build EVERY dL of a session through
THIS module.  A second `class dL(sp.Function)` definition (e.g. importing
the head from another engine copy) creates a DIFFERENT SymPy class;
kinematics keys built with it never match and it will lead to a sparse exception
"""

import sympy as sp
from sympy.core.function import ArgumentIndexError

from .constants import d, U_POL, N_EDGES, L, _pfun, _kfun, _adjfun
from .general_tools import _linear_terms, _ckey
# ---------------------------------------------------------------------------
#  Kinematics registry:  Python analogue of the Mathematica definitions
#          dL[p[i_], p[j_]] := s[i, j]/2;   dL[p[i_], p[i_]] := 0;   ...)
#  Observed in mathematica that setting these definitions from the beginning improve
#  performance: the equivalent path in Python is memoization so we use it
# ---------------------------------------------------------------------------


_KIN_REGISTRY = {}   # {(externalA, externalB): value}, canonical arg order

def _clear_sympy_cache():
    """Drop the SymPy Function-application caches so dL atoms re-evaluate
    under the registry (Function applications are memoised by SymPy).
    ONLY the `Application.__new__` / `Function.__new__` lru stores are
    cleared (every defined-Function application -- dL/lV/MT among them --
    is memoised there).  SymPy's GLOBAL cache wipe
    (sympy.core.cache.clear_cache()) must NOT be used: it also discards
    the `Symbol.__xnew_cached_` creation cache and the Pow/Integer
    caches, after which a Symbol created by any LATER import (e.g. the
    engine oracle's F_POL) is a NEW, equal-but-distinct object; the sparse
    layers classify monomial factors with IDENTITY checks (`base is F_POL`
    -- sparse_tools and the engine alike), so a global wipe silently
    degrades their fast paths (observed as hybrid 'final' tags in the
    parity checks when several test suites run in one process).  Registry
    semantics only need the Function-application caches dropped; object
    identity of symbols survives."""
    from sympy.core.function import Application, Function
    for _klass in (Application, Function):
        _wrapper = _klass.__dict__.get('__new__')
        _wrapper = getattr(_wrapper, '__func__', _wrapper)
        if _wrapper is not None and hasattr(_wrapper, 'cache_clear'):
            _wrapper.cache_clear()
        else:
            # fallback for a future SymPy with a different layout: the
            # global wipe (the registry still needs SOME cache drop)
            from sympy.core.cache import clear_cache
            clear_cache()
            return


def set_kinematics(kinematics:dict=None):
    """
    Register the external kinematics as EVALUATION RULES of `dL` invariants
    """
    _KIN_REGISTRY.clear()
    if kinematics:
        items = list(kinematics.items()) if isinstance(kinematics, dict) else list(kinematics)
        for a, v in items:
            a = sp.sympify(a)
            v = sp.sympify(v)
            if not (isinstance(a, dL) and len(a.args) == 2):
                raise ValueError('set_kinematics: key must be dL(x, y), got %s'
                                 % a)
            x, y = a.args
            if x.has(_kfun) or y.has(_kfun):
                raise ValueError(
                    'set_kinematics: %s contains a loop momentum k[i]; '
                    'loop invariants cannot be replaced' % a)
            if _ckey(y) < _ckey(x):
                x, y = y, x
            _KIN_REGISTRY[(x, y)] = v
    _clear_sympy_cache()

def clear_kinematics():
    """Remove all registered kinematics (back to symbolic invariants)."""
    set_kinematics(None)

def get_kinematics():
    """A copy of the currently registered kinematic replacements."""
    return dict(_KIN_REGISTRY)


# ---------------------------------------------------------------------------
# Lorentz algebra
# ---------------------------------------------------------------------------
class dL(sp.Function):
    r"""Symmetric invariant dL(a, b).
    Implements the Mathematica rules
        dL[sum_Plus, A__] / dL[A__, sum_Plus] : bilinear distribution,
        dL[Times[a_Integer, A__], X__]        : pull out numeric coefficients,
        dL[x_]  := dL[x, x],   dL[0, x_] := 0,
        Orderless                             : canonical argument order.
    Differentiation follows the MINT replacement rules
        Derivative[1,0][dL] -> #2&   and   Derivative[0,1][dL] -> #1&,
    i.e.  d dL(a,b)/da -> b  and  d dL(a,b)/db -> a   (see `fdiff`).
    """
    is_commutative = True
    @classmethod
    def eval(cls, *args):
        if len(args) == 1:
            a = b = args[0]
        elif len(args) == 2:
            a, b = args
        else:
            return None
        if a == 0 or b == 0:
            return sp.S.Zero
        a_terms = _linear_terms(a)
        b_terms = _linear_terms(b)
        if a_terms is None or b_terms is None:
            # exotic argument: keep it, but fix the order (symmetry)
            if _ckey(b) < _ckey(a):
                return cls(b, a)
            return None
        if (len(a_terms) == 1 and len(b_terms) == 1
                and a_terms[0][0] == 1 and b_terms[0][0] == 1):
            aa, bb = a_terms[0][1], b_terms[0][1]
            if _ckey(bb) < _ckey(aa):
                return cls(bb, aa)
            if _KIN_REGISTRY:
                val = _KIN_REGISTRY.get((aa, bb))
                if val is not None:
                    return val
            return None
        total = sp.S.Zero
        for ca, aa in a_terms:
            for cb, bb in b_terms:
                total += ca * cb * cls(aa, bb)
        return total

    def fdiff(self, argindex=1):
        # d/d(first argument)  -> the second argument, and vice versa
        if argindex == 1:
            return self.args[1]
        if argindex == 2:
            return self.args[0]
        raise ArgumentIndexError(self, argindex)

class lV(sp.Function):
    r"""Lorentz-vector  lV(vec, index).
    Implements  lV[sum_Plus, m] := lV[#,m]&/@sum,
                lV[Times[a_Integer, A], m] := a lV[A, m],
                lV[0, x_] := 0.
    The quadratic rules  lV[i,mu]^2 = dL[i,i]  and
    lV[i,mu] lV[j,mu] = dL[i,j]  are applied by `lorentz_simplify`.
    """
    is_commutative = True

    @classmethod
    def eval(cls, vec, idx):
        vec = sp.sympify(vec)
        if vec == sp.S.Zero:
            return sp.S.Zero
        terms = _linear_terms(vec)
        if terms is None:
            return None
        if len(terms) > 1 or terms[0][0] != 1:
            return sp.Add(*[c * cls(v, idx) for c, v in terms])
        return None


class MT(sp.Function):
    r"""Metric tensor MT(mu, nu) with  MT[mu_, mu_] := d  and Orderless."""
    is_commutative = True

    @classmethod
    def eval(cls, a, b):
        if a == b:
            return d
        if _ckey(b) < _ckey(a):
            return cls(b, a)
        return None

def dL_of(u, v):
    """dL between general (possibly summed) momenta: full bilinear expansion."""
    u = sp.expand(u)
    v = sp.expand(v)
    ut = _linear_terms(u)
    vt = _linear_terms(v)
    if ut is None or vt is None:
        return dL(u, v)
    total = sp.S.Zero
    for cu, uu in ut:
        for cv, vv in vt:
            total += cu * cv * dL(uu, vv)
    return total

def _contract_mul_once(mul):
    """Apply ONE contraction rule to a product; None if nothing applies.
    Rules (the Times/Power-upvalues of the Mathematica code):
        lV[i,mu] lV[j,mu]        -> dL(i, j)
        lV[i,mu]^2               -> dL(i, i)
        MT[mu,nu] lV[i,mu/nu]    -> lV(i, nu/mu)
        MT[mu,mu]                -> d            (done in MT.eval)
        MT[mu,nu] MT[mu,rho]     -> MT[nu,rho]   (all shared-index cases)
        MT[mu,nu]^2              -> d
    (NB: SymPy merges equal factors into a Pow, so the squared rules are
    essential -- they correspond to lV/: lV[i_,mu_]^2 and MT/: MT[mu_,mu_]^2.)
    """
    factors = list(sp.Mul.make_args(mul))
    n = len(factors)
    # Power rules first (Mathematica: lV[i_,mu_]^2 :> dL[i,i], MT[mu_,mu_]^2 :> d)
    for k, f in enumerate(factors):
        if isinstance(f, sp.Pow) and f.exp == sp.simpify(2):
            b = f.base
            if isinstance(b, lV):
                v = b.args[0]
                rest = factors[:k] + factors[k + 1:]
                return sp.Mul(*rest) * dL_of(v, v)
            if isinstance(b, MT):
                rest = factors[:k] + factors[k + 1:]
                return sp.Mul(*rest) * d
    for i in range(n):
        fi = factors[i]
        for j in range(i + 1, n):
            fj = factors[j]
            rest = [factors[t] for t in range(n) if t != i and t != j]
            # lV . lV with a common index
            if isinstance(fi, lV) and isinstance(fj, lV) and fi.args[1] == fj.args[1]:
                return sp.Mul(*rest) * dL_of(fi.args[0], fj.args[0])
            # MT . lV
            pair = None
            if isinstance(fi, MT) and isinstance(fj, lV):
                pair = (fi, fj)
            elif isinstance(fi, lV) and isinstance(fj, MT):
                pair = (fj, fi)
            if pair is not None:
                mt, lv = pair
                a, b = mt.args
                v, idx = lv.args
                if idx == a:
                    return sp.Mul(*rest) * lV(v, b)
                if idx == b:
                    return sp.Mul(*rest) * lV(v, a)
            # MT . MT
            if isinstance(fi, MT) and isinstance(fj, MT):
                a, b = fi.args
                c, e = fj.args
                if a == c:
                    shared = (b, e)
                elif a == e:
                    shared = (b, c)
                elif b == c:
                    shared = (a, e)
                elif b == e:
                    shared = (a, c)
                else:
                    shared = None
                if shared is not None:
                    return sp.Mul(*rest) * MT(*shared)
    return None


def _split_term_factors(t):
    """Split ONE additive term into (scalar, index-factor list).
    The index factors are ('M', (a, b)) for MT(a, b) and ('L', (vec, mu))
    for lV(vec, mu); integer powers are split into repeated factors.
    Everything else (numbers, z(i), Mandelstams, dL invariants, ...) goes
    into the scalar.  Returns None when a factor cannot be classified
    (non-integer powers of lV/MT, ...).
    """
    scalar = sp.S.One
    factors = []
    for f in sp.Mul.make_args(t):
        base, expo = f.as_base_exp()
        if isinstance(base, (lV, MT)):
            if not (expo.is_Integer and expo > 0):
                return None
            for _ in range(int(expo)):
                factors.append(('M' if isinstance(base, MT) else 'L',
                                base.args))
        else:
            scalar *= f
    return scalar, factors


def _split_scalar_packable(c):
    """Scalar prefactor -> (packable, deferred) with c == packable*deferred.
    `packable` holds the factors that `_sp_from_expr` accepts (rational
    number factors and positive-integer powers of registry atoms), so the
    sparse arithmetic on it stays on packed int keys and exact Fractions.
    The deferred rest (negative exponents such as 1/s23, non-rational
    numbers, unevaluated sums, ...) is a multiplicative CONSTANT of the
    term: callers keep it OUT of the hot sparse arithmetic and apply it
    ONCE to the final result instead. """
    pack = sp.S.One
    defer = sp.S.One
    for f in sp.Mul.make_args(sp.sympify(c)):
        base, expo = f.as_base_exp()
        if base.is_Number:
            if f.is_Rational:
                pack *= f
            else:
                defer *= f
            continue
            #We are interested only in z variables and kinematic invariants: scalars
        if (expo.is_Integer and expo > 0
                and not isinstance(base, (sp.Add, sp.Mul, sp.Pow))
                and not base.has(dL, lV, MT, sp.gamma, U_POL, N_EDGES, L)
                and getattr(base, 'func', None) not in (_pfun, _kfun,
                                                        _adjfun)):
            pack *= f
        else:
            defer *= f
    return pack, defer


def _contract_factors(factors):
    """Contract a list of index factors in ONE pass (no re-expansion).
       lV(v,mu) - (MT chain) - lV(w,mu)  ->  dL(v, w)      [dL_of]
        MT cycle / MT(mu,nu)^2            ->  d
        lV(v,mu)^2                        ->  dL(v, v)
     Returns the contracted product, or None when the term does not fit the pattern for example
     when an index occurs more than twice """
    occ = {}
    for fi, (kind, args) in enumerate(factors):
        mus = args if kind == 'M' else (args[1],)
        for slot, mu in enumerate(mus):
            occ.setdefault(mu, []).append((fi, slot))
    link = {}
    for lst in occ.values():
        if len(lst) == 2:
            (i, si), (j, sj) = lst
            link[(i, si)] = (j, sj)
            link[(j, sj)] = (i, si)
        elif len(lst) != 1:
            return None
    if any(len(lst) == 1 for lst in occ.values()):
        return None                      # free index: classic path
    seen = [False] * len(factors)
    result = sp.S.One
    for start in range(len(factors)):
        if seen[start]:
            continue
        comp = []
        stack = [start]
        seen[start] = True
        while stack:
            fi = stack.pop()
            comp.append(fi)
            kind, _args = factors[fi]
            for slot in (range(2) if kind == 'M' else (0,)):
                nxt = link.get((fi, slot))
                if nxt is not None and not seen[nxt[0]]:
                    seen[nxt[0]] = True
                    stack.append(nxt[0])
        lvs = [factors[fi][1][0] for fi in comp if factors[fi][0] == 'L']
        if len(lvs) == 2:
            result *= dL_of(lvs[0], lvs[1])
        elif len(lvs) == 0:
            result *= d                   # closed metric cycle: trace = d
        else:
            return None
    return result


def lorentz_simplify(expr):
    """Contract all Lorentz indices (expand + repeat rules).
    """
    expr = sp.expand(expr)
    out = []
    for t in sp.Add.make_args(expr):
        done = False
        if t.has(lV, MT):
            split = _split_term_factors(t)
            if split is not None:
                res = _contract_factors(split[1])
                if res is not None:
                    out.append(split[0] * res)
                    done = True
        if not done:
            stack = [t]
            while stack:
                t = stack.pop()
                nt = _contract_mul_once(t)
                if nt is None:
                    out.append(t)
                else:
                    stack.extend(sp.Add.make_args(sp.expand(nt)))
    return sp.Add(*out)


# ---------------------------------------------------------------------------
# rank inference
# ---------------------------------------------------------------------------
def _term_slots(expr, loops):
    """Loop-momentum slots of one multiplicative term."""
    if isinstance(expr, dL):
        return sum(1 for a in expr.args
                   if any(a.has(k) for k in loops))
    if isinstance(expr, sp.Pow):
        base = _term_slots(expr.base, loops)
        if base and expr.exp.is_Integer and expr.exp > 0:
            return int(expr.exp) * base
        return base
    if expr.args:
        return sum(_term_slots(a, loops) for a in expr.args)
    return 0


def infer_rank(num_expr, loops):
    """Tensor rank = max number of loop-momentum slots over all terms.

    dL(k(1), k(2)) -> 2, dL(P(1), k(2)) -> 1, dL(P(1), k(2))**2 -> 2,
    dL(P(1), k(1))*dL(P(1), k(2))*dL(P(2), k(2)) -> 3, dL(P, P) -> 0.
    Matches the library identity deg(num) = rank*loops after
    polNumerator (checked for the rank-2 and rank-3 double boxes).
    """
    expr = sp.expand(sp.sympify(num_expr) if isinstance(num_expr, str)
                     else num_expr)
    ranks = [_term_slots(t, loops) for t in sp.Add.make_args(expr)]
    return int(max(ranks)) if ranks else 0

# ---------------------------------------------------------------------------
# Public aliases
# ---------------------------------------------------------------------------
#: ONE additive term -> (scalar, [('M', mt) / ('L', lv), ...]); None if
#: a factor cannot be classified
split_term_factors = _split_term_factors
#: scalar prefactor -> (packable, deferred) with packable * deferred == c
split_scalar_packable = _split_scalar_packable

__all__ = [
    'dL', 'lV', 'MT', 'dL_of', 'lorentz_simplify',
    'set_kinematics', 'clear_kinematics', 'get_kinematics',
    'split_term_factors', 'split_scalar_packable',"infer_rank"
]