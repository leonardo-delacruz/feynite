# -*- coding: utf-8 -*-
"""general_tools.py
mostly rule applications

 linear-term splits, //. kin and Gamma helpers.

"  Linear-combination helper"        (_ckey, _linear_terms)
"  Rule helpers:  //. kin, Gamma reduction, Gamma -> 1"
                                           (_apply_kinematics, _kin_early,
                                            _gamma_numeric_split,
                                            gamma_reduce, _drop_gamma)
"""
import sympy as sp

# ---------------------------------------------------------------------------
# Linear-combination helper
# ---------------------------------------------------------------------------

def _ckey(x):
    """Deterministic sort key for canonical argument order."""
    return sp.default_sort_key(x)


def _linear_terms(x):
    """Split x into [(coeff, atom)] treating x as a linear combination.
    Returns None when x is genuinely non-linear (e.g. contains a Pow).
    """
    x = sp.sympify(x)
    if x.is_Number:
        return [(sp.S.One, x)]
    if isinstance(x, sp.Add):
        out = []
        for t in x.args:
            sub = _linear_terms(t)
            if sub is None:
                return None
            out.extend(sub)
        return out
    if isinstance(x, sp.Mul):
        c, rest = x.as_coeff_Mul()
        if c == sp.S.One:
            return [(sp.S.One, x)]
        if isinstance(rest, sp.Add):
            sub = _linear_terms(rest)
            if sub is None:
                return None
            return [(c * cc, aa) for cc, aa in sub]
        return [(c, rest)]
    if isinstance(x, sp.Pow):
        return None
    return [(sp.S.One, x)]


# ---------------------------------------------------------------------------
# Rule helpers:  //. kin,  Gamma reduction,  Gamma -> 1
# ---------------------------------------------------------------------------

def _apply_kinematics(expr, kin, max_iter=25):
    """Emulate  expr //. kin  for a dict (or list of pairs) of replacements.
    NOTE: build the keys with the same constructors (e.g. dL(pmom(1), pmom(1)))
    so that the canonical argument order matches automatically.
    """
    if not kin:
        return expr
    rules = list(kin.items()) if isinstance(kin, dict) else list(kin)
    keys = [sp.sympify(a) for a, _b in rules]
    if not expr.has(*keys):
        # no rule LHS occurs anywhere in expr: the first subs pass would be
        # a no-op, hence the fixpoint is expr itself.
        return expr
    prev = expr
    for i in range(max_iter):
        new = prev.subs(rules, simultaneous=True)
        if new == prev:
            break
        prev = new
        if i == 0 and not new.has(*keys):
            # After the first pass no rule LHS occurs anywhere in `new`,
            # so the second subs pass would be a no-op -- the //. has
            # already reached its fixpoint (same reasoning as the fast
            # path above).  Rules that fire only on results of other
            # rules still work: their LHS would occur in `new`.
            break
    return prev


def _kin_early(kin, loopMom):
    """True when `kin` may be applied BEFORE the loop-momentum
    differentiations inside UFdata and before toDataNum on the raw
    numerator, i.e. when no rule mentions a loop momentum k[i].  This is
    the standard external kinematics ({dL(P_a,P_b): Mandelstam, mass
    rules, ...}), for which the replacements commute with d/dk[i] and
    with k -> 0, so early application is EXACT.  Rules that do mention a
    loop momentum keep the original late-only route (the derivatives must
    see the raw invariants, as in the Mathematica program).
    """
    if not kin:
        return False
    items = list(kin.items()) if isinstance(kin, dict) else list(kin)
    return not any(sp.sympify(a).has(*loopMom) or sp.sympify(b).has(*loopMom)
                   for a, b in items)


def _gamma_numeric_split(arg):
    """Split a Gamma argument into (numeric part or None, symbolic rest)."""
    arg = sp.expand(arg)
    num = None
    rest = sp.S.Zero
    for t in sp.Add.make_args(arg):
        if t.is_number:
            num = (t if num is None else num + t)
        else:
            rest += t
    return num, rest


def gamma_reduce(expr, threshold, max_iter=200):
    """Emulate  expr //. Gamma[n_ + x_] /; n > threshold :> (n-1+x) Gamma[n-1+x]
    where n is the numeric part of the Gamma argument."""
    prev = expr
    for _ in range(max_iter):
        def _red(g):
            num, _rest = _gamma_numeric_split(g.args[0])
            if num is not None and num > threshold:
                return (g.args[0] - 1) * sp.gamma(g.args[0] - 1)
            return g
        new = prev.replace(lambda e: isinstance(e, sp.gamma), _red)
        if new == prev:
            return new
        prev = new
    return prev


def _drop_gamma(expr):
    """The final  /. Gamma[x_] :> 1: harmless trick"""
    return expr.replace(lambda e: isinstance(e, sp.gamma), lambda e: sp.S.One)





# ---------------------------------------------------------------------------
# public aliases
# ---------------------------------------------------------------------------
#: split x into [(coeff, atom)] treating x as a linear combination
linear_terms = _linear_terms
#: deterministic sort key for canonical (Orderless-like) argument order
ckey = _ckey
#: expr //. kin  (fast path when no rule LHS occurs)
apply_kin = _apply_kinematics
#: True when `kin` may be applied BEFORE the loop-momentum
#: differentiations (no rule mentions a loop momentum k[i])
kin_early = _kin_early
#: the final /. Gamma[x_] :> 1 of the pipeline
drop_gamma = _drop_gamma

__all__ = [
    'linear_terms', 'ckey', 'apply_kin', 'kin_early',
    'gamma_reduce', 'drop_gamma'
]