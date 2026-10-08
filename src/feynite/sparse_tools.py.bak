# -*- coding: utf-8 -*-
"""sparse_tools.py

IMPORTANT (single-registry contract): the atom registry
(_ATOM_RANK/_ATOM_BY_RANK) is append-only module state.  Packed keys are
only meaningful INSIDE this module's registry -- never mix sparse dicts
computed here with dicts from another copy of the engine (e.g. the
immutable ufdata_tensor.py): the rank -> atom tables are independent.
Cross-stack comparisons must normalise through `sp_emit` (or unpack both
sides with their own registries), as the verification scripts do.
"""

from fractions import Fraction

import sympy as sp

from .constants import U_POL, F_POL, N_EDGES, L, MU, _pfun, _kfun, _adjfun
from .lorentz import dL, lV, MT

# -- sparse polynomial helpers (rank-keyed monomials, exact coefficients) ----
#
# Monomial keys are ((rank, exp), ...) tuples with STRICTLY ASCENDING small
# integer ranks; the atom behind a rank lives in the append-only module
# registry _ATOM_BY_RANK (assigned on first sight).  Integer keys keep the
# products pure Python -- no SymPy sort keys / comparisons in the inner
# loops (the previous atom-keyed representation spent ~2/3 of the rank-4
# numSingleTerm runtime re-sorting SymPy atoms inside sorted()).

_ATOM_RANK = {}       # atom -> rank (first-seen order, never renumbered)
_ATOM_BY_RANK = []    # rank -> atom


def _atom_rank(a):
    """Small integer id of a monomial atom (assigned on first sight)."""
    r = _ATOM_RANK.get(a)
    if r is None:
        r = _ATOM_RANK[a] = len(_ATOM_BY_RANK)
        _ATOM_BY_RANK.append(a)
    return r


# ---------------------------------------------------------------------------
# A monomial key ((rank, exp), ...) with ascending ranks is represented as
# ONE Python integer:  key = Sum_i exp_i << (rank_i * _KEY_W).  Two keys
# MULTIPLY by integer ADDITION (exponent fields add, exactly the merge the
# tuple keys performed field by field), which turns the sparse product
# inner loop into a single machine-word-ish add + dict store instead of an
# O(len) tuple rebuild -- worth 1-2 orders of magnitude on the big
# group-apply products.  Zero fields are free, so the unit monomial is 0.
# _KEY_W bits per field bound the representable exponent; the pack guard
# rejects exponents far below that bound (any real workload stays below
# ~100), and more than _KEY_MAX_ATOMS distinct atoms falls back to the
# legacy expression pipeline via _SparseError.
_KEY_W = 20
_KEY_MASK = (1 << _KEY_W) - 1
_KEY_MAX_ATOMS = 96
_KEY_MAX_EXP = 1023

_UNPACK_MEMO = {}


def _pack_key(mono):
    """{rank: exponent} dict -> packed int key (raises _SparseError on
    out-of-range exponents / atom counts)."""
    key = 0
    for r, e in mono.items():
        if e:
            if r >= _KEY_MAX_ATOMS or e > _KEY_MAX_EXP:
                raise _SparseError(
                    'monomial key out of packed range: rank %d exp %d'
                    % (r, e))
            key += e << (r * _KEY_W)
    return key


def _unpack_key(k):
    """Packed int key -> ((rank, exp), ...) with ascending ranks."""
    t = _UNPACK_MEMO.get(k)
    if t is None:
        out = []
        rest = k
        while rest:
            low = (rest & -rest).bit_length() - 1
            r = low // _KEY_W
            e = (rest >> (r * _KEY_W)) & _KEY_MASK
            out.append((r, e))
            rest -= e << (r * _KEY_W)
        t = tuple(out)
        if len(_UNPACK_MEMO) < 2000000:
            _UNPACK_MEMO[k] = t
    return t



def _sp_from_expr(expr, fc_pow=None):
    """Expanded polynomial -> {mono-key: coeff} (None if not representable).

    Mono-keys are ((rank, exp), ...) tuples over the atom registry (see
    `_atom_rank`).  Every additive term must be a product of numbers and
    positive integer powers of monomial atoms (symbols, z(i) parameters,
    the dimension d, ...).  dL/lV/MT/gamma heads, unevaluated sums and the
    bookkeeping symbols U_POL/N_EDGES/L are rejected.  When `fc_pow` is an int,
    exactly that total F_POL-power is expected and stripped (F_POL is applied
    separately as the ufdata[[2]] polynomial)."""
    D = {}
    for t in sp.Add.make_args(sp.expand(expr)):
        coeff = Fraction(1)
        mono = {}
        fct = 0
        for f in sp.Mul.make_args(t):
            base, expo = f.as_base_exp()
            if fc_pow is not None and base is F_POL:
                if not (expo.is_Integer and expo > 0):
                    return None
                fct += int(expo)
            elif base.is_Number:
                if not f.is_Rational:
                    return None
                n, dv = f.as_numer_denom()
                coeff *= Fraction(int(n), int(dv))
            elif (expo.is_Integer and expo > 0
                    and not isinstance(base, (sp.Add, sp.Mul, sp.Pow))
                    and not base.has(dL, lV, MT, sp.gamma, U_POL, N_EDGES, L)
                    and getattr(base, 'func', None)
                    not in (_pfun, _kfun, _adjfun)):
                r = _atom_rank(base)
                mono[r] = mono.get(r, 0) + int(expo)
            else:
                return None
        if fc_pow is not None and fct != fc_pow:
            return None
        try:
            k = _pack_key(mono)
        except _SparseError:
            return None
        if k in D:
            D[k] += coeff
        else:
            D[k] = coeff
    return {m: c for m, c in D.items() if c != 0}


def _sparse_multiplication(A, B):
    """Product of two sparse polynomials {packed int mono-key: coeff}.

    Packed keys add exponents by INTEGER ADDITION (see `_pack_key`), so
    each product monomial is one big-int add + dict store -- no tuple
    rebuilds in the inner loop.  Coefficients are exact Python Fractions;
    they are multiplied only when neither factor is the (very common)
    unit -- `== 1` on a Fraction is an exact int comparison, orders of
    magnitude cheaper than a SymPy multiply."""
    out = {}
    for ka, ca in A.items():
        unit_a = (ca == 1)
        for kb, cb in B.items():
            k = ka + kb
            if unit_a:
                c = cb
            else:
                c = ca if cb == 1 else ca * cb
            if k in out:
                out[k] += c
            else:
                out[k] = c
    return {m: c for m, c in out.items() if c != 0}


def _sparse_add(A, B):
    """Sum of two sparse polynomials."""
    out = dict(A)
    for m, c in B.items():
        if m in out:
            out[m] += c
        else:
            out[m] = c
    return {m: c for m, c in out.items() if c != 0}


def _sp_drop_z(D, zedge):
    """Mono-level application of  z(edges) -> 1  (atom removal + merge)."""
    rz = _atom_rank(zedge)
    shift = rz * _KEY_W
    out = {}
    for k, c in D.items():
        e = (k >> shift) & _KEY_MASK
        k2 = k - (e << shift) if e else k
        if k2 in out:
            out[k2] += c
        else:
            out[k2] = c
    return {m: c for m, c in out.items() if c != 0}


class _SparseError(Exception):
    """Raised when a result cannot be represented as a sparse monomial
    dict (exotic input only).  Callers of the sparse pipeline catch it
    and fall back to the original expression-based code path."""

def _rat(c):
    """Fraction (or int) -> cached SymPy Rational.

    SymPy values pass through unchanged: sparse dicts pick up SymPy
    coefficients when a numerator term carries a scalar prefactor that
    the packed-key representation cannot hold (negative exponents such
    as 1/s23, non-rational numbers, ...).  The term is then evaluated by
    the fast route with the scalar deferred (see `term_parts`), and the
    emit paths multiply the symbolic scalar into the monomials here."""
    if isinstance(c, (Fraction, int)):
        return sp.Rational(c.numerator, c.denominator)
    return c


def _mono_expr(k):
    """Mono-key -> SymPy atom product, memoised (keys repeat heavily)."""
    e = _MONO_EXPR_CACHE.get(k)
    if e is None:
        e = sp.Mul(*[_ATOM_BY_RANK[r] ** ex for r, ex in k])
        _MONO_EXPR_CACHE[k] = e
    return e


_MONO_EXPR_CACHE = {}


def _sparse_to_sympy(D):
    """Sparse polynomial -> expanded sympy expression (canonical Add)."""
    if not D:
        return sp.S.Zero
    return sp.Add(*[_rat(c) * _mono_expr(_unpack_key(k))
                    for k, c in D.items()])


def _Qvec_vector_pieces(Qveci):
    """Qvec_i = Sum coeff*lV(vec, MU)  ->  [(coeff, vec)];  None if malformed.

    (Qvec entries are built as exactly such sums inside UFdata; the split
    only fails for exotic  input, in which case the caller falls back
    to the classic route.)"""
    if Qveci == 0:
        return []
    pieces = []
    for term in sp.Add.make_args(sp.expand(Qveci)):
        lv = None
        coeff = sp.S.One
        for f in sp.Mul.make_args(term):
            if isinstance(f, lV):
                if lv is not None or f.args[1] != MU:
                    return None
                lv = f
            else:
                coeff *= f
        if lv is None:
            return None
        if coeff != 0:
            pieces.append((coeff, lv.args[0]))
    return pieces


def _slot_to_arrays(d, extra, syms, exact, subs):
    """One accumulated slot dict (+ classic-route extras) -> the plain
    Python (coeffs, exps) pair of `_zdict_rows`, without emitting a
    SymPy expression when possible.

    `_sparse_zdict` projects the packed mono-keys onto the z-variables:
    z-powers become the exponent row, every OTHER atom (Mandelstam
    symbols, the dimension d, ...) folds into the coefficient factor --
    exactly the monomial `_sparse_to_sympy` + `sp.Poly` would build, so the rows
    and coefficients agree with the expression route."""
    if extra is not None:
        e = _sparse_to_sympy(d) + extra
        if subs is not None:
            e = e.subs(subs, simultaneous=True)
        return _poly_rows(sp.expand(e), syms, exact)
    zd = _sparse_zdict(d, syms)
    if subs is not None:
        # substitute AFTER the projection: the kinematic atoms folded in
        # from the mono-keys (d, s12, ...) are part of the coefficient
        # only from here on
        zd = {k: v.subs(subs, simultaneous=True) for k, v in zd.items()}
        zd = {k: v for k, v in zd.items() if v != 0}
    return _zdict_rows(zd, syms, exact)


def _float_coeffs(cs):
    """Coefficient list -> plain float (or complex) list (clear error on
    symbolic coefficients)"""
    try:
        vals = [complex(sp.N(c)) for c in cs]
    except (TypeError, ValueError):
        raise TypeError(
            'coefficients are symbolic; numeric output requires '
            'exact=False AND numeric coefficients (substitute numbers '
            'into the kinematics and d first)')
    if all(v.imag == 0 for v in vals):
        return [v.real for v in vals]
    return vals

def _zdict_rows(zd, syms, exact=True):
    """{z-exponent tuple: coeff} dict -> plain Python (coeffs, exps)."""
    rows = sorted(((tuple(int(e) for e in k), v)
                   for k, v in zd.items() if v != 0),
                  key=lambda kv: kv[0], reverse=True)
    if not rows:
        return [sp.S.Zero], [(0,) * len(syms)]
    cs = [r[1] for r in rows]
    exps = [r[0] for r in rows]
    if not exact:
        cs = _float_coeffs(cs)
    return cs, exps


def _poly_rows(expr, syms, exact=True):
    """Expanded expression -> plain Python (coeffs, exps) rows via
    sp.Poly """
    poly = sp.Poly(expr, *syms)
    cs = list(poly.coeffs())
    if not exact:
        cs = _float_coeffs(cs)
    return cs, [tuple(m) for m in poly.monoms()]

def _sparse_zdict(D, syms):
    """Sparse {mono-key: coeff} dict -> {z-exponent tuple: coeff} dict.

    The mono-keys are ((rank, exp), ...) tuples over the atom registry
    `_ATOM_BY_RANK` (see section 9).  Every atom that is one of `syms`
    contributes its exponent to the z-exponent row; every other atom
    (Mandelstam symbols produced by the kinematics, the dimension d,
    ...) folds into the coefficient as `atom**exponent` -- precisely the
    monomial the `_sparse_to_sympy` + `sp.Poly` route would form.  Multiple keys
    projecting onto the same z-row are summed (like terms)."""
    pos = {}
    for j, s in enumerate(syms):
        pos[s] = j
    nz = len(syms)
    rows = {}
    for k, c in D.items():
        ku = _unpack_key(k) if isinstance(k, int) else k
        exp = [0] * nz
        cf = sp.S.One
        for r, e in ku:
            a = _ATOM_BY_RANK[r]
            j = pos.get(a)
            if j is not None:
                exp[j] += e
            else:
                cf *= _rat(a) ** e
        v = _rat(c) * cf
        # Deferred scalars leave products like
        # s12**4*s23*(45/(8*s23) + 75/(8*s12)) whose like terms only
        # combine after expansion -- without it, rows that cancel
        # mathematically would survive as hidden zeros and the row set
        # would differ from the sp.Poly route.  Pure monomial
        # coefficients (the overwhelming majority) skip the expand.
        if v.has(sp.Add):
            v = sp.expand(v)
        rows.setdefault(tuple(exp), []).append(v)
    out = {}
    for t, terms in rows.items():
        if len(terms) == 1:
            s = terms[0]
        else:
            s = sp.Add(*terms)     # one collect pass; mathematically-zero
            if s == sp.S.Zero:             # rows disappear, as in sp.Poly
                continue
        out[t] = s
    return out


# ---------------------------------------------------------------------------
# the shared refusal contract of the sparse route
# ---------------------------------------------------------------------------
class SparseRouteError(Exception):
    """ This class handles exception by the sparse computation:
    Attributes
    ----------
    site : str
        Where the refusal happened -- one of
        'constructor', 'numerator-split', 'gauge-kin', 'gauge-term',
        'structure', 'piece'.
    reason : str
        Human-readable exact cause.
    momentaNumerator, setLoop, si, sparse :
        Optional context (the offending term, its loop-label word, the
        structure index, the sparse flag of the call).

        to be extended...
    """

    def __init__(self, site, reason, momentaNumerator=None, setLoop=None,
                 si=None, sparse=None):
        self.site = site
        self.reason = reason
        self.momentaNumerator = momentaNumerator
        self.setLoop = setLoop
        self.si = si
        self.sparse = sparse
        Exception.__init__(self, '%s: %s' % (site, reason))

    def __str__(self):
        lines = ['sparse route refuses at site %r' % (self.site,),
                 '  reason : %s' % (self.reason,)]
        if self.si is not None:
            lines.append('  structure index si = %s' % (self.si,))
        if self.setLoop is not None:
            lines.append('  loop labels: %s' % (self.setLoop,))
        if self.momentaNumerator is not None:
            lines.append('  numerator term: %s' % (self.momentaNumerator,))
        return '\n'.join(lines)

# ---------------------------------------------------------------------------
# public aliases -- the toolbox surface
# ---------------------------------------------------------------------------
#: expanded polynomial -> {packed mono-key: Fraction} (None if the
#: expression does not fit the sparse pattern; optional `fc_pow` strips
#: exactly that total F_POL power first)
sp_from_expr = _sp_from_expr
#: product of two sparse polynomials (packed keys add = one int add)
spmul = _sparse_multiplication
#: sum of two sparse polynomials
spadd = _sparse_add
#: mono-level application of z(edges) -> 1 (atom removal + key merge)
sp_drop_z = _sp_drop_z
#: sparse polynomial -> expanded SymPy expression (canonical Add)
sp_emit = _sparse_to_sympy
#: Qvec_i = Sum coeff*lV(vec, MU) -> [(coeff, vec)]; None if malformed
Qvec_vector_pieces = _Qvec_vector_pieces
#: one accumulated slot dict -> the plain (coeffs, exps) rows of the
#: polNumerator arrays route
slot_to_arrays = _slot_to_arrays

__all__ = [
    'sp_from_expr', 'spmul', 'spadd', 'sp_drop_z', 'sp_emit',
    'Qvec_vector_pieces', 'slot_to_arrays',
    'SparseRouteError',
    # Fraction is re-exported: the sparse coefficients ARE Fractions
    'Fraction',
]
