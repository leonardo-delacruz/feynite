# -*- coding: utf-8 -*-
"""num_evaluator.py -- the WHOLE-NUMERATOR
"""

import sympy as sp

from .constants import z
from .general_tools import _apply_kinematics, _kin_early
from .symanzik import (
    UFdata, _map_ufdata, adjA, generate_tensor_structures, _uncontract_term,
    _polst_highest_rank, _pipeline_setup,
)
from .sparse_tools import (
    # the shared refusal contract of the pure sparse route
    SparseRouteError,
    # the sparse accumulate / emit helpers
    _sparse_multiplication, _sparse_to_sympy, _slot_to_arrays, _sparse_zdict
)
from .sparse_evaluator import SparseNumEvaluator


# ---------------------------------------------------------------------------
# Numerator routes (module level)
# ---------------------------------------------------------------------------
def _build_evaluator(dens, kin, no_gauge, rmax):
    edges = len(dens[0])
    nprops = edges
    loops = len(dens[2])
    uf_raw = UFdata(dens, kin)
    z_e = z(edges)
    if no_gauge:
        uf = _map_ufdata(uf_raw, lambda e: sp.expand(_apply_kinematics(e, kin)))
    else:
        uf = _map_ufdata(uf_raw,
                         lambda e: sp.expand(_apply_kinematics(e.subs(z_e, 1),
                                                        kin)))
    datas = (dens, uf, adjA(uf[3]))
    tensors = [generate_tensor_structures(i) for i in range(rmax + 1)]
    return SparseNumEvaluator(datas, rmax, tensors, nprops, loops,
                                  kin, no_gauge, edges)


def _accumulate_terms(ev, dataNum, on_error='raise', verbose=True):
    """Grouped (a_exp, b, c_def) accumulation over ALL numerator terms.
    Mirrors the hybrid `_allTermsNum_sparse_dicts` exactly -- one shared
    evaluator, per-group deferred U_POL^a F_POL^b / c_def application -- with
    the pure refusal policy: a term whose term_parts raises makes the
    whole call raise (on_error='raise') or is PRINTED and SKIPPED
    (on_error='skip'); skipped terms NEVER contribute through a hidden
    second engine.

    Returns (totals, failures): totals = one sparse dict per slot
    (covering ONLY the accepted terms), failures = [(coeff, labels, err)]
    of the refused terms."""
    coeffs, labels = dataNum
    n_hi = len(ev.tensorsUptoHighest[ev.highestRank][0])
    totals = [{} for _ in range(n_hi)]
    groups = {}                          # (a_exp, b, c_def) -> [slot dicts]
    failures = []
    for c, lab in zip(coeffs, labels):
        try:
            parts = ev.term_parts(c, lab, sparse=True)
        except SparseRouteError as e:
            if on_error == 'raise':
                raise
            if verbose:
                print(e)
                print('  -> term skipped; the totals below cover the '
                      'accepted terms only')
            failures.append((c, lab, e))
            continue
        for s, p in enumerate(parts):
            if p[0] == 'final':
                v = p[1]
                if v:                    # exact-zero dict in the pure route
                    dst = totals[s]
                    for k, cf in v.items():
                        dst[k] = dst.get(k, 0) + cf
                continue
            _, acc, csp, a_exp, b, c_def = p
            if len(csp) == 1 and 0 in csp and csp[0] == 1:
                core = acc
            else:
                core = _sparse_multiplication(acc, csp)
            g = groups.get((a_exp, b, c_def))
            if g is None:
                g = groups[(a_exp, b, c_def)] = [{} for _ in range(n_hi)]
            dst = g[s]
            for k, cf in core.items():
                dst[k] = dst.get(k, 0) + cf
    for (a_exp, b, c_def), gd in groups.items():
        pab = ev._pab(a_exp, b)
        for s in range(n_hi):
            g = gd[s]
            if not g:
                continue
            dst = totals[s]
            for k, cf in _sparse_multiplication(g, pab).items():
                if c_def != 1:
                    cf = cf * c_def
                    if cf == 0:
                        continue
                dst[k] = dst.get(k, 0) + cf
    return totals, failures


def all_terms(num, dens, kin=None, no_gauge=True, on_error='raise',
              verbose=True):
    """The WHOLE numerator through the pure sparse per-term pipeline.
    Runs `_pipeline_setup`, builds ONE `SparseNumEvaluator` and
    accumulates every term with the shared (a_exp, b, c_def) grouping --
    the pure counterpart of the hybrid's batched sparse accumulator.

    on_error : 'raise' -> the first SparseRouteError propagates;
               'skip'  -> every refusal is printed and the term skipped
               (totals then cover the accepted terms only -- refused
               terms NEVER fall back to a classic engine).
    Returns (totals, failures): per-slot sparse dicts + the
    [(coeff, labels, err)] list of refused terms (empty under 'raise')."""
    dataNum, datas, highestRank, edges, tensorsUptoHighest, nprops, loops = \
        _pipeline_setup(num, dens, kin, no_gauge)
    ev = SparseNumEvaluator(datas, highestRank, tensorsUptoHighest,
                                nprops, loops, kin, no_gauge, edges)
    return _accumulate_terms(ev, dataNum, on_error=on_error,
                             verbose=verbose)


def pol_numerator(num, dens, kin=None, no_gauge=True, sparse=True,
                  highest_rank=None, slot=None, output=None, syms=None,
                  exact=True, subs=None, on_error='raise', verbose=True,
                  refusals=None, ev=None):
    """
    Processes the WHOLE numerator monomial by monomial through ONE
    `SparseNumEvaluator`, with the three big-numerator optimizations
      1. every piece is accumulated in the lean sparse-dict
         representation {mono-key: coeff}; SymPy expressions are built
         only ONCE per requested slot at the end;
      2. monomials that share the structure index and the same
         U_POL^a F_POL^b / deferred-scalar factorization are merged into one
         core BEFORE U_POL^a F_POL^b is applied (exact by distributivity);
      3. the ufdata/evaluator is built once for the whole numerator
         (pass `ev=` to reuse one across calls).
    Arguments
    -----------------------------------------
    num, dens, kin, no_gauge : the problem (kin applied early when the
        rules are loop-momentum-free, exactly as the hybrid).
    sparse : True -> per-slot sparse dicts; False -> expanded SymPy
        polynomials in z(1)..z(edges) (no_gauge=True) or
        z(1)..z(edges-1) (no_gauge=False) emitted from the same dicts
        (the sparse flag never changes the arithmetic here -- the pure
        route has no second engine to switch to).
    highest_rank : override the tensor rank rmax derived from num.
    slot : None -> all structures of rmax; integer s -> structure s only.
    output : None, or 'arrays' -> per slot the plain Python pair
        (coeffs, exps) of z-exponent rows (sp.Poly lex order, exact
        SymPy coefficients or floats with exact=False; the pure route
        has no classic extras, so the rows are always the direct
        sparse-dict projection).
    syms, exact, subs : as in the hybrid (output='arrays' only).
    on_error, verbose, refusals, ev : the pure-route additions above.

    Returns: one polynomial per requested slot -- exactly the hybrid's
    shapes whenever the hybrid does not fall back)."""
    kin = kin or {}
    if output not in (None, 'arrays'):
        raise ValueError(
            "pol_numerator: output must be None or 'arrays', got %r"
            % (output,))
    want_arrays = output == 'arrays'
    edges = len(dens[0])
    if want_arrays:
        if syms is None:
            zall = [z(i + 1) for i in range(edges)]
            syms_l = zall if no_gauge else zall[:-1]
        else:
            syms_l = [sp.sympify(s) for s in syms]
    num_ex = sp.expand(sp.sympify(num))
    if highest_rank is not None:
        rmax = int(highest_rank)
    elif num_ex != 0:
        rmax = _polst_highest_rank(num_ex)
    else:
        rmax = 0
    if ev is None:
        ev = _build_evaluator(dens, kin, no_gauge, rmax)
    nst_out = len(ev.tensorsUptoHighest[rmax][0])
    if slot is not None and not 0 <= int(slot) < nst_out:
        raise ValueError(
            'pol_numerator: slot %d out of range 0..%d (rmax=%d has %d '
            'tensor structures)' % (slot, nst_out - 1, rmax, nst_out))
    sel = None if slot is None else int(slot)
    early = _kin_early(kin, list(dens[2]))

    groups = {}                            # (si, a_exp, b, c_def) -> core
    for m in sp.Add.make_args(num_ex):
        mono = _apply_kinematics(m, kin) if early else m
        coeff, labels = _uncontract_term(mono)
        r = len(labels)
        if r > rmax:
            raise ValueError(
                'pol_numerator: monomial rank %d exceeds the highest '
                'rank %d of the numerator' % (r, rmax))
        nst = len(ev.tensorsUptoHighest[r][0])
        try:
            parts = ev.term_parts(coeff, labels, sparse=True)
        except SparseRouteError as e:
            if on_error == 'raise':
                raise
            if verbose:
                print(e)
                print('  -> monomial skipped; the slot polynomials below '
                      'cover the accepted monomials only')
            if refusals is not None:
                refusals.append((m, e))
            continue
        for si in range(nst):
            if sel is not None and si != sel:
                continue
            p = parts[si]
            if p[0] == 'final':
                # exact zeros, nothing to accumulate
                continue
            _tag, acc, csp, a_exp, b, c_def = p
            if len(csp) == 1 and 0 in csp and csp[0] == 1:
                core = acc
            else:
                core = _sparse_multiplication(acc, csp)
            gkey = (si, a_exp, b, c_def)
            g = groups.get(gkey)
            if g is None:
                groups[gkey] = g = {}
            for k, c in core.items():
                g[k] = g.get(k, 0) + c

    out = []
    for si in range(nst_out):
        if sel is not None and si != sel:
            continue
        d_acc = {}
        for (sj, a_exp, b, c_def), g in groups.items():
            if sj != si:
                continue
            prod = _sparse_multiplication(g, ev._pab(a_exp, b))
            if c_def == 1:
                for k, c in prod.items():
                    d_acc[k] = d_acc.get(k, 0) + c
            else:
                for k, c in prod.items():
                    v = c * c_def
                    if v != 0:
                        d_acc[k] = d_acc.get(k, 0) + v
        d_acc = {k: c for k, c in d_acc.items() if c != 0}
        if want_arrays:
            out.append(_slot_to_arrays(d_acc, None, syms_l, exact, subs))
        elif sparse:
            out.append(d_acc)
        else:
            out.append(_sparse_to_sympy(d_acc))
    if sel is not None:
        return out[0]
    return out


def write_polynomial_sum(num, dens, kin=None, path=None, no_gauge=True,
                       highest_rank=None, slot=None, syms=None):
    """write_polynomial_sum -- OOM-safe text export of the summed slot
    polynomial(s) of the whole numerator. Huge numerators must be stored
    carefully. The text is produced straight from the lean sparse dicts, never
    touching a big SymPy expression.
    Arguments
    ---------
    num, dens, kin, no_gauge, highest_rank : exactly as in `polNumerator`.
    slot : None (default) -> the SUM over all tensor-structure slots,
        i.e. the polynomial str(sum(polNumerator(..., sparse=False)))
        represents.  An integer s -> only structure s.
    syms : the polynomial variables; default [z(1)..z(edges)] for
        no_gauge=True and [z(1)..z(edges-1)] for no_gauge=False -- the
        same defaults as polNumerator(output="arrays").
    path : output file name.  If None, no file is written and the term
        strings are returned as a GENERATOR (still streamed -- join or
        consume incrementally, never sp.list(sp.Add)).

    Returns
    -------
    The number of terms written (0 for a vanishing polynomial; the file
    then contains the single character 0).

    Notes
    -----
    * The file contains ONE line: the terms joined by ' + ', each in the
      ordinary SymPy printer format -- read back with sympify(text,
      locals={'z': z, ...}) in Python or ToExpression[...] in
      Mathematica
    """
    edges = len(dens[0])
    if syms is None:
        zall = [z(i + 1) for i in range(edges)]
        syms_l = zall if no_gauge else zall[:-1]
    else:
        syms_l = [sp.sympify(s) for s in syms]

    res = pol_numerator(num, dens, kin, no_gauge=no_gauge, sparse=True,
                       highest_rank=highest_rank, slot=slot)
    dicts = [res] if slot is not None else res
    tot = {}
    for D in dicts:
        for k, c in D.items():
            tot[k] = tot.get(k, 0) + c
    tot = {k: c for k, c in tot.items() if c != 0}
    zd = _sparse_zdict(tot, syms_l)

    def _terms():
        for row in sorted(zd):
            yield str(sp.Mul(zd[row],
                             *[s ** int(e) for s, e in zip(syms_l, row)
                               if e]))

    if path is None:
        return _terms()
    n = 0
    with open(path, 'w') as f:
        first = True
        for t in _terms():
            f.write(t if first else ' + ' + t)
            first = False
            n += 1
        if first:
            f.write('0')
    return n