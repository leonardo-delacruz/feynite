# -*- coding: utf-8 -*-
"""symanzik.py -- the UFdata / Symanzik-polynomial and tensor-integrand to construct
the tensor representation of a general Feynman integral with numerators

The representration follows the conventions of the paper https://arxiv.org/pdf/2410.18014
by the author and in particular the idea of using Wick contractions as discussed in
Appendix A of the aforementioned paper

 UFdata"       (UFdata, Symanzik, adjA)
 Wick contractions and generate_tensor_structures"
                              (_perfect_matchings, WickContraction, wick2,
                               generate_tensor_structures)

index / unContractnumerator / ... / toDataNum"
                              (_loop_label, _check_dL_args,
                               _uncontract_term, unContractnumeratorList,
                               toDataNum)
bmat2 / cP3"         (bmat2, cP3)
    tensor_rank, _map_ufdata, _pipeline_setup (the shared pipeline
                               front end), _polst_highest_rank (memoised
                               tensor rank)
"""

from itertools import product

import sympy as sp

from .constants import (d, L, N_EDGES, U_POL, F_POL, MU, _kfun, _pfun, _adjfun, z, sigma, muidx)
from .general_tools import _apply_kinematics, _kin_early
from .lorentz import dL, lV, MT, lorentz_simplify

# ---------------------------------------------------------------------------
# UFdata -- translation of the previous Mathematica block
# ---------------------------------------------------------------------------
def UFdata(dens, kin=None):
    """UFdata[{momprop, masses, loopMom}] -> (detMmatrix, fpol, Qvec, Mmatrix).
    den  = Sum_i z[i] (dL[p_i, p_i] - m_i^2)
    t1   = -1/2 d den/dk            (d dL(a,b)/da -> b,  d dL(a,b)/db -> a)
    Mmatrix[j][i] = -d t1[i]/d k[j]      (the Gaussian quadratic form)
    Qvec   = -1/2 d den/dk with the dL-derivative replaced by lV(., mu),
           loop momenta set to zero                    (the linear form)
    detMmatrix  = Factor[Det[Mmatrix]]
    jmatrix   = detMmatrix * den(k=0)
    fpol = -jmatrix + Qvec . adj(Mmatrix) . Qvec^T      (expanded; the numerator of the
           Mathematica Together[...] expression)
    """
    momprop, masses, loopMom = dens
    momprop = list(momprop)
    masses = list(masses)
    loopMom = list(loopMom)
    loops = len(loopMom)
    early = _kin_early(kin, loopMom)

    den = sp.S.Zero
    for i, (p, m) in enumerate(zip(momprop, masses)):
        den += z(i + 1) * (dL(p, p) - sp.sympify(m) ** 2)
    if early:
        den = _apply_kinematics(den, kin)
    # t1:  dL.fdiff implements the Mathematica derivative replacements
    t1 = [-sp.Rational(1, 2) * sp.diff(den, km) for km in loopMom]

    # Mmatrix[j][i] = -d t1[i]/d k[j]
    Mmatrix = sp.Matrix(loops, loops, lambda j, i: -sp.diff(t1[i], loopMom[j]))

    # Qvec: same first derivative, but Derivative -> lV(., mu), then k -> 0
    zero_subs = [(km, 0) for km in loopMom]
    Qvec = []
    for km in loopMom:
        acc = sp.S.Zero
        for i, p in enumerate(momprop):
            c = sp.diff(p, km)
            if c == 0:
                continue
            q0 = p.subs(zero_subs, simultaneous=True)
            acc -= z(i + 1) * c * lV(q0, MU)   # lV distributes over sums / signs
        Qvec.append(sp.expand(acc))

    detMmatrix = sp.factor(Mmatrix.det())

    den0 = sp.S.Zero
    for i, (p, m) in enumerate(zip(momprop, masses)):
        p0 = p.subs(zero_subs, simultaneous=True)
        den0 += z(i + 1) * (dL(p0, p0) - sp.sympify(m) ** 2)
    if early:
        den0 = _apply_kinematics(den0, kin)
    jmatrix = detMmatrix * den0

    # Det[Mmatrix] * Qvec.Inverse[Mmatrix].Transpose[{Qvec}]  ==  Qvec.adj(Mmatrix).Qvec^T
    adj2 = Mmatrix.adjugate()
    quad = sp.S.Zero
    for a in range(loops):
        for b in range(loops):
            if Qvec[a] == 0 or Qvec[b] == 0 or adj2[a, b] == 0:
                continue
            quad += Qvec[a] * adj2[a, b] * Qvec[b]
    quad = lorentz_simplify(quad)     # lV(x,mu) lV(y,mu) -> dL(x, y)
    if early:
        quad = _apply_kinematics(quad, kin)  # fresh dL(P_a,P_b) from the contraction
    fpol = sp.expand(-jmatrix + quad)
    if early:
        fpol = _apply_kinematics(fpol, kin)  # the explicit `_apply_kinematics(fpol, kin)`

    return detMmatrix, fpol, Qvec, Mmatrix


def Symanzik(dens, kin=None, pos=None):
    """Symanzik(dens, kin) -> [U, F] -- the Symanzik polynomials.
    no_gauge=False (default) applies the gauge constraint

        z(pos) -> 1                       (pos = number of edges by default)

    so `pos` chooses WHICH Feynman parameter is gauge-fixed to 1.
    Arguments are exactly as in `UFdata`: dens = (momenta, masses,
    loop momenta), kin = optional dL kinematic rules.
    """

    edges= len(dens[0])
    varsz = [z(i) for i in range(1, edges + 1)]
    if pos:
        if isinstance(pos,int):
            if pos not in range(1,edges+1):
                raise ValueError(
                    "The index must be an integer between 1 and", edges)
            else:
                uf = UFdata(dens, kin)
                U, F = [uf[0], uf[1]]
                rules = {varsz[pos - 1]: 1}
                res = [U.subs(rules), F.subs(rules)]
        else:
            raise ValueError(
                "The index must be an integer between 1 and", edges)

    else:
        uf = UFdata(dens, kin)
        U, F = [uf[0], uf[1]]
        res = [U, F]
    return res


def adjA(mat):
    """adjA[x] = Simplify[Inverse[x].Det[x]]  -- the adjugate matrix."""
    return mat.adjugate()


# ---------------------------------------------------------------------------
#  Wick contractions and generate_tensor_structures
# ---------------------------------------------------------------------------

def _perfect_matchings(items):
    """All perfect matchings of `items`, in wick2's enumeration order
    (pair the first element with the n-th, recurse on the rest)."""
    if len(items) == 0:
        return [[]]
    if len(items) == 2:
        return [[(items[0], items[1])]]
    first, rest = items[0], items[1:]
    out = []
    for n in range(len(rest)):
        remain = rest[:n] + rest[n + 1:]
        for m in _perfect_matchings(remain):
            out.append([(first, rest[n])] + m)
    return out


class WickContraction(sp.Function):
    r"""Inert Wick-contraction object WickContraction[lV(k[s_1],mu_1), ..., lV(k[s_m],mu_m)].
    `generate_tensor_structures` emits exactly ONE WickContraction per even k-slot subset (the
    arguments are the loop-momentum factors of the contracted slots, in
    slot order), so the number of rank-r structures is 2^(r-1) -- as in
    the Mathematica program.  The contraction is expanded by `wick2`,
    which numSingleTerm applies right after the sigma -> loop-label
    substitution (and before the adjugate entries are substituted).  WickContraction
    is inert: it has no eval rules and only disappears through an
    explicit wick2 call.
    """

    is_commutative = True


def wick2(arg):
    """wick2[expr] -- expand the Wick contraction carried by `arg`.
    `arg` is either a WickContraction(...) object or the product of its lV(k[s], mu)
    factors.  Every factor contributes one slot (loop label s, index mu);
    the result is the Gaussian contraction

        Sum over perfect matchings of  Prod adj[s_a, s_b] * MT[mu_a, mu_b]

    enumerated in the original wick2 order (pair the first factor with
    the n-th, recurse on the rest), with symbolic adj entries -- exactly
    the factors the per-matching expansion attaches to each structure, so
    the subsequent  adj[e1_, e2_] :> adjugates[[e1, e2]]  rule of
    numSingleTerm fires unchanged.
    """
    if isinstance(arg, WickContraction):
        factors = list(arg.args)
    else:
        factors = list(sp.Mul.make_args(sp.sympify(arg)))
    parsed = []
    for f in factors:
        if not (isinstance(f, lV) and f.args and f.args[0].func == _kfun):
            raise ValueError('wick2: expected lV(k[s], mu) factors, got %s' % f)
        parsed.append((f.args[0].args[0], f.args[1]))

    def _mu_pos(lab_mu):
        name = str(lab_mu[1])
        if name.startswith('mu') and name[2:].isdigit():
            return int(name[2:])          # restore slot order (mu1, mu2, ...)
        return sp.default_sort_key(lab_mu[1])

    parsed.sort(key=_mu_pos)
    out = sp.S.Zero
    for match in _perfect_matchings(list(range(len(parsed)))):
        term = sp.S.One
        for (i, j) in match:
            term *= (_adjfun(parsed[i][0], parsed[j][0])
                     * MT(parsed[i][1], parsed[j][1]))
        out += term
    return sp.expand(out)


def generate_tensor_structures(r):
    """generate_tensor_structures[r] -> (terms, factor).
    Parametric (post T-integration) rank-r tensor structures, before the
    substitution lV(P[s], mu) -> cP3 / adj -> adjugate entries.  ONE term
    per even k-slot subset, so len(terms) = 2^(r-1) for r >= 1:

        term = (-1)^(r+j) * (-1/2)^p * F_POL^p * Gamma-product
               * WickContraction[lV(k[s_a],mu_a), ..., lV(k[s_b],mu_b)]
               * Prod_{P-slots i} lV(P[s_i], mu_i)

    where j = number of P-slots, 2p = number of k-slots and the Wick
    contraction over the k-slots is carried INERT by WickContraction.  Expanding it
    with `wick2` yields

        Sum over perfect matchings of  Prod adj[s_a,s_b] * MT[mu_a,mu_b]
    """
    factor = (F_POL ** (d * L / 2 - N_EDGES)
              * U_POL ** (-d / 2 - d * L / 2 + N_EDGES - r)
              * sp.I ** (-2 * N_EDGES))
    if r == 0:
        return [sp.gamma(N_EDGES - d * L / 2)], factor

    terms = []
    mu_ids = [muidx(i) for i in range(1, r + 1)]
    sg = [sigma(i) for i in range(1, r + 1)]
    half = r // 2
    # slot choices (k=0 / P=1), slot 1 slowest -- same order as Expand of
    # Product[lV[k[s_i],mu_i] + U_POL^-1 lV[-P[s_i],mu_i], {i,1,r}]; ONE term
    # per choice with an even number of k-slots (the odd ones are removed
    # by DeleteCases[-X__] in the original)
    for choice in product((0, 1), repeat=r):
        pslots = [i + 1 for i, c in enumerate(choice) if c == 1]
        kslots = [i + 1 for i, c in enumerate(choice) if c == 0]
        j = len(pslots)
        if (r - j) % 2:
            continue          # odd number of loop momenta: DeleteCases[-X__]
        p = (r - j) // 2
        sign = sp.Integer(-1) ** (r + j) * sp.Rational(-1, 2) ** p
        if r >= 2:
            grest = sp.Mul(*[N_EDGES - d * L / 2 - q for q in range(p + 1, half + 1)])
            g = grest * sp.gamma(N_EDGES - d * L / 2 - half)
        else:
            g = sp.gamma(N_EDGES - d * L / 2 - p)
        pfac = sp.S.One
        for i in pslots:
            pfac *= lV(_pfun(sg[i - 1]), mu_ids[i - 1])
        term = sign * F_POL ** p * g * pfac
        if kslots:
            # the Wick contraction over the k-slots, kept inert (WickContraction head):
            # wick2 expands it into the sum over perfect matchings
            term = term * WickContraction(*[lV(_kfun(sg[i - 1]), mu_ids[i - 1])
                               for i in kslots])
        terms.append(term)
    return terms, factor

# ---------------------------------------------------------------------------
# index / unContractnumerator / unContractnumeratorList / toDataNum
# ---------------------------------------------------------------------------

def _loop_label(x):
    """Return i if x is the loop momentum k[i], else None."""
    if isinstance(x, sp.core.function.AppliedUndef) and x.func == _kfun:
        return int(x.args[0])
    return None


def _check_dL_args(a, b):
    for x in (a, b):
        if x.has(_kfun) and _loop_label(x) is None:
            raise ValueError(
                "numerator dL arguments must be single momenta k(i) or P(a); "
                "got %s" % x)


def _uncontract_term(term):
    """
    Every dL(k_i, k_j) becomes MT[mu_a, mu_b] with two lambda slots,
    every dL(k_i, u) (u external) becomes lV(u, mu_a) with one slot
    (the metric factor index[x,y] is absorbed by the MT.lV contraction,
    exactly as in Mathematica where the rules fire on re-evaluation),
    and dL of two external momenta is kept as an invariant.

    Returns (coefficient, labels):
      coefficient -- the term with all lambda factors -> 1, its Lorentz
                     indices renumbered positionally to mu(1)..mu(r)
                     (the /. indexRules step of toDataNum),
      labels      -- the loop labels of the slots, in slot order
                     (the dataNum[[2]] entry).
    """
    others = []
    entries = []   # {'kind': 'kk'|'kp', 'labels': [...], 'ext': momentum}
    for f in sp.Mul.make_args(term):
        base, expo = f.as_base_exp()
        if isinstance(base, dL) and expo.is_Integer and expo > 0:
            for _ in range(int(expo)):
                a, b = base.args
                _check_dL_args(a, b)
                la, lb = _loop_label(a), _loop_label(b)
                if la is not None and lb is not None:
                    entries.append({'kind': 'kk', 'labels': [la, lb]})
                elif la is not None:
                    entries.append({'kind': 'kp', 'labels': [la], 'ext': b})
                elif lb is not None:
                    entries.append({'kind': 'kp', 'labels': [lb], 'ext': a})
                else:
                    others.append(base)
        else:
            others.append(f)
    # positional indices: slots sorted by (loop label, appearance order) --
    # the Times-canonical order of the lambda[i, unique] factors
    lam = []
    for eid, e in enumerate(entries):
        for sid, lab in enumerate(e['labels']):
            lam.append((lab, eid, sid))
    lam.sort(key=lambda t: (t[0], t[1], t[2]))
    mu_of = {}
    labels_out = []
    for pos, (lab, eid, sid) in enumerate(lam):
        mu_of[(eid, sid)] = muidx(pos + 1)
        labels_out.append(lab)

    struct = sp.S.One
    for eid, e in enumerate(entries):
        i1 = mu_of[(eid, 0)]
        if e['kind'] == 'kk':
            struct *= MT(i1, mu_of[(eid, 1)])
        else:
            struct *= lV(e['ext'], i1)
    coeff = sp.Mul(*others) * struct
    return sp.expand(coeff), labels_out


def unContractnumeratorList(num):
    """unContractnumeratorList[num] -> list of (coefficient, labels) pairs."""
    num = sp.expand(num)
    if num == 1:
        return [(sp.S.One, [])]
    return [_uncontract_term(term) for term in sp.Add.make_args(num)]

def toDataNum(num):
    """toDataNum[num] -> (coeffs, labels).
    coeffs[i] : index-carrying coefficient of numerator term i
    labels[i] : list of loop labels of term i (its rank = len(labels[i]))
    """
    pairs = unContractnumeratorList(num)
    coeffs = [c for c, _lab in pairs]
    labels = [lab for _c, lab in pairs]
    return coeffs, labels


# ---------------------------------------------------------------------------
# bmat2 / cP3
# ---------------------------------------------------------------------------
def bmat2(rho_idx, ufdata):
    """bmat2[r][ufdata] := ufdata[[3]] /. mu -> r  (Qvec with index renamed)."""
    return [t.subs(MU, rho_idx) for t in ufdata[2]]


def cP3(sig_lab, rho_idx, datas):
    """cP3[sigma, rho, {dens, ufdata, adjugates}] =
       Sum_i adjugates[sigma][i] * (Qvec_i with index mu -> rho).

    This is the sigma component of adj(Mmatrix).Qvec -- the external-momentum part
    of the shifted loop momentum -- carrying the free index rho.
    """
    dens, ufdata, adjugates = datas
    bmat = bmat2(rho_idx, ufdata)
    res = sp.S.Zero
    for i in range(len(dens[2])):
        res += adjugates[int(sig_lab) - 1, i] * bmat[i]
    return sp.expand(res)


def _map_ufdata(uf, f):
    """Apply f to every scalar entry of (detMmatrix, fpol, Qvec, Mmatrix)."""
    detMmatrix, fpol, Qvec, Mmatrix = uf
    return (f(detMmatrix), f(fpol), [f(t) for t in Qvec], Mmatrix.applyfunc(f))


def _pipeline_setup(num, dens, kin, no_gauge):
    """Shared front end of arrayIntegrandData / integrandSlotDicts:
    numerator -> (dataNum, datas, highestRank, edges, tensorsUptoHighest,
    nprops, loops)."""
    kin = kin or {}
    if _kin_early(kin, list(dens[2])):
        num = _apply_kinematics(sp.sympify(num), kin)
    coeffs, labels = toDataNum(num)
    edges = len(dens[0])
    nprops = edges
    uf_raw = UFdata(dens, kin)
    z_e = z(edges)
    if no_gauge:
        uf = _map_ufdata(uf_raw, lambda e: sp.expand(_apply_kinematics(e, kin)))
    else:
        uf = _map_ufdata(uf_raw,
                         lambda e: sp.expand(_apply_kinematics(e.subs(z_e, 1), kin)))

    loops = len(dens[2])
    mat = uf[3]
    adjugates = adjA(mat)
    highestRank = max(len(lab) for lab in labels)
    tensorsUptoHighest = [generate_tensor_structures(i) for i in range(highestRank + 1)]
    datas = (dens, uf, adjugates)
    return ((coeffs, labels), datas, highestRank, edges, tensorsUptoHighest,
            nprops, loops)


def tensor_rank(num):
    """Rank r of the numerator (the `highestRank` of the Mathematica code):
    the largest number of loop-momentum slots among its terms."""
    _coeffs, labels = toDataNum(num)
    return max(len(lab) for lab in labels)


_POLST_RANK_CACHE = {}     # numerator polynomial -> its highest rank
_POLST_RANK_LAST = None    # (num, rank) identity fast path

def _polst_highest_rank(num):
    """tensor_rank(num), memoised on the (hashable) numerator expression."""
    global _POLST_RANK_LAST
    if _POLST_RANK_LAST is not None and _POLST_RANK_LAST[0] is num:
        return _POLST_RANK_LAST[1]
    ent = _POLST_RANK_CACHE.get(num, False)
    if ent is False:
        ent = tensor_rank(num)
        if len(_POLST_RANK_CACHE) >= 64:
            _POLST_RANK_CACHE.clear()
        _POLST_RANK_CACHE[num] = ent
    _POLST_RANK_LAST = (num, ent)
    return ent

# ---------------------------------------------------------------------------
# public aliases
# ---------------------------------------------------------------------------
#: map every ufdata component through f (structure-preserving)
map_ufdata = _map_ufdata
#: ONE unexpanded numerator term -> (coeff, loop-label tuple)
uncontract_term = _uncontract_term
#: tensor_rank(num), memoised
polst_highest_rank = _polst_highest_rank
#: numerator -> (dataNum, datas, highestRank, edges, tensorsUptoHighest,
#: nprops, loops) -- the shared problem front end
pipeline_setup = _pipeline_setup

__all__ = [
    'UFdata', 'Symanzik', 'adjA', 'WickContraction', 'wick2', 'generate_tensor_structures',
    'unContractnumeratorList', 'toDataNum', 'bmat2', 'cP3',
    'tensor_rank', 'map_ufdata', 'uncontract_term',
    'polst_highest_rank', 'pipeline_setup',
]