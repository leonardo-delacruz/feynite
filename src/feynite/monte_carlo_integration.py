import datetime
import os

from .constants import  _zfun
from .lorentz import infer_rank
from .symanzik import Symanzik
from .num_evaluator import pol_numerator
import vegas

import itertools
import time

import numpy as np
import sympy as sp

EPS = 1e-12
CHUNK = 1 << 17


BUILTIN_LOCALS = {}

def make_np_arrays(spec, gauge_pos=None, sparse=False, quiet=False):
    """Build num/U/F monomial tables for an arbitrary diagram.
    spec (dict -- per-diagram inputs, see the DIAGRAMS builders):
      dens       : (props, powers, loop_momenta)
      num_expr   : numerator as a sympy expression (already sympify'd
                   with the diagram's locals)
      kin        : dict {dL(pi, pj): value} in sympy invariant symbols
      parameters : dict {Symbol: float} -- must map EVERY non-z symbol
                   appearing in num/U/F (invariants + d)
      rank, dim  : numerator tensor rank, spacetime dimension

    gauge_pos: which Symanzik variable to fix to 1.  Default: the LAST
    edge
    Returns (arrays, exponent_u, exponent_f, info) with
        arrays = {'num': (coeffs, exps), 'U': ..., 'F': ...}
        info   = dict(edges, loops, rank, dim, gauge_pos, nvars,
                      num_zero)
    info['num_zero'] is True when the parametric numerator vanishes
    identically (raw zero / zero after contraction / zero after
    parameter substitution): the integrand is then exactly 0, the
    integral is exactly I = 0 +- 0, a WARNING is printed and the num
    table is normalized to a single zero monomial so every downstream
    consumer stays safe.
    """
    dens = spec['dens']
    rank, dim = int(spec['rank']), spec['dim']
    edges, loops = len(dens[0]), len(dens[2])
    overall_sign=sp.sympify((-1)**(edges)) #Matching pySecDec conventions
    if gauge_pos is None:
        gauge_pos = edges
    elif gauge_pos != edges:
        print('WARNING: gauge_pos=%d != edges=%d: ufdata_tensor fixes the'
              ' LAST variable inside polNumerator -- num and U/F may use'
              ' inconsistent gauges' % (gauge_pos, edges))

    num_in = spec['num_expr']
    if sp.expand(num_in) == 0:
        num = sp.S.Zero
    else:
        num = overall_sign*sum(pol_numerator(num_in, dens, spec['kin'],
                               sparse=sparse, no_gauge=False))
    upol, fpol = Symanzik(dens, spec['kin'], pos=gauge_pos)

    # powersUF(rank, dim)
    exponent_u = int(-(edges - dim * (loops + 1) / 2 - rank))
    exponent_f = int(-(-edges + dim * loops / 2))

    # integration variables: all z(i) except the gauge-fixed one
    zvars = tuple(_zfun(i) for i in range(1, edges + 1) if i != gauge_pos)

    def poly_to_arrays(expr, syms):
        poly = sp.Poly(sp.expand(expr), *syms)
        coeffs = np.array([float(c) for c in poly.coeffs()])
        exps = np.array(poly.monoms(), dtype=np.int64)
        return coeffs, exps

    # EXACT rational arithmetic: float parameters are coerced to exact
    # decimals BEFORE the expansion.  Substituting floats first creates
    # 1e-14 cancellation junk in coefficients whose exact value is zero
    params_exact = {}
    for k, v in spec['parameters'].items():
        if isinstance(v, float):
            try:
                v = sp.Rational(str(v))
            except (ValueError, OverflowError, sp.PolynomialError):
                pass
        params_exact[k] = v
    arrays = {}
    arrays['num'] = poly_to_arrays(num.subs(params_exact), zvars)
    arrays['U'] = poly_to_arrays(upol.subs(params_exact), zvars)
    arrays['F'] = poly_to_arrays(fpol.subs(params_exact), zvars)

    # identically-zero parametric numerator: raw zero, zero after the kinematic
    # contraction, or zero after parameter substitution -- in every case
    # the integrand is EXACTLY 0 for every z, so the integral is exactly
    # I = 0 +- 0 and no Monte-Carlo is needed (nor meaningful: the
    # degree-based audits would emit spurious divergence warnings).
    num_zero = (arrays['num'][0].size == 0
                or not np.any(arrays['num'][0] != 0.0))
    if num_zero:
        arrays['num'] = (np.zeros(1),
                         np.zeros((1, len(zvars)), dtype=np.int64))
        print('WARNING make_np_arrays: the parametric numerator is '
              'identically ZERO (every monomial coefficient vanished '
              'after parameter substitution) -- the integrand is exactly '
              '0 for every z, so the integral is exactly I = 0 +- 0.  '
              'No Monte-Carlo is needed; run_numerator returns the '
              'exact zero and flags the row "zero".')

    info = dict(edges=edges, loops=loops, rank=rank, dim=dim,
                gauge_pos=gauge_pos, nvars=len(zvars), num_zero=num_zero)
    if not quiet:
        print('make_np_arrays: edges=%d loops=%d rank=%d dim=%g -> '
              'U^%d F^%d, %d integration vars (z(%d)->1), monomials: '
              'num %d, U %d, F %d'
              % (edges, loops, rank, dim, exponent_u, exponent_f,
                 len(zvars), gauge_pos,
                 len(arrays['num'][1]), len(arrays['U'][1]),
                 len(arrays['F'][1])))
    return arrays, exponent_u, exponent_f, info


def taming_exponent(edges, num_e, U_e, F_e, eU, eF, quiet=False):
    """Derive the taming exponent a of the map  z = (u/(1-u))^a.

    All degrees are MEASURED from the built monomial tables, so the
    formula is self-correcting: a generic rank-r tensor numerator has
    deg(num) = rank*loops by construction and the identity kappa2 =
    edges below yields a = edges-1 independent of loops/dim/rank.
    Numerators containing loop-momentum squares (k.k traces) come out
    with a higher deg(num); the same formula then returns the milder
    exponent they need (or 1.0 if z->inf is already convergent).

    kappa2 = eU*deg(U) + eF*deg(F) - deg(num)     # f ~ z^-kappa2
    with deg(U)=L, deg(F)=L+1, deg(num)=rL and
    eU = d(L+1)/2 + r - N, eF = N - dL/2  this collapses to
    kappa2 = N identically, hence through z = w^a:
        g = f*J ~ s^(a(N-n) - n) = s^(a-n),  n = N-1
    -> unique flat point a = n = N-1; variance finite for a > (N-1)/2.

    The z->0 corner is bounded at a = n iff
        kappa1 = eU*mindeg(U) + eF*mindeg(F) - mindeg(num) <= n-1,
    exactly the condition for the corner value integral to converge --
    automatic for convergent integrals (checked, warned if violated).
    """
    n = edges - 1
    dU, dF = int(U_e.sum(axis=1).max()), int(F_e.sum(axis=1).max())
    dnum = int(num_e.sum(axis=1).max())
    mU, mF = int(U_e.sum(axis=1).min()), int(F_e.sum(axis=1).min())
    mnum = int(num_e.sum(axis=1).min())
    kappa2 = eU * dU + eF * dF - dnum
    kappa1 = eU * mU + eF * mF - mnum
    if not quiet:
        print('taming exponent: kappa2 = %d*%d + %d*%d - %d = %d'
              % (eU, dU, eF, dF, dnum, kappa2))
        print('  kappa2 == edges: %s (deg(num) = rank*loops construction)'
              % (kappa2 == edges))
        print('  kappa1 (corner strength) = %d -> auto-tamed at a=n: %s'
              % (kappa1, kappa1 <= n - 1))
    if kappa1 > n - 1:
        print('  WARNING: kappa1=%d > n-1=%d: z->0 corner may be '
              'value-divergent' % (kappa1, n - 1))
    if kappa2 <= n:
        if not quiet:
            print('  kappa2 <= n: z->inf end already convergent -> a = 1')
        return 1.0
    a = n / (kappa2 - n)
    if not quiet:
        print('  derived a = n/(kappa2-n) = %d/%d = %g%s'
              % (n, kappa2 - n, a,
                 '  (= edges-1)' if kappa2 == edges else ''))
    return a


# ---------------------------------------------------------------------------
# Boundary-face audit and per-variable taming repair
# ---------------------------------------------------------------------------
def face_E(num_e, U_e, F_e, eU, eF, a, S):
    """Boundary exponent of g = num/(U^eU F^eF) * J on the face where
    the variables in S -> 1 together (z_j = w^{a_j} -> inf, others O(1)):
    g ~ w^{E_S}.  E_S <= 0  <=>  g bounded there (honest MC variance);
    E_S < |S|  <=>  the face value integral converges.

    Returns (E_S, c) with c the per-variable combined leading exponent
    (dE_S/da_j = c_j + 1 while the leading monomials stay in charge).
    """
    idx = list(S)
    aS = a[idx]
    ln = num_e[:, idx] @ aS
    lu = U_e[:, idx] @ aS
    lf = F_e[:, idx] @ aS
    jn, ju, jf = int(ln.argmax()), int(lu.argmax()), int(lf.argmax())
    E = ln[jn] - eU * lu[ju] - eF * lf[jf] + aS.sum() + len(S)
    c = num_e[jn, idx] - eU * U_e[ju, idx] - eF * F_e[jf, idx]
    return E, c


def audit_faces(num_e, U_e, F_e, eU, eF, a):
    """Audit all 2^n - 1 boundary faces at weights a, worst E_S first.
    Returns a list of [E_S, S, c]."""
    n = num_e.shape[1]
    faces = []
    for r in range(1, n + 1):
        for S in itertools.combinations(range(n), r):
            E, c = face_E(num_e, U_e, F_e, eU, eF, a, S)
            faces.append([E, S, c])
    faces.sort(key=lambda f: -f[0])
    return faces


def repair_taming(num_e, U_e, F_e, eU, eF, a0, verbose=False):
    """Raise per-variable exponents until every face has E_S <= 0.
    Single-variable faces give the hard lower bounds
        a_j >= 1/(eU*u_j + eF*f_j - d_j - 1)
    (d_j/u_j/f_j = maximal per-variable powers); starting there, each
    step cures the current worst face with one exact linear move on its
    strongest lever variable.

    If a face has c_j + 1 >= 0 for ALL its variables (no power map can tame it), if the a_j >= 50
    over-taming cap is hit, or if the 200-step budget is exhausted, it
    prints a WARNING and returns the best map found so far.  Any map
    is still an EXACT change of variables, so the integral value is
    untouched; only the variance/quality of the MC estimate can
    suffer -- and a genuinely divergent integral betrays itself
    through the diagnostics (Q << 1 or erratic estimates).

    Returns (a, n_steps, final_faces_worst_first).
    """
    a = np.array(a0, dtype=float)
    for it in range(200):
        faces = audit_faces(num_e, U_e, F_e, eU, eF, a)
        E, S, c = faces[0]
        if E <= 1e-9:
            return a, it, faces
        jj = int(np.argmin(c))               # strongest lever in face S
        if c[jj] + 1.0 >= -1e-12:
            print('  WARNING repair_taming: face %s has E_S=%+g '
                  'divergent for ANY power map (c=%s) -- no exact power '
                  'map can tame this boundary; RUNNING with the best '
                  'map found. '
                  % (S, E, c))
            return a, it, faces
        if a[S[jj]] >= 50.0:
            print('  WARNING repair_taming: face %s still violates '
                  '(E_S=%+g) with a_j >= 50 -- refusing to over-tame '
                  '(interior Jacobian peaks would ruin the bulk); '
                  'RUNNING with the best map found.' % (S, E))
            return a, it, faces
        step = E / (-(c[jj] + 1.0))
        a[S[jj]] += step
        if verbose and (it < 10 or it % 20 == 0):
            print('    repair it%02d: worst E=%+8.3g on face %s -> a[%d] '
                  '+= %.3f' % (it, E, S, S[jj], step))
    print('  WARNING repair_taming: no convergence in 200 steps -- '
          'RUNNING with the best map found (worst E_S=%+g on %s). '
          % (faces[0][0], faces[0][1]))
    return a, 200, faces


# ---------------------------------------------------------------------------
# Fast vectorised polynomial evaluation (z-power table, no blow-up)
# ---------------------------------------------------------------------------
def eval_mons(z, c, e):
    """(n,d) z-values -> (n,) sum_j c_j prod_i z[:,i]**e[j,i]."""
    n = z.shape[0]
    emax = int(e.max())
    out = np.empty(n)
    # gather array is chunk*nterms*nvars*8 bytes: shrink chunk for heavy
    # numerators so the transient stays ~<= 130 MB (rank-2 keeps CHUNK)
    chunk = int(min(CHUNK, max(4096, (1 << 27)
                                     // max(1, e.shape[0] * e.shape[1] * 8))))
    for s in range(0, n, chunk):
        zc = z[s:s + chunk]
        zp = np.ones((zc.shape[0], emax + 1, zc.shape[1]))
        for k in range(1, emax + 1):
            zp[:, k, :] = zp[:, k - 1, :] * zc
        idx = np.take_along_axis(
            zp, np.broadcast_to(e[None], (zc.shape[0],) + e.shape), axis=1)
        out[s:s + chunk] = idx.prod(axis=2) @ c
    return out


class Integrand:
    """num/(U^eU F^eF) over [0,inf)^nvars sampled through z = (u/(1-u))^a.
    a=1 is exactly the 's original map t/(1-t)."""

    def __init__(self, arrays, exponent_u, exponent_f, a):
        self.num_c, self.num_e = arrays['num']
        self.U_c, self.U_e = arrays['U']
        self.F_c, self.F_e = arrays['F']
        self.eU = int(exponent_u)
        self.eF = exponent_f
        self.a = np.asarray(a, dtype=float)   # scalar or per-variable

    def __call__(self, u):
        u = np.clip(u, EPS, 1.0 - EPS)
        a = self.a
        w = u / (1.0 - u)
        z = w ** a
        jac = np.prod(a * w ** (a - 1.0) / (1.0 - u) ** 2, axis=1)
        numv = eval_mons(z, self.num_c, self.num_e)
        Uv = eval_mons(z, self.U_c, self.U_e)
        Fv = eval_mons(z, self.F_c, self.F_e)
        Ud = Uv if self.eU == 1 else Uv ** self.eU   # eU=1: bit-identical
        with np.errstate(all='ignore'):
            val = numv / (Ud * Fv ** self.eF) * jac
        return np.where(np.isfinite(val), val, 0.0)


def audit_corners(num_e, U_e, F_e, eU, eF):
    """Value-convergence audit of ALL 2^n-1 z->0 corner faces of the
    ORIGINAL integral (map-independent!).

    On the face z_i -> 0 for i in T together (others O(1)):
        f = num/(U^eU F^eF) ~ r^{D_T},
        D_T = mindeg_T(num) - eU*mindeg_T(U) - eF*mindeg_T(F)
    (joint min degrees restricted to T), and the value integral
    converges iff D_T > -|T|, i.e. margin = D_T + |T| - 1 >= 0.
    T = all variables reproduces the classic kappa1 <= n-1 rule.
    A violation STRONGLY SUGGESTS the integral itself diverges (no
    change of variables can fix it).
    Returns [(margin, T), ...] worst first.
    """
    n = num_e.shape[1]
    out = []
    for r in range(1, n + 1):
        for T in itertools.combinations(range(n), r):
            dn = int(num_e[:, T].sum(axis=1).min())
            du = int(U_e[:, T].sum(axis=1).min())
            df = int(F_e[:, T].sum(axis=1).min())
            out.append((dn - eU * du - eF * df + r - 1, T))
    out.sort(key=lambda x: x[0])
    return out

# ---------------------------------------------------------------------------
# Results file
# ---------------------------------------------------------------------------
RESULT_COLUMNS = ('# timestamp tag rank eU eF nvars map flag value sdev '
                  'Q sobol sobol_sdev evals seconds')

QUICK_SETTINGS = dict(train=(10, 20000), nitn=10, neval=100000)
PROD_SETTINGS = dict(train=(15, 50000), nitn=20, neval=200000)


def _append_result(outfile, line_fields):
    header_needed = not (os.path.exists(outfile)
                         and os.path.getsize(outfile) > 0)
    with open(outfile, 'a') as fh:
        if header_needed:
            fh.write(RESULT_COLUMNS + '\n')
        fh.write(' '.join(line_fields) + '\n')


# ---------------------------------------------------------------------------
# parallel Sobol scrambles (
# ---------------------------------------------------------------------------
_SOBOL_INTEGRAND = None

def _sobol_init(ctx):
    global _SOBOL_INTEGRAND
    arrays, eU, eF, a_map = ctx
    _SOBOL_INTEGRAND = Integrand(arrays, eU, eF, a_map)


def _sobol_scramble(job):
    """One scrambled-Sobol estimate (mean over 2^m points)."""
    m, seed0, r = job
    from scipy.stats import qmc
    d = _SOBOL_INTEGRAND.num_e.shape[1]
    sob = qmc.Sobol(d=d, scramble=True, seed=seed0 + r)
    return float(_SOBOL_INTEGRAND(sob.random_base2(m)).mean())


def run_sobol_parallel(arrays, eU, eF, a_map, m, R, workers=1, seed0=12345):
    """mc.run_sobol with the R scrambles spread over `workers` processes.
    workers=1 (or R=1) reproduces mc.run_sobol bit-for-bit (same seed0,
    same order, same combination).  Each worker rebuilds the Integrand
    from the picklable (coeff, exponent) tables; the 2^m x nvars point
    block costs 2^m * nvars * 8 bytes of RAM per worker -- size m
    accordingly.  Returns (value, sdev, npts, seconds)."""
    ctx = (arrays, eU, eF, a_map)
    jobs = [(int(m), int(seed0), r) for r in range(int(R))]
    t0 = time.time()
    if workers > 1 and len(jobs) > 1:
        try:
            import multiprocessing as mp
            with mp.Pool(processes=int(workers), initializer=_sobol_init,
                         initargs=(ctx,)) as pool:
                est = np.array(pool.map(_sobol_scramble, jobs))
        except Exception as exc:            # any pool problem -> serial
            print('[mc_batch] parallel Sobol unavailable (%s) -- serial '
                  'fallback' % exc)
            _sobol_init(ctx)
            est = np.array([_sobol_scramble(j) for j in jobs])
    else:
        _sobol_init(ctx)
        est = np.array([_sobol_scramble(j) for j in jobs])
    err = est.std(ddof=1) / np.sqrt(len(est))
    return (float(est.mean()), float(err),
            len(est) * (1 << int(m)), time.time() - t0)


# ---------------------------------------------------------------------------
# Engines: vegas and scrambled-Sobol QMC
# ---------------------------------------------------------------------------
def run_vegas(integrand, nitn, neval, adapt=True):
    import vegas
    integ = vegas.Integrator([[0, 1]] * integrand.num_e.shape[1])

    @vegas.batchintegrand
    def f(u):
        return integrand(u)

    t0 = time.time()
    r = integ(f, nitn=nitn, neval=neval, adapt=adapt)
    mean = getattr(r, 'mean', None)
    if mean is None:                       # fallback for other vegas versions
        mean = getattr(r, 'central', r)
    return (float(mean), float(r.sdev), float(r.Q), time.time() - t0, r)


def run_sobol(integrand, m, R, seed0=12345):
    from scipy.stats import qmc
    d = integrand.num_e.shape[1]
    t0 = time.time()
    est = np.empty(R)
    for r in range(R):
        sob = qmc.Sobol(d=d, scramble=True, seed=seed0 + r)
        U = sob.random_base2(m)                     # 2^m points
        est[r] = integrand(U).mean()
    err = est.std(ddof=1) / np.sqrt(R)
    return (float(est.mean()), float(err), R * (1 << m), time.time() - t0)


# ---------------------------------------------------------------------------
# The one-call driver
# ---------------------------------------------------------------------------
def run_numerator(num, kin=None, dens=None, parameters=None, dim=4,
                  rank=None, tag=None, outfile='mc_results.txt',
                  train=(15, 50000), nitn=20, neval=200000, adapt=False,
                  alpha=None, taming='tamed', sobol=None, workers=1,
                  quiet=False):
    """Build, tame and MC-integrate ONE numerator; save + return results.

    num        : sympy expression or numerator string
    kin, dens  : kinematic rules / propagators (default: dbox)
    parameters : {Symbol: value} for every non-z symbol (default:
                 s12=-7, s23=-1, d=dim)
    rank       : force the tensor rank (default: infer from num)
    train      : (nitn, neval) vegas training before production;
                 None disables training (not recommended for a=1)
    nitn, neval, adapt : production vegas call
    taming     : 'tamed' (derived a + face audit/repair) or 'plain'
                 (a=1, the original t/(1-t) map)
    sobol      : None or (m, R) for a scrambled-Sobol QMC estimate
                 (added as its own results-file columns)
    workers    : processes for the Sobol scrambles (1 = serial, identical
                 to mc.run_sobol; >1 needs R > 1, ~linear speedup)
    alpha      : vegas map-adaptation rate (use together with adapt=True;
                 None = vegas default 0.5, 0.3 = slower/stabler)
    outfile    : results txt (one whitespace-separated line per call)

    When BOTH the vegas production and the Sobol cross-check ran, the
    returned dict also carries 'combined' = inverse-variance-weighted
    (value, sdev) of the two estimators (stdout only, txt schema kept).

    If the parametric numerator vanishes identically, the exact result
    is returned without any MC: value = 0.0, sdev = 0.0, Q = nan,
    flag = 'zero' (a WARNING is printed by make_np_arrays).

    Returns dict(tag, rank, eU, eF, nvars, a, value, sdev, Q, sobol,
    seconds, info).
    """
    t0 = time.time()
    if kin is None:
        kin = []
        print("Error, you must define the kinematics for a numerical evaluation")
    if dens is None:
        dens = []
        print("Error, you must define the denominators for a numerical evaluation")
    loops = list(dens[2])
    edges, loops_n = len(dens[0]), len(dens[2])

    if isinstance(num, str):
        num = sp.sympify(num, locals=dict(BUILTIN_LOCALS), evaluate=False)

    if rank is None:
        rank = infer_rank(num, loops)
    if rank < 0:
        raise ValueError('negative tensor rank %d' % rank)
    if rank == 0 and not quiet:
        print('[mc_batch] %s: scalar numerator (rank 0 -- no loop '
              'momenta)' % (tag or 'numerator'))

    if parameters is None:
        parameters = {}
        print("you must provide parameters")
    params = {}
    params.update(parameters)

    spec = dict(dens=dens, num_expr=num, kin=kin, rank=int(rank),
                dim=dim, parameters=params)
    missing = {s for s in num.free_symbols} - set(params) - set(loops)
    missing = {s for s in missing if not s.free_symbols == set()}
    if missing:
        raise ValueError('parameters must map every invariant symbol of '
                         'the numerator; missing: %s'
                         % ', '.join(sorted(str(s) for s in missing)))

    arrays, eU, eF, info = make_np_arrays(spec, quiet=quiet)

    #   Nothing is rejected: print the
    # message, write the 'zero' row and return the exact answer.
    if info.get('num_zero'):
        tag = str(tag).replace(' ', '_') if tag is not None \
            else 'num%d' % int(time.time())
        seconds = time.time() - t0
        _append_result(outfile, [
            datetime.datetime.now().isoformat(timespec='seconds'), tag,
            str(rank), str(eU), str(eF), str(info['nvars']), 'a=NA',
            'zero', '%.8e' % 0.0, '%.3e' % 0.0, 'NA', 'NA NA', '0',
            '%.1f' % seconds])
        if not quiet:
            print('[mc_batch] %s: parametric numerator vanishes '
                  'identically -> returning the EXACT result '
                  'I = 0 +- 0, no MC run [zero] (%.1f s)'
                  % (tag, seconds))
        return dict(tag=tag, rank=rank, eU=eU, eF=eF, nvars=info['nvars'],
                    a=None, a_str='a=NA', flag='zero', corner_faces=[],
                    value=0.0, sdev=0.0, Q=float('nan'),
                    sobol=(None, None), seconds=seconds, info=info,
                    arrays=arrays, result=None, combined=None)

    # z->0 corner-face audit of the ORIGINAL integral (map-independent;
    # see audit_corners).
    corner_faces = audit_corners(arrays['num'][1], arrays['U'][1],
                                 arrays['F'][1], eU, eF)
    corner_bad = corner_faces[0][0] < -1e-9
    if corner_bad:
        print('[mc_batch] WARNING %s: z->0 corner audit FAILED on %d of '
              '%d faces (worst D_T+|T|-1 = %+.0f on T=%s) -- the '
              'integral may be value-divergent there; RUNNING the MC '
              'anyway (results row flagged corner!).  The diagnostics will '
              'show it: Q << 1, erratic '
              'estimates.'
              % (tag, sum(1 for f_ in corner_faces if f_[0] < -1e-9),
                 len(corner_faces), corner_faces[0][0], corner_faces[0][1]))

    # derived taming map (identical logic to mc_precision_dbox.main)
    a_scalar = taming_exponent(edges, arrays['num'][1], arrays['U'][1],
                                  arrays['F'][1], eU, eF, quiet=quiet)
    a_vec = np.full(info['nvars'], a_scalar)
    faces = audit_faces(arrays['num'][1], arrays['U'][1],
                           arrays['F'][1], eU, eF, a_vec)
    if faces[0][0] > 1e-9:
        a_vec, nrep, faces = repair_taming(
            arrays['num'][1], arrays['U'][1], arrays['F'][1],
            eU, eF, a_vec, verbose=not quiet)
        if faces[0][0] > 1e-9:
            print('[mc_batch] WARNING %s: %d z->inf boundary face(s) '
                  'remain untamed after repair (worst E_S=%+.3g on %s) '
                  '-- RUNNING with the best map found; the MC '
                  'diagnostics (Q, stability) will judge the result'
                  % (tag, sum(1 for f_ in faces if f_[0] > 1e-9),
                     faces[0][0], faces[0][1]))
    a_map = a_vec if taming == 'tamed' else 1.0
    if taming == 'plain':
        a_str = 'a=1(plain)'
    elif np.all(a_vec == a_scalar):
        a_str = 'a=%g' % a_scalar
    else:
        a_str = 'a=[' + ','.join('%.2f' % v for v in a_vec) + ']'
    flag = 'corner!' if corner_bad else 'ok'

    integrand = Integrand(arrays, eU, eF, a_map)
    integ = vegas.Integrator([[0, 1]] * info['nvars'])

    @vegas.batchintegrand
    def f(t):
        return integrand(t)

    if train is not None:
        integ(f, nitn=int(train[0]), neval=int(train[1]))
    kw = {} if alpha is None else dict(alpha=float(alpha))
    r = integ(f, nitn=nitn, neval=neval, adapt=adapt, **kw)
    mean = getattr(r, 'mean', None)
    if mean is None:
        mean = getattr(r, 'central', r)
    value, sdev, Q = float(mean), float(r.sdev), float(r.Q)

    sob_line, sob_vals = 'NA NA', (None, None)
    if sobol is not None:
        sm, sR = sobol
        sv, se, npts, _ = run_sobol_parallel(
            arrays, eU, eF, a_map, m=sm, R=sR,
            workers=max(1, int(workers or 1)))
        sob_line, sob_vals = '%.8e %.3e' % (sv, se), (sv, se)

    seconds = time.time() - t0
    tag = str(tag).replace(' ', '_') if tag is not None \
        else 'num%d' % int(time.time())
    # inverse-variance combination of the two independent estimators
    # (vegas production + Sobol QMC); >2sigma disagreement = bias alarm
    combined = None
    if sob_vals[0] is not None and sdev > 0:
        wv, ws = 1.0 / sdev ** 2, 1.0 / sob_vals[1] ** 2
        combined = ((value * wv + sob_vals[0] * ws) / (wv + ws),
                    float(1.0 / np.sqrt(wv + ws)))
    total_evals = int((train[0] * train[1] if train else 0)
                      + nitn * neval)
    _append_result(outfile, [
        datetime.datetime.now().isoformat(timespec='seconds'), tag,
        str(rank), str(eU), str(eF), str(info['nvars']), a_str, flag,
        '%.8e' % value,
        '%.3e' % sdev,
        '%.3f' % Q, sob_line,
        str(total_evals), '%.1f' % seconds])
    if not quiet:
        print('[mc_batch] %-18s rank=%d eU=%d eF=%d %s [%s]: '
              '%+.6f +- %.6f (Q=%.2f, %.1f s)%s'
              % (tag, rank, eU, eF, a_str, flag, value, sdev, Q,
                 seconds, '' if sob_vals[0] is None else
                 '  Sobol %+.6f +- %.6f' % sob_vals))
        if combined is not None:
            pull = (value - sob_vals[0]) / np.sqrt(
                sdev ** 2 + sob_vals[1] ** 2)
            print('[mc_batch]   combined %+.6f +- %.6f   '
                  'vegas-vs-Sobol pull = %+.1f sigma%s'
                  % (combined[0], combined[1], pull,
                     '  <-- >2 sigma: check for bias' if abs(pull) > 2
                     else ''))
    return dict(tag=tag, rank=rank, eU=eU, eF=eF, nvars=info['nvars'],
                a=a_vec, a_str=a_str, flag=flag,
                corner_faces=corner_faces, value=value, sdev=sdev, Q=Q,
                sobol=sob_vals, seconds=seconds, info=info,
                arrays=arrays, result=r, combined=combined)

