# -*- coding: utf-8 -*-
"""sparse_evaluator.py

"""

from fractions import Fraction

import sympy as sp

from .constants import (
    d, N_EDGES, L, z, sigma, _pfun, _adjfun,
)
from .lorentz import (
    dL_of, _split_term_factors, _split_scalar_packable,
)
from .general_tools import (
    gamma_reduce, _apply_kinematics, _drop_gamma,
)
from .symanzik import (
    WickContraction, wick2,
)
from .sparse_tools import (
    SparseRouteError,
    # the sparse primitives (module-level, fallback-free)
    _sp_from_expr, _sparse_multiplication, _sp_drop_z, _Qvec_vector_pieces,_sparse_to_sympy
)
# ---------------------------------------------------------------------------
# the sparse evaluator
# ---------------------------------------------------------------------------
class SparseNumEvaluator(object):
    """
The main class for calculation of the tensor representation using sparse arrays
    """
    def __init__(self, datas, highestRank, tensorsUptoHighest, nprops,
                 loops, kin, no_gauge, edges):
        self.datas = datas
        self.highestRank = highestRank
        self.tensorsUptoHighest = tensorsUptoHighest
        self.nprops = nprops
        self.loops = loops
        self.kin = kin
        self.no_gauge = no_gauge
        self.edges = edges
        self.threshold = nprops - highestRank // 2
        self.zedge = z(edges)
        self.n_hi = (len(tensorsUptoHighest[highestRank][0])
                     if highestRank != 0 else None)
        dens, ufdata, adjugates = datas
        self.ufdata = ufdata
        self.adjugates = adjugates

        # --  Loop momenta should be different in the list dens[2]
        loop_moms = list(dens[2])
        if len(set(loop_moms)) != len(loop_moms):
            raise SparseRouteError(
                'constructor',
                'the loop momenta dens[2] = %s are not distinct: Mmatrix is '
                'singular, UFdata degenerates to U == F == 0 and every '
                'term would be silently zero -- the loop momenta must '
                'be distinct symbols' % (loop_moms,))
        zero_dens = [i for i, q in enumerate(dens[0]) if sp.expand(q) == 0]
        if zero_dens:
            raise SparseRouteError(
                'constructor',
                'denominator(s) %s of dens[0] are identically zero '
                '(e.g. two equal loop momenta in a k_i - k_j entry): '
                'the propagator list is degenerate and the ufdata '
                'would silently vanish' % (zero_dens,))

        # cP3 vector decomposition:  cP3(s, rho) = Sum_v Z[s][v] lV(v, rho)
        Qvec_pieces = {}
        for i in range(loops):
            pp = _Qvec_vector_pieces(ufdata[2][i])
            if pp is None:
                raise SparseRouteError(
                    'constructor',
                    'Qvec_%d is not a Sum of coeff*lV(vec, MU) terms -- '
                    'the cP3 vector decomposition cannot be built' % i,
                    sparse=None)
            Qvec_pieces[i] = pp
        Zsp = {}
        for s in range(1, loops + 1):
            acc = {}
            for i in range(loops):
                a = adjugates[s - 1, i]
                if a == 0:
                    continue
                for coeff, vec in Qvec_pieces[i]:
                    acc[vec] = acc.get(vec, sp.S.Zero) + a * coeff
            zd = {}
            for vec, val in acc.items():
                sd = _sp_from_expr(val)
                if sd is None:
                    raise SparseRouteError(
                        'constructor',
                        'the adjugate*Qvec combination Z[%d][%s] = %s is not '
                        'sparsely packable' % (s, vec, val))
                if sd:
                    zd[vec] = sd
            Zsp[s] = zd
        self.Zsp = Zsp

        # sparse U_POL / F_POL values; their power tables are built on demand
        self.sp0 = _sp_from_expr(ufdata[0])
        if self.sp0 is None:
            raise SparseRouteError(
                'constructor',
                'U_POL (ufdata[0]) is not sparsely packable: %s' % (ufdata[0],))
        self.sp1 = _sp_from_expr(ufdata[1])
        if self.sp1 is None:
            raise SparseRouteError(
                'constructor',
                'F_POL (ufdata[1]) is not sparsely packable: %s' % (ufdata[1],))
        self._uc_pows = {0: {0: Fraction(1)}}
        self._fc_pows = {0: {0: Fraction(1)}}
        self._pab_cache = {}
        # component-level rs memo for the block-factorised piece values:
        # key ('L', u, w) -> sparse(kin(dL_of(u, w))) or None
        self._rsmemo = {}
        self._sp_d = _sp_from_expr(d)
        # exactness guard for the gauge placement (see term_parts): the
        # z(edges)->1 substitution is applied to the per-structure acc
        # instead of the per-rk merged pieces; that is the same ring
        # homomorphism ONLY when no z(edges) can enter through the
        # kinematic rules (checked here once) -- exotic  kinematics
        # are REFUSED at the gauge-kin site by term_parts.
        self._kin_zsafe = not any(
            sp.sympify(v).has(self.zedge)
            for v in (kin.values() if isinstance(kin, dict)
                      else (b for _a, b in kin))) if kin else True
        # label-level cache: many numerator monomials share the same
        # loop-label word, and EVERYTHING piece-level -- the prepared
        # structure, the piece splits, the cP3 vector combinations and
        # their prefix products -- depends on the label only.
        self.label_cache = {}

    # -- component values --------------------------------------------------
    def _rs_lvpair(self, u, w):
        """sparse(kin(dL_of(u, w))) for ONE contraction component.
        Memoised; None when the value is not sparsely representable
        (term_parts then refuses the structure at the 'piece' site).
        An empty dict is a genuine zero component.  The 400000-entry cap
        only stops CACHING -- the value is still computed and returned."""
        key = ('L', u, w)
        memo = self._rsmemo
        ent = memo.get(key, False)
        if ent is not False:
            return ent
        val = _apply_kinematics(dL_of(u, w), self.kin)
        sd = _sp_from_expr(val)
        if sd is not None and len(memo) < 400000:
            memo[key] = sd
        return sd

    # -- per-label structure preparation -----------------------------------
    def _label_structs(self, setLoop):
        """Per-structure parse for one loop-label word (cached)."""
        key = tuple(setLoop)
        ent = self.label_cache.get(key)
        if ent is None:
            ent = self._build_label_structs(setLoop)
            if len(self.label_cache) < 4096:
                self.label_cache[key] = ent
        return ent

    def _build_label_structs(self, setLoop):
        r = len(setLoop)
        gentensors, _factor = self.tensorsUptoHighest[r]
        sig_rules = {sigma(i + 1): setLoop[i] for i in range(r)}
        out = []
        tt_cache = {}
        for t in gentensors:
            tt = t.subs(sig_rules)
            tt = tt.replace(lambda e: isinstance(e, WickContraction), lambda e: wick2(e))
            tt = tt.subs({N_EDGES: self.nprops, L: self.loops})
            tt = gamma_reduce(tt, self.threshold)
            if tt == 0:
                out.append(('zero',))
                continue
            if tt in tt_cache:
                out.append(tt_cache[tt])
                continue
            st = self._parse_structure(tt, r)
            tt_cache[tt] = st
            out.append(st)
        return out

    def _parse_structure(self, tt, r):
        """Split a prepared structure into pieces and pre-decompose the
        cP3 slots ONCE (shared by every term with this label).  Returns
            ('zero',)              -- the whole structure is zero
            ('bad', tt, why)       -- piece does not fit: REFUSED by
                                      term_parts (the hybrid would use
                                      its classic route here)
            ('fast', b, items, tt) -- per-term walkable pieces, with
                 items = [(mts, slots, ssp), ...],
                 slots = [(mu, [(v, zd_v), ...]), ...] per cP3 slot

        The per-term evaluation factorises the cP3 choice sum over the
        slot-pairing blocks (see `_piece_value`), so the exponential
        combination space is NEVER enumerated here.  The walk stops at
        the first 'EMPTY' piece (cP3(s) == 0): the structure evaluates
        to zero, exactly like the original sequential evaluation."""
        Zsp = self.Zsp
        adjugates = self.adjugates
        pieces = []
        bset = set()
        for piece in sp.Add.make_args(sp.expand(tt)):
            split = _split_term_factors(piece)
            if split is None:
                return ('bad', tt,
                        'a structure piece does not split into scalar * '
                        'lV/MT index factors (non-integer lV/MT power?)')
            scalar_m, facs = split
            mts = tuple(f[1] for f in facs if f[0] == 'M')
            plv = [f[1] for f in facs if f[0] == 'L']
            if any(v.func != _pfun for (v, _m) in plv):
                return ('bad', tt,
                        'an lV vector head inside the structure is not '
                        'the P[s] placeholder: %s'
                        % ([v for v, _m in plv if v.func != _pfun],))
            p_exp = (r - len(plv)) // 2
            bset.add(p_exp)
            # scalar: adjugate entries in, Gamma out, F_POL^p stripped
            scalar_m = scalar_m.replace(
                lambda e: getattr(e, 'func', None) == _adjfun,
                lambda e: adjugates[int(e.args[0]) - 1,
                                    int(e.args[1]) - 1])
            scalar_m = _drop_gamma(scalar_m)
            ssp = _sp_from_expr(scalar_m, fc_pow=p_exp)
            if ssp is None:
                return ('bad', tt,
                        'the piece scalar (with F_POL^%d stripped) is not '
                        'sparsely packable: %s' % (p_exp, scalar_m))
            # per-slot vector decompositions of the cP3 slots: the
            # choice space is stored per slot and FACTORISED per term
            slots = []
            for (v, mu) in plv:
                sarg = v.args[0]
                if not getattr(sarg, 'is_Integer', False):
                    return ('bad', tt,
                            'the cP3 slot label %s is not an integer'
                            % (sarg,))
                zd = Zsp.get(int(sarg))
                if not zd:
                    # cP3(s) == 0 for this loop label: the walk stops
                    # here (the structure evaluates to zero)
                    pieces.append((mts, 'EMPTY', None))
                    return self._finish_parse(tt, bset, pieces)
                slots.append((mu, [(vv, zd[vv]) for vv in zd]))
            pieces.append((mts, slots, ssp))
        return self._finish_parse(tt, bset, pieces)

    @staticmethod
    def _finish_parse(tt, bset, pieces):
        if not bset:
            return ('zero',)                   # nothing survived
        if len(bset) > 1:              # inconsistent F_POL powers: refuse
            return ('bad', tt,
                    'inconsistent F_POL powers across the structure pieces')
        return ('fast', bset.pop(), pieces, tt)

    # -- block-factorised piece values --------------------------------------
    def _piece_value(self, mts, slots, cfacts):
        """Block-factorised value of ONE structure piece for a term with
        external index factors `cfacts` (the ssp scalar NOT included).

        The piece is  Sum_{v-choices} prod_i zd_i(v_i) * C(factors)  with
        C the full contraction of the piece factors (mts + slot lV's +
        cfacts).  Every Lorentz index occurs exactly twice, so C is a
        product of component atoms (dL_of(v, w) chains, dL(v, v), d) and
        every cP3 slot lives in EXACTLY ONE component.  rs (the sparse
        kin-substituted value) is a ring homomorphism away from C, hence

            Sum_choices prod_i zd_i * rs(C)
                = prod_over_blocks [ Sum_small prod zd * prod rs ]

        where the blocks are the slot-pairing classes: 2-slot components
        give |Z_a|*|Z_b| sums, 1-slot components |Z_s| sums.  The
        exponential choice space is never enumerated.

        Returns the sparse piece dict, or None when the contraction does
        not fit the fully-contracted pattern -- term_parts REFUSES at the
        'piece' site then (the hybrid would use its classic route)."""
        facs = []
        for mt in mts:
            facs.append(('M', mt))
        for si, (mu, _zd) in enumerate(slots):
            facs.append(('S', (si, mu)))
        facs.extend(cfacts)
        occ = {}
        for fi, (kind, args) in enumerate(facs):
            mus = args if kind == 'M' else (args[1],)
            for pos, mu in enumerate(mus):
                lst = occ.get(mu)
                if lst is None:
                    occ[mu] = [(fi, pos)]
                else:
                    lst.append((fi, pos))
        link = {}
        for lst in occ.values():
            if len(lst) == 2:
                (i, si), (j, sj) = lst
                link[(i, si)] = (j, sj)
                link[(j, sj)] = (i, si)
            else:
                return None          # free / multiply-paired index
        const = {0: Fraction(1)}    # components without cP3 slots
        singles = {}                 # slot -> ('vv',) | ('vw', w)
        pairs = []                   # (slot_a, slot_b)
        seen = [False] * len(facs)
        rs_lvpair = self._rs_lvpair
        sp_d = self._sp_d
        for start in range(len(facs)):
            if seen[start]:
                continue
            comp = []
            stack = [start]
            seen[start] = True
            while stack:
                fi = stack.pop()
                comp.append(fi)
                kind, _args = facs[fi]
                for pos in (range(2) if kind == 'M' else (0,)):
                    nxt = link.get((fi, pos))
                    if nxt is not None and not seen[nxt[0]]:
                        seen[nxt[0]] = True
                        stack.append(nxt[0])
            sl = [facs[fi][1][0] for fi in comp if facs[fi][0] == 'S']
            cl = [facs[fi][1][0] for fi in comp if facs[fi][0] == 'L']
            nlv = len(sl) + len(cl)
            if nlv > 2:
                return None
            if nlv == 0:
                const = _sparse_multiplication(const, sp_d)          # closed MT cycle
            elif nlv == 1:
                if sl:
                    singles[sl[0]] = ('vv',)
                else:
                    rs = rs_lvpair(cl[0], cl[0])
                    if rs is None:
                        return None
                    const = _sparse_multiplication(const, rs)
            else:                    # exactly 2 lV's in the component
                if len(sl) == 2:
                    pairs.append((sl[0], sl[1]))
                elif len(sl) == 1:
                    singles[sl[0]] = ('vw', cl[0])
                else:
                    rs = rs_lvpair(cl[0], cl[1])
                    if rs is None:
                        return None
                    const = _sparse_multiplication(const, rs)
        piece = const
        for si, kind in singles.items():
            poly = {}
            _mu, zl = slots[si]
            if kind[0] == 'vv':
                for v, zd in zl:
                    rs = rs_lvpair(v, v)
                    if rs is None:
                        return None
                    if not rs:
                        continue
                    for k, c in _sparse_multiplication(zd, rs).items():
                        poly[k] = poly.get(k, 0) + c
            else:
                w = kind[1]
                for v, zd in zl:
                    rs = rs_lvpair(v, w)
                    if rs is None:
                        return None
                    if not rs:
                        continue
                    for k, c in _sparse_multiplication(zd, rs).items():
                        poly[k] = poly.get(k, 0) + c
            if not poly:
                return {}            # every choice of this slot is zero
            piece = _sparse_multiplication(piece, poly)
        for (a, b) in pairs:
            _mua, zla = slots[a]
            _mub, zlb = slots[b]
            poly = {}
            for va, zda in zla:
                for vb, zdb in zlb:
                    rs = rs_lvpair(va, vb)
                    if rs is None:
                        return None
                    if not rs:
                        continue
                    core = _sparse_multiplication(zda, zdb)
                    if len(rs) == 1 and 0 in rs:
                        cf1 = rs[0]
                        for k, c in core.items():
                            poly[k] = poly.get(k, 0) + c * cf1
                    else:
                        for k, c in _sparse_multiplication(core, rs).items():
                            poly[k] = poly.get(k, 0) + c
            if not poly:
                return {}            # every choice pair is zero
            piece = _sparse_multiplication(piece, poly)
        return piece

    # -- shared U_POL^a F_POL^b products ------------------------------------------
    def _power(self, store, base, n):
        """store[n] = base**n as a sparse dict, built incrementally."""
        cur = store.get(n)
        if cur is not None:
            return cur
        best = 0
        for k in store:
            if best < k < n:
                best = k
        cur = store[best]
        for _ in range(n - best):
            cur = _sparse_multiplication(cur, base)
        store[n] = cur
        return cur

    def _pab(self, a_exp, b):
        """The shared U_POL^a F_POL^b factor as one sparse dict (cached)."""
        key = (a_exp, b)
        if key not in self._pab_cache:
            self._pab_cache[key] = _sparse_multiplication(
                self._power(self._uc_pows, self.sp0, a_exp),
                self._power(self._fc_pows, self.sp1, b))
        return self._pab_cache[key]

    # -- the per-term contract ----------------------------------------------
    def term_parts(self, momentaNumerator, setLoop, sparse=True):
        """ONE numerator term -> one entry per array slot, either

            ('fast', acc, csp, a_exp, b, c_def)
                -- slot value = (acc * csp) * U_POL^a F_POL^b * c_def, with
                   c_def the DEFERRED scalar prefactor (1 normally): the
                   caller applies it ONCE to its final result, keeping
                   every dict operation on packed keys / Fractions
            ('final', value)   -- an EXACT ZERO (the pure route has no
                   classic fallback, so 'final' parts are always zero:
                   the 'zero' structures and the 'EMPTY' pieces)

        (value is a sparse dict for sparse=True else the SymPy zero; the
        sparse flag changes the ZERO SENTINEL only -- the fast route
        always computes dicts, and `term` emits expressions from them.)"""
        zero = {} if sparse else sp.S.Zero
        ufdata = self.ufdata
        kin = self.kin
        highestRank = self.highestRank
        nprops = self.nprops
        loops = self.loops
        no_gauge = self.no_gauge
        zedge = self.zedge
        r = len(setLoop)
        a_exp = highestRank - r

        # -- refusal site 2a: the numerator split --------------------------
        csplit = _split_term_factors(sp.expand(momentaNumerator))
        if csplit is None:
            raise SparseRouteError(
                'numerator-split',
                'the term does not split into a scalar times positive '
                'integer powers of lV/MT index factors',
                momentaNumerator=momentaNumerator, setLoop=setLoop,
                sparse=sparse)
        c_scalar, c_factors = csplit
        c_scalar = _apply_kinematics(c_scalar, kin)
        # packable part rides the sparse arithmetic; the DEFERRED rest
        # (negative exponents such as 1/s23, irrational numbers, ...) is
        # a multiplicative constant applied ONCE by the caller
        c_pack, c_def = _split_scalar_packable(c_scalar)
        if not no_gauge:
            c_def = c_def.subs(zedge, 1)
        cfacts = tuple(c_factors)

        # -- refusal sites 2b/2c: the exactness guards of the gauge --------
        if not no_gauge and not self._kin_zsafe:
            raise SparseRouteError(
                'gauge-kin',
                'no_gauge=False but a kinematic rule value contains '
                'z(edges): the deferred z(edges)->1 substitution would '
                'not be a ring homomorphism on the sparse values',
                momentaNumerator=momentaNumerator, setLoop=setLoop,
                sparse=sparse)
        if not no_gauge and any(f[0] == 'L' and f[1][0].has(zedge)
                                for f in cfacts):
            raise SparseRouteError(
                'gauge-term',
                'no_gauge=False and an lV vector of the term itself '
                'carries z(edges): the gauge substitution cannot be '
                'deferred to the accumulated value',
                momentaNumerator=momentaNumerator, setLoop=setLoop,
                sparse=sparse)

        structs = self._label_structs(setLoop)
        piece_value = self._piece_value
        drop_z = _sp_drop_z

        out = []
        for si, st in enumerate(structs):
            # -- exact zeros (NOT refusals) --------------------------------
            if st[0] == 'zero':
                out.append(('final', zero))
                continue
            # -- refusal site 3: the structure does not fit -----------------
            if st[0] == 'bad':
                raise SparseRouteError(
                    'structure', st[2],
                    momentaNumerator=momentaNumerator, setLoop=setLoop,
                    si=si, sparse=sparse)
            _, b, items, _tt = st
            acc = {}
            dead = False
            for (mts, slots, ssp) in items:
                if slots == 'EMPTY':
                    # cP3(s) == 0 for this loop label: the structure
                    # (which carries this lV factor) is exactly zero
                    dead = True
                    break
                # -- refusal site 4: the component walk ----------------------
                piece = piece_value(mts, slots, cfacts)
                if piece is None:
                    raise SparseRouteError(
                        'piece',
                        'a contraction component does not fit the '
                        'fully-contracted pattern: a Lorentz index occurs '
                        'not exactly twice, a component carries more than '
                        'two lV factors, or a required kin(dL(u, w)) value '
                        'is not sparsely representable',
                        momentaNumerator=momentaNumerator, setLoop=setLoop,
                        si=si, sparse=sparse)
                if not piece:
                    continue
                for k, c in _sparse_multiplication(ssp, piece).items():
                    acc[k] = acc.get(k, 0) + c
            if dead:
                out.append(('final', zero))
                continue
            if not no_gauge:
                # z(edges) -> 1 on the accumulated value
                acc = drop_z(acc, zedge)
            csp = _sp_from_expr(c_pack)
            if csp is None:              # unreachable: c_pack is
                csp = {0: c_pack}        # packable by construction
            if not no_gauge:
                csp = _sp_drop_z(csp, zedge)
            out.append(('fast', acc, csp, a_exp, b, c_def))

        # zero padding up to the number of structures at the highest rank
        if self.n_hi is not None:
            out += [('final', zero)] * (self.n_hi - len(out))
        return out

    def term(self, momentaNumerator, setLoop, sparse=False):
        """numSingleTerm for ONE numerator term: term_parts with the
        U_POL^a F_POL^b factor and the deferred scalar applied per structure
        (the standalone API).  sparse=True returns the dicts,
        sparse=False the expanded SymPy expressions emitted from them."""
        out = []
        for p in self.term_parts(momentaNumerator, setLoop, sparse):
            if p[0] == 'final':
                out.append(p[1])
                continue
            _, acc, csp, a_exp, b, c_def = p
            fin = _sparse_multiplication(_sparse_multiplication(acc, self._pab(a_exp, b)), csp)
            if c_def != 1:
                fin = {k: cf * c_def for k, cf in fin.items()}
            out.append(fin if sparse else _sparse_to_sympy(fin))
        return out

    def report_term(self, momentaNumerator, setLoop):
        """term_parts in REPORT mode: print the SparseRouteError and
        return None instead of raising (the print-and-carry-on route)."""
        try:
            return self.term_parts(momentaNumerator, setLoop, sparse=True)
        except SparseRouteError as e:
            print(e)
            return None

"""
# ---------------------------------------------------------------------------
# backward-compat forward: the whole-numerator routes live in
# num_evaluator.py now.  Importing them from HERE still works -- the
# module-level __getattr__ below resolves them lazily on first access,
# so there is no import cycle between the two modules.
# ---------------------------------------------------------------------------
_MOVED_ROUTES = ('_build_evaluator', '_accumulate_terms',
                 'all_terms', 'pol_numerator')


def __getattr__(name):
    if name in _MOVED_ROUTES:
        import num_evaluator
        return getattr(num_evaluator, name)
    raise AttributeError('module %r has no attribute %r (the whole-'
                         'numerator routes live in num_evaluator: '
                         'all_terms, pol_numerator)' % (__name__, name))
"""