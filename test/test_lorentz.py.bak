
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_STACK = os.path.dirname(_HERE)
if _STACK not in sys.path:
    sys.path.insert(0, _STACK)

import sympy as sp

from constants import kmom, pmom, muidx, d
from lorentz import (dL, lV, MT, dL_of, lorentz_simplify,
                 set_kinematics, clear_kinematics, get_kinematics,
                 _split_term_factors, _contract_factors,
                 _split_scalar_packable)


def main():
    n = 0

    def check(cond, msg):
        nonlocal n
        assert cond, 'LORENTZ SELF-CHECK FAILED: ' + msg
        n += 1
        print('  ok  %s' % msg)

    print('== lorentz self-check ==')
    k1, k2 = kmom(1), kmom(2)
    p1, p2 = pmom(1), pmom(2)
    mu1, mu2 = muidx(1), muidx(2)
    mu3, mu4 = muidx(3), muidx(4)
    mu9 = muidx(9)
    x, y = sp.Symbol('x'), sp.Symbol('y')

    # ---- dL constructor rules ----------------------------------------------
    check(dL(k1).func is dL and len(dL(k1).args) == 1,
          'dL(x_) is preserved as a ONE-argument head (exactly like the '
          'engine; the pipeline always builds the two-argument form)')
    check(dL(k1, k1).func is dL and len(dL(k1, k1).args) == 2,
          'dL(k1, k1) is the canonical identical-argument invariant')
    check(dL(0, k1) == 0, 'dL[0, x_] := 0')

    check(dL(k2, k1) == dL(k1, k2),
          'Orderless: canonical argument order (k1 < k2)')
    check(dL(2 * k1, k2) == 2 * dL(k1, k2),
          'dL[Times[a_Integer, A__], X__] pulls out numeric factors')
    check(sp.expand(dL(k1 + 2 * k2, p1) - (dL(k1, p1) + 2 * dL(k2, p1))) == 0,
          'dL[sum_Plus, A__] distributes bilinearly')
    a, b = sp.Symbol('a'), sp.Symbol('b')
    check(sp.diff(dL(a, b), a) == b and sp.diff(dL(a, b), b) == a,
          'fdiff: d dL(a,b)/da -> b and d dL(a,b)/db -> a (MINT rules)')

    # ---- lV / MT ------------------------------------------------------------
    check(lV(0, mu1) == 0, 'lV[0, x_] := 0')
    check(sp.expand(lV(k1 - 2 * k2, mu1)
                    - (lV(k1, mu1) - 2 * lV(k2, mu1))) == 0,
          'lV distributes over sums and pulls out numeric factors')
    check(MT(mu1, mu1) == d, 'MT[mu_, mu_] := d')
    check(MT(mu2, mu1) == MT(mu1, mu2), 'MT is Orderless')

    # ---- dL_of ---------------------------------------------------------------
    check(sp.expand(dL_of(k1 + k2, p1) - (dL(k1, p1) + dL(k2, p1))) == 0,
          'dL_of: full bilinear expansion of summed momenta')

    # ---- contraction rules via lorentz_simplify ------------------------------
    check(lorentz_simplify(lV(k1, mu1) * lV(k2, mu1)) == dL(k1, k2),
          'lV[i,mu] lV[j,mu] -> dL(i, j)')
    check(lorentz_simplify(lV(k1, mu1) ** 2) == dL(k1, k1),
          'lV[i,mu]^2 -> dL(i, i)')
    check(lorentz_simplify(MT(mu1, mu2) * lV(k1, mu1)) == lV(k1, mu2),
          'MT[mu,nu] lV[i,mu] -> lV(i, nu)')
    check(lorentz_simplify(MT(mu1, mu2) * MT(mu2, mu1)) == d,
          'MT[mu,nu] MT[nu,mu] -> d (metric cycle)')
    check(lorentz_simplify(MT(mu1, mu2) * MT(mu2, mu3)) == MT(mu1, mu3),
          'MT[mu,nu] MT[nu,rho] -> MT(mu, rho)')
    mixed = 3 * x * lV(k1, mu1) * MT(mu1, mu2) * lV(k2, mu2)
    check(sp.expand(lorentz_simplify(mixed) - 3 * x * dL(k1, k2)) == 0,
          'mixed scalar * lV * MT * lV contracts to scalar * dL')

    # ---- the one-shot and rule-by-rule routes agree ---------------------------
    term = (2 * lV(k1, mu1) * lV(k2, mu1) * MT(mu2, mu3)
            * lV(p1, mu2) * lV(p2, mu3))
    split = _split_term_factors(sp.expand(term))
    one_shot = (split[0] * _contract_factors(split[1])
                if split is not None and _contract_factors(split[1]) is not None
                else None)
    check(one_shot is not None
          and sp.expand(one_shot - lorentz_simplify(term)) == 0,
          '_contract_factors (one-shot) == the rule-by-rule fixpoint')

    # ---- _split_term_factors --------------------------------------------------
    split = _split_term_factors(3 * x * lV(k1, mu1) ** 2 * MT(mu1, mu2))
    check(split is not None
          and sp.expand(split[0] - 3 * x) == 0
          and split[1] == [('L', (k1, mu1)), ('L', (k1, mu1)),
                           ('M', (mu1, mu2))],
          '_split_term_factors -> (scalar, M/L index factors)')
    mu9 = muidx(9)
    check(_split_term_factors(lV(k1, mu9) ** -1) is None,
          '_split_term_factors -> None on lV**-1 (cannot be classified)')

    # ---- _split_scalar_packable (the deferred channel) -------------------------
    pack, defer = _split_scalar_packable(3 * sp.sqrt(2) / y)
    check(pack == 3 and defer == sp.sqrt(2) / y
          and sp.expand(pack * defer - 3 * sp.sqrt(2) / y) == 0,
          '_split_scalar_packable: 3*sqrt(2)/y -> pack 3, defer sqrt(2)/y')
    pack, defer = _split_scalar_packable(dL(k1, k2) * x ** 2)
    check(pack == x ** 2 and defer == dL(k1, k2),
          '_split_scalar_packable: dL heads and other atoms stay deferred')

    # ---- the kinematics registry ----------------------------------------------
    s12 = sp.Symbol('s12')
    set_kinematics({dL(p1, p1): 0, dL(p1, p2): s12 / 2})
    check(dL(p1, p2) == s12 / 2 and dL(p2, p1) == s12 / 2
          and dL(p1, p1) == 0,
          'set_kinematics: dL between external momenta evaluates AT '
          'CONSTRUCTION TIME (no compound dL head enters any tree)')
    check(get_kinematics() == {(p1, p1): sp.Integer(0),
                               (p1, p2): s12 / 2},
          'get_kinematics returns the registered (canonical-key) rules')
    clear_kinematics()
    check(dL(p1, p2).func is dL,
          'clear_kinematics: back to symbolic invariants')

    raised = None
    try:
        set_kinematics({kmom(1): 0})
    except ValueError:
        raised = 'value'
    check(raised == 'value',
          'set_kinematics: a key that is not dL(x, y) raises ValueError')
    raised = None
    try:
        set_kinematics({dL(k1, k2): s12})
    except ValueError:
        raised = 'loop'
    check(raised == 'loop',
          'set_kinematics: a rule containing a loop momentum k[i] raises '
          'ValueError (loop invariants stay symbolic)')

    print('== %d checks passed ==' % n)
    return n


if __name__ == '__main__':
    main()