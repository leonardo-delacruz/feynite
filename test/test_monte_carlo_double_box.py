
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


import sympy as sp

from feynite.lorentz import dL
from feynite.constants import (pmom, kmom)
from timeit import default_timer as timer

from feynite.monte_carlo_integration import  run_numerator

import json

s12 = sp.Symbol('s12')
s23 = sp.Symbol('s23')


ns = {
    "s12": s12, "s23": s23,
    "dL": dL,
    "k1": kmom(1),
    "k2": kmom(2),
    "p1": pmom(1),
    "p2": pmom(2),
    "p3": pmom(3)}


k1, k2 = kmom(1), kmom(2)
p1, p2, p3, p4 = pmom(1), pmom(2), pmom(3), pmom(4)

kin = {dL(p1, p1): 0, dL(p2, p2): 0, dL(p3, p3): 0, dL(p1, p2): s12/2, dL(p2, p3): s23/2,  dL(p1, p3):(-s12-s23)/2}

kin = {dL(p1, p1): 0, dL(p2, p2): 0, dL(p3, p3): 0, dL(p1, p2): s12/2, dL(p2, p3): s23/2,  dL(p1, p3):(-s12-s23)/2}
dens = ([k1, k1-p1, k1-p1-p2, k1 - k2, k2-p1-p2, k2, k2-p1-p2-p3], [0, 0, 0, 0, 0, 0, 0], [k1, k2])

start = timer()

numrank = "2"
with open("double_box_rank" + numrank + "_numerators.json", "r") as f:
    expr_strings = json.load(f)

nums = [sp.sympify(s, locals=ns, evaluate=False) for s in expr_strings]

num=nums[0]
rank=2
dim=4
parameters={s12: -7.0, s23:-1.0,  sp.Symbol('d'): float(dim)}


run_numerator(num, kin=kin, dens=dens, parameters=parameters, dim=4,
                  rank=rank, tag=None, outfile='mc_results.txt',
                  train=(15, 50_000), nitn=5 , neval=200_000, adapt=False,
                  alpha=0.3, taming='tamed', sobol=None, workers=1,
                  quiet=False)

