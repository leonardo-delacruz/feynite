

import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import sympy as sp

import feynite

from feynite.lorentz import dL
from feynite.constants import (pmom, kmom, _kfun, _pfun)
from timeit import default_timer as timer

from feynite.monte_carlo_integration import  run_numerator

import json


start = timer()
k1 = kmom(1)
k2 = kmom(2)
k3 = kmom(3)

p1 = pmom(1)
p2 = pmom(2)
p3 = pmom(3)



m = sp.Symbol('m')
s12 = sp.Symbol('s12')
s23 = sp.Symbol('s23')

ns = {
    "s12": s12, "s23": s23,
    "dL": dL,
    "k1": kmom(1),
    "k2": kmom(2),
    "k2": kmom(2),
    "p1": pmom(1),
    "p2": pmom(2),
    "p3": pmom(3)
}

kin = {dL(p1, p1): 0, dL(p2, p2): 0, dL(p3, p3): 0, dL(p1, p2): s12 / 2, dL(p2, p3): s23 / 2,
       dL(p1, p3): (-s12 - s23) / 2}
dens = ([k1, k2,k3, k3-p1-p2-p3, k3-p1-p2, k2-p1-p2, k1 - p1- p2, k1 - p1,
         k1 - k2, k2 -k3 ], [0, 0, 0, 0, 0, 0, 0,0,0,0], [k1, k2, k3])

#denstriplebox = {{k[1], k[2], k[3], k[3] - p[1] - p[2] - p[3],
#    k[3] - p[1] - p[2], k[2] - p[1] - p[2], k[1] - p[1] - p[2],
#    k[1] - p[1], k[1] - k[2], k[2] - k[3]},
 #  Table[0, {i, 1, 10}], {k[1], k[2], k[3]}};


with open("triple_box_polynomials.json", "r") as f:
    expr_strings = json.load(f)

nums = [sp.sympify(s, locals=ns, evaluate=False) for s in expr_strings]

#nums = [sp.sympify(s, evaluate=False) for s in expr_strings]
num=nums[-1]
dim=4


parameters={s12: -7.0, s23:-1.0,  sp.Symbol('d'): float(dim)}

run_numerator(num, kin=kin, dens=dens, parameters=parameters, dim=4,  tag=None, outfile='mc_results.txt',
                  train=(20, 50_000), nitn=15 , neval=200_000, adapt=False,
                  alpha=0.3, taming='tamed', sobol=None, workers=1,
                  quiet=False, allow_divergent=False)


end = timer()
print(end - start)
