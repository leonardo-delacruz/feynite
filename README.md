# feynite 

A package for parametric tensor representacion of Feynman integrals and Monte Carlo
evaluation using Vegas. Numerical integration is focused only on IR/UV finite
integrals in Euclidean kinematics.


Finite numerators can be constructed from the algorithms described in the references

- [Finite integrals from Feynman polytopes](https://arxiv.org/pdf/2311.16907)

- [Finite Feynman integrals](https://arxiv.org/pdf/2410.18014)


The parametric tensor representation is based on Appendix A of the first reference




## Installation 

From the project directory, run:


`python -m pip install . `



## Quick start

`import sympy as sp`

`from feynite.lorentz import dL`

`from feynite.constants import (pmom, kmom)`

`from feynite.num_evaluator import pol_numerator`

`from feynite.monte_carlo_integration import run_numerator`


###  Invariants

`s = sp.Symbol('s')`


### vectors
`k1, k2 = kmom(1), kmom(2)`

`p1  = pmom(1)`


###  kinematic rules
`kin = {dL(p1, p1): s}`

`dens= ([k1, k2, k1-k2+p1], [0,0,0],[k1,k2])`


### tensor representation

`num=dL(k1,p1)**2`

`polynomial_num=sum(pol_numerator(num,dens, kin, sparse=False))`

`print(polynomial_num)`








