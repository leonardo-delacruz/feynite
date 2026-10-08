

from . import constants, general_tools, lorentz, sparse_tools, symanzik
from . import sparse_evaluator

from .constants import d, L, N_EDGES, U_POL, F_POL, MU, kmom, pmom, z, sigma, muidx
from .lorentz import (dL, lV, MT, dL_of, lorentz_simplify, set_kinematics,
                      clear_kinematics, get_kinematics,
                      split_term_factors, split_scalar_packable)
from .sparse_tools import (sp_from_expr, spmul, spadd, sp_drop_z, sp_emit,
                           Qvec_vector_pieces, slot_to_arrays,
                           SparseRouteError, Fraction)
from .symanzik import (UFdata, Symanzik, adjA, WickContraction, wick2,
                       generate_tensor_structures, unContractnumeratorList,
                       toDataNum, bmat2, cP3, tensor_rank, map_ufdata,
                       uncontract_term, polst_highest_rank, pipeline_setup)
from .sparse_evaluator import SparseNumEvaluator

# The whole-numerator routes are resolved LAZILY (PEP 562): `import nums`
# does NOT load num_evaluator, and `nums.pol_numerator` triggers the load
# on first touch -- mirroring the lazy forward inside
# pure_sparse_evaluator and keeping the package import graph acyclic.
_NUM_EVALUATOR_ROUTES = ('pol_numerator', 'all_terms')


def __getattr__(name):
    if name in _NUM_EVALUATOR_ROUTES:
        from . import num_evaluator
        return getattr(num_evaluator, name)
    raise AttributeError('module %r has no attribute %r'
                         % (__name__, name))

version = '1.0.0'

__all__ = [
    # vocabulary
    'd', 'L', 'N_EDGES', 'U_POL', 'F_POL', 'MU', 'kmom', 'pmom', 'z', 'sigma',
    'muidx',
    # lorentz
    'dL', 'lV', 'MT', 'dL_of', 'lorentz_simplify', 'set_kinematics',
    'clear_kinematics', 'get_kinematics', 'split_term_factors',
    'split_scalar_packable',
    # sparse toolbox
    'sp_from_expr', 'spmul', 'spadd', 'sp_drop_z', 'sp_emit',
    'Qvec_vector_pieces', 'slot_to_arrays', 'SparseRouteError', 'Fraction',
    # symanzik / pipeline
    'UFdata', 'Symanzik', 'adjA', 'WickContraction', 'wick2', 'generate_tensor_structures',
    'unContractnumeratorList', 'toDataNum', 'bmat2', 'cP3', 'tensor_rank',
    'map_ufdata', 'uncontract_term', 'polst_highest_rank',
    'pipeline_setup',
    # pure sparse route
    'SparseNumEvaluator', 'pol_numerator', 'all_terms',
    'version',
]