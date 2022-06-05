import numpy as np
from enterprise.signals.parameter import sample as sample_params

# PTA specific functions

def initial_sample(pta, ntemps=None, nchains=None):
    if ntemps is not None and nchains is not None:  # both pt and parallel
        x0 = [[np.array(list(sample_params(pta.params).values())) for _ in range(ntemps)] for _ in range(nchains)]
        ndim = len(x0[0][0])
    elif ntemps is None and nchains is not None:  # no ptsampling
        x0 = [np.array(list(sample_params(pta.params).values())) for _ in range(nchains)]
        ndim = len(x0[0])
    elif ntemps is not None and nchains is None:  # no parallel
        x0 = [np.array(list(sample_params(pta.params).values())) for _ in range(ntemps)]
        ndim = len(x0[0])
    else:  # normal sampling without pt or parallel
        x0 = np.array(list(sample_params(pta.params).values()))
        ndim = len(x0)
    return ndim, x0


