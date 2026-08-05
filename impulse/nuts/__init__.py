"""
NUTS (No-U-Turn Sampler) subpackage.

Provides a gradient-based MCMC sampler with Stan-style warmup,
mass matrix adaptation, and automatic step size tuning.
"""

from .core import NUTSState, leapfrog, nuts_step
from .gradient_helpers import (
    compose_logp_and_grad,
    make_logp_and_grad_numerical,
    numerical_gradient,
)
from .mass_matrix import MassMatrix, MassMatrixType
from .sampler import NUTSSampler
from .warmup import DualAveraging, WarmupSchedule, find_reasonable_step_size

#: Re-exported names. Declared explicitly so linters see these imports as the
#: subpackage's public surface rather than as unused.
__all__ = [
    "DualAveraging",
    "MassMatrix",
    "MassMatrixType",
    "NUTSSampler",
    "NUTSState",
    "WarmupSchedule",
    "compose_logp_and_grad",
    "find_reasonable_step_size",
    "leapfrog",
    "make_logp_and_grad_numerical",
    "nuts_step",
    "numerical_gradient",
]
