"""
impulse - A Parallel Tempering MCMC Sampler
==========================================

A modular and efficient implementation of parallel tempering MCMC with adaptive
proposals, vectorized likelihood evaluation, and comprehensive diagnostics.

Main Classes
------------
PTSampler :
    Primary interface for parallel tempering MCMC sampling with adaptive proposals,
    checkpoint/resume functionality, and automatic temperature ladder optimization.

RJPTSampler :
    Parallel tempering with optional NUTS and reversible-jump MCMC support.

NUTSSampler :
    No-U-Turn Sampler for gradient-based MCMC.

Quick Start
-----------
>>> import numpy as np
>>> from impulse import PTSampler
>>>
>>> # Define likelihood and prior
>>> def log_likelihood(x):
...     return -0.5 * np.sum(x**2)
>>>
>>> def log_prior(x):
...     return 0.0 if np.all(np.abs(x) <= 5) else -np.inf
>>>
>>> # Create and run sampler
>>> sampler = PTSampler(ndim=2, lnlike=log_likelihood, lnprior=log_prior, ntemps=10)
>>> sampler.sample([0.0, 0.0], num_iterations=10000)

Features
--------
- Parallel tempering for improved mixing in complex posterior landscapes
- Adaptive proposal distributions (AM, SCAM, DE) that learn during sampling
- Vectorized likelihood evaluation for computational efficiency
- Automatic temperature ladder optimization
- Checkpoint and resume functionality for long runs
- Comprehensive convergence diagnostics
- Support for both vectorized and non-vectorized user functions
- RJMCMC model selection with birth/death proposals
- NUTS (No-U-Turn Sampler) for gradient-based transitions

Examples
--------
See the examples/ directory for complete usage examples including:
- Sinusoidal model fitting
- High-dimensional sampling
- RJMCMC model selection
- Hybrid RJMCMC + NUTS + PT sampling
"""

__version__ = "1.0.0"

# Samplers
from .samplers import PTSampler
from .rjpt_sampler import RJPTSampler
from .nuts import NUTSSampler

# RJMCMC
from .rjmcmc import RJMCMCProductSpace
from .product_space import NestedProductSpace

# Diagnostics
from .diagnostics import (
    model_visitation_stats,
    bayes_factor_from_chain,
    grubin,
    effective_sample_size,
    autocorr_length_ips_ims,
)

# Checkpoint / resume
from .resume import (
    checkpoint_sampler,
    load_checkpoint,
    load_nuts_checkpoint,
    load_rjpt_checkpoint,
    check_for_checkpoint,
)

# NUTS utilities
from .nuts import compose_logp_and_grad, make_logp_and_grad_numerical
from .nuts import MassMatrix, MassMatrixType

__all__ = [
    # Samplers
    "PTSampler",
    "RJPTSampler",
    "NUTSSampler",
    # Model spaces
    "RJMCMCProductSpace",
    "NestedProductSpace",
    # Diagnostics
    "model_visitation_stats",
    "bayes_factor_from_chain",
    "grubin",
    "effective_sample_size",
    "autocorr_length_ips_ims",
    # Checkpoint / resume
    "checkpoint_sampler",
    "load_checkpoint",
    "load_nuts_checkpoint",
    "load_rjpt_checkpoint",
    "check_for_checkpoint",
    # NUTS utilities
    "compose_logp_and_grad",
    "make_logp_and_grad_numerical",
    "MassMatrix",
    "MassMatrixType",
]
