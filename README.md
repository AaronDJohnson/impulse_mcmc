# impulse-mcmc

A modular and efficient implementation of parallel tempering MCMC with adaptive proposals, vectorized likelihood evaluation, and comprehensive diagnostics.

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

**impulse-mcmc** is a high-performance parallel tempering MCMC sampler designed for efficient exploration of complex posterior distributions. It combines multiple adaptive proposal mechanisms with vectorized computation and robust checkpoint/resume functionality for production-scale Bayesian inference.

## Key Features

- **Parallel Tempering**: Enhanced mixing in complex posterior landscapes through temperature-based chain swapping
- **Adaptive Proposals**: Multiple proposal distributions (AM, SCAM, DE) that automatically adapt during sampling
- **Vectorized Computation**: Efficient likelihood evaluation through vectorized operations
- **Temperature Optimization**: Automatic temperature ladder optimization for optimal acceptance rates
- **Checkpoint/Resume**: Robust checkpointing for long-running inference tasks
- **Comprehensive Diagnostics**: Built-in convergence monitoring and chain statistics
- **Flexible Interface**: Support for both vectorized and scalar user-defined functions

## Installation

Requires Python 3.10 or newer.

```bash
pip install impulse-mcmc
```

The SBC plotting helpers in `impulse.validation` (ECDF, coverage, and rank-histogram plots) need matplotlib, available via the `plots` extra:

```bash
pip install "impulse-mcmc[plots]"
```

Or install from source:

```bash
git clone https://github.com/AaronDJohnson/impulse_mcmc.git
cd impulse_mcmc
pip install -e .            # core
pip install -e ".[plots]"   # with plotting support
```

## Quick Start

```python
import numpy as np
from impulse import PTSampler

# Define likelihood and prior
def log_likelihood(x):
    return -0.5 * np.sum(x**2)

def log_prior(x):
    return 0.0 if np.all(np.abs(x) <= 5) else -np.inf

# Create and run sampler
sampler = PTSampler(ndim=2, lnlike=log_likelihood, lnprior=log_prior, ntemps=10)
sampler.sample([0.0, 0.0], num_iterations=10000)

# Load results from saved chain files
chain = sampler.load_chain()
cold_chain_samples = chain['samples'][0]  # shape (nsamples, ndim)
```

## Advanced Usage

### Vectorized Likelihood Functions

For improved performance with expensive likelihood calculations:

```python
def vectorized_log_likelihood(x_array):
    """
    x_array has shape (n_samples, ndim)
    Returns array of shape (n_samples,)
    """
    return -0.5 * np.sum(x_array**2, axis=1)

sampler = PTSampler(
    ndim=2,
    lnlike=vectorized_log_likelihood,
    lnprior=log_prior,
    ntemps=10,
    vectorized=True
)
```

### Custom Proposal Distributions

```python
# Configure proposal weights (relative, automatically normalized)
sampler = PTSampler(
    ndim=10,
    lnlike=log_likelihood,
    lnprior=log_prior,
    am_weight=15,    # Adaptive Metropolis
    scam_weight=30,  # Single Component AM
    de_weight=50,    # Differential Evolution
)

# Add a custom proposal to all temperature chains
def my_proposal(chain_stats):
    new_sample = chain_stats.current_sample + chain_stats.rng.standard_normal(chain_stats.ndim) * 0.1
    log_proposal_ratio = 0.0  # symmetric proposal
    return new_sample, log_proposal_ratio

sampler.add_custom_jump(my_proposal, weight=25)
```

### Checkpoint and Resume

```python
# Checkpoints are saved automatically during sampling.
# To resume from a previous run, set resume=True:
sampler = PTSampler(
    ndim=2,
    lnlike=log_likelihood,
    lnprior=log_prior,
    outdir="./chains",
    resume=True
)

sampler.sample([0.0, 0.0], num_iterations=50000)
```

## Core Components

### PTSampler
The main interface for parallel tempering MCMC sampling with:
- Adaptive temperature ladders
- Multiple proposal mechanisms
- Checkpoint/resume capabilities

### Proposal Distributions
- **AM**: Adaptive Metropolis with global covariance adaptation
- **SCAM**: Single Component Adaptive Metropolis for high-dimensional problems
- **DE**: Differential Evolution proposals using chain history

### Diagnostics
Built-in convergence diagnostics including:
- Gelman-Rubin statistic
- Effective sample size
- Acceptance rate monitoring
- Temperature swap statistics

## Requirements

- Python ≥ 3.10
- NumPy ≥ 1.24
- SciPy ≥ 1.10
- tqdm ≥ 4.60
- matplotlib (optional — only for the plotting helpers in `impulse.validation`; install via the `plots` extra)

## Examples

Complete examples are available in the `examples/` directory:
- `sinusoid.ipynb` — Sinusoidal model fitting with PTSampler
- `high_dimensional_test.ipynb` — High-dimensional sampling
- `product_space_sinusoids.ipynb` — Product-space RJMCMC
- `rjmcmc_sinusoids.ipynb` — RJMCMC model selection for sinusoids
- `rjmcmc_nuts_sinusoids.ipynb` — Hybrid RJMCMC + NUTS sampling
- `rjpt_sinusoids.ipynb` — RJPTSampler with MH and NUTS comparison

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use impulse-mcmc in your research, please cite:

```bibtex
@software{impulse_mcmc,
  author = {Johnson, Aaron D.},
  title = {impulse-mcmc: A modular parallel tempering MCMC sampler},
  url = {https://github.com/AaronDJohnson/impulse_mcmc},
  version = {1.0.0},
  year = {2025}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/AaronDJohnson/impulse_mcmc/issues)
- **Documentation**: See docstrings and examples for detailed usage
- **Email**: aaron9035@gmail.com