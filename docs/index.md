# impulse-mcmc

**impulse-mcmc** is a parallel tempering MCMC sampler with adaptive proposals,
vectorized likelihood evaluation, reversible-jump model selection, NUTS
(No-U-Turn Sampler) support, and robust checkpoint/resume.

```python
import numpy as np
from impulse import PTSampler

def log_likelihood(x):
    return -0.5 * np.sum(x**2)

def log_prior(x):
    if np.all(np.abs(x) <= 5):
        return 0.0
    return -np.inf

sampler = PTSampler(ndim=2, lnlike=log_likelihood, lnprior=log_prior,
                    ntemps=8, seed=42)
sampler.sample(np.zeros(2), num_iterations=10_000)
chain = sampler.load_chain()
cold_samples = chain["samples"][0]   # (nsamples, ndim) cold chain
```

## Highlights

- **Parallel tempering** with an automatically constructed, adaptively tuned
  temperature ladder (optionally topped by an infinite-temperature chain).
- **Adaptive proposals** — AM, SCAM, differential evolution, and a
  min-fill-gated EarlyDE variant for reversible-jump runs.
- **Reversible-jump MCMC** via a product-space embedding with an exact
  combined birth/death kernel (`PTSampler.from_rjmcmc`).
- **NUTS** — standalone (`NUTSSampler`) or interleaved with PT and RJ moves
  (`RJPTSampler`), with Stan-convention mass matrices.
- **Checkpoint/resume** that is bit-exact for `PTSampler`/`RJPTSampler`
  checkpoints written by 2.0: an interrupted run resumed to `N` iterations
  reproduces an uninterrupted `N`-iteration run (see the
  [checkpointing guide](user_guide/checkpointing.md) for boundary conditions).
- **Validation** utilities for simulation-based calibration (SBC).

## Installation

```bash
pip install impulse-mcmc
```

See {doc}`user_guide/getting-started` for extras and source installs.

```{toctree}
:maxdepth: 2
:caption: User guide

user_guide/getting-started
user_guide/parallel-tempering
user_guide/rjmcmc
user_guide/nuts
user_guide/checkpointing
user_guide/custom-proposals
```

```{toctree}
:maxdepth: 1
:caption: Examples

examples/index
```

```{toctree}
:maxdepth: 2
:caption: API reference

api/index
```

## Links

- [Source code](https://github.com/AaronDJohnson/impulse_mcmc)
- [Issue tracker](https://github.com/AaronDJohnson/impulse_mcmc/issues)
- [Changelog](https://github.com/AaronDJohnson/impulse_mcmc/blob/main/CHANGELOG.md)
- [Security policy](https://github.com/AaronDJohnson/impulse_mcmc/blob/main/SECURITY.md)
