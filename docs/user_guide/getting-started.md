# Getting started

## Installation

impulse-mcmc requires Python 3.10 or newer. The core package depends only on
`numpy`, `scipy`, and `tqdm`:

```bash
pip install impulse-mcmc
```

Optional extras:

```bash
pip install "impulse-mcmc[plots]"   # matplotlib, for the SBC plotting helpers
pip install "impulse-mcmc[flow]"    # coppuccino, for normalizing-flow proposals
pip install "impulse-mcmc[dev]"     # pytest, black, isort, ... for development
```

Or install from source:

```bash
git clone https://github.com/AaronDJohnson/impulse_mcmc.git
cd impulse_mcmc
pip install -e .
```

## Quick start

Define a log-likelihood and a log-prior, construct a
{class}`~impulse.PTSampler`, and sample. Both functions take a single
parameter vector of shape `(ndim,)` and return a float; the prior must
return `-inf` outside its support.

```python
import numpy as np
from impulse import PTSampler

def log_likelihood(x):
    return -0.5 * np.sum(x**2)

def log_prior(x):
    if np.all(np.abs(x) <= 5):
        return 0.0
    return -np.inf

sampler = PTSampler(
    ndim=2,
    lnlike=log_likelihood,
    lnprior=log_prior,
    ntemps=8,            # number of temperature chains
    seed=42,             # reproducible runs
    outdir="./chains",   # chain files + checkpoint go here
)
sampler.sample(np.zeros(2), num_iterations=10_000)

chain = sampler.load_chain()
cold = chain["samples"][0]          # shape (nsamples, ndim): the T=1 chain
print("posterior mean:", cold[2000:].mean(axis=0))
print("posterior std: ", cold[2000:].std(axis=0))
```

### Chain files on disk

Each temperature writes one file in `outdir`. The default encoding is **raw
binary** (`chain_0.bin`, `chain_1.bin`, ...): fixed-width `float64` records,
`ndim + 4` values per row (the parameters, then log-likelihood, log-posterior,
acceptance flag, temperature). It is the default because formatting floats as
text dominated chain I/O -- roughly 18% of wall time on a 21-temperature run --
and the binary files are also ~3x smaller.

`load_chain()` reads either encoding, so most users never need to think about
it. To read one directly:

```python
import numpy as np

ndim = 2
data = np.fromfile("./chains/chain_0.bin").reshape(-1, ndim + 4)
samples, lnlike, lnprob, accepted, temperature = (
    data[:, :ndim], data[:, ndim], data[:, ndim+1], data[:, ndim+2], data[:, ndim+3])
```

Pass `chain_format="text"` to get the historical `chain_<i>.txt` files instead
(`%.18e` columns, readable with `np.loadtxt`, greppable on a cluster). Both
encodings store identical values -- `%.18e` round-trips a `float64` exactly --
so the choice only affects speed, size, and readability. `load_chain()`
detects whichever is present, so a text run stays readable from a
default-configured sampler.

Key points:

- `chain["samples"]` has shape `(ntemps, nsamples, ndim)`. Index `0` is the
  cold (temperature 1) chain — the one that samples your posterior. The
  other chains sample tempered versions of it and exist to improve mixing.
- `chain` also carries `"lnlike"`, `"lnprob"`, `"accepted"`, and
  `"temperature"` arrays of shape `(ntemps, nsamples)`.
- Chains are streamed to `<outdir>/chain_<i>.txt` during the run, and a
  checkpoint is written to `<outdir>/sampler_checkpoint.npz` plus
  `<outdir>/sampler_checkpoint.json` every `save_freq` iterations. Loading one
  executes no code (see {doc}`checkpointing`).
- The initial position must be inside the prior support with finite
  likelihood, otherwise `sample` raises `ValueError`. You can pass a single
  `(ndim,)` vector (replicated across chains) or an `(ntemps, ndim)` array.

## Vectorized likelihoods

Each iteration evaluates the likelihood and prior once per temperature
chain. If your functions can operate on a *batch* of parameter vectors,
pass `vectorized=True` — the sampler then calls them once per iteration
with an array of shape `(n, ndim)` and expects an array of shape `(n,)`
back:

```python
import numpy as np
from impulse import PTSampler

def log_likelihood_v(x):
    # x has shape (n, ndim); return shape (n,)
    return -0.5 * np.sum(x**2, axis=1)

def log_prior_v(x):
    inside = np.all(np.abs(x) <= 5, axis=1)
    return np.where(inside, 0.0, -np.inf)

sampler = PTSampler(
    ndim=2,
    lnlike=log_likelihood_v,
    lnprior=log_prior_v,
    vectorized=True,
    ntemps=8,
    seed=42,
    outdir="./chains_vec",
)
sampler.sample(np.zeros(2), num_iterations=10_000)
```

For expensive likelihoods this removes the per-chain Python loop and lets
NumPy (or your own batched code) do the work in one call.

Notes:

- The prior is evaluated first; rows with `-inf` prior are skipped when the
  likelihood is called, so the likelihood only sees in-support rows.
- Extra positional/keyword arguments can be forwarded with
  `loglargs`/`loglkwargs` and `logpargs`/`logpkwargs`.

### JIT-compiled (JAX) likelihoods

If the likelihood is JAX-traced/JIT-compiled, also pass `jax=True`.
Skipping out-of-prior rows would change the batch shape between iterations
and force JAX to recompile; with `jax=True` the sampler always evaluates
the full `(ntemps, ndim)` batch and masks invalid rows afterwards, keeping
the JIT cache warm. Leave `jax=False` for plain NumPy likelihoods so
out-of-prior rows can genuinely be skipped.

## Where to go next

- {doc}`parallel-tempering` — the temperature ladder, adaptive proposals,
  and when to freeze adaptation.
- {doc}`model-selection` — trans-dimensional sampling and model selection.
- {doc}`nuts` — gradient-based sampling with `NUTSSampler` and
  `HybridPTSampler`.
- {doc}`checkpointing` — resuming long runs bit-exactly.
- {doc}`custom-proposals` — registering your own proposal distributions.
