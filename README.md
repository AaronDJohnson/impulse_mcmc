# impulse-mcmc

A modular and efficient implementation of parallel tempering MCMC with adaptive proposals, vectorized likelihood evaluation, and comprehensive diagnostics.

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-github%20pages-blue.svg)](https://aarondjohnson.github.io/impulse_mcmc/)

## Overview

**impulse-mcmc** is a parallel tempering MCMC sampler for production-scale Bayesian
inference. **`PTSampler`** combines adaptive Metropolis-Hastings proposals (AM,
SCAM, differential evolution, optional normalizing flows), automatic
temperature-ladder adaptation, and batched (vectorized or JAX-jitted)
likelihood evaluation. Every run checkpoints and resumes bit-exactly, in a
format that executes no code on load. **`RJPTSampler`** extends it to reversible-jump
sampling of a variable number of components.

## Installation

Requires Python ≥ 3.10. Core dependencies are just NumPy, SciPy, and tqdm.

```bash
pip install impulse-mcmc              # core
pip install "impulse-mcmc[flow]"      # + coppuccino for normalizing-flow proposals (Python 3.11+)
pip install "impulse-mcmc[dev]"       # + test and lint tooling
```

Or from source:

```bash
git clone https://github.com/AaronDJohnson/impulse_mcmc.git
cd impulse_mcmc
pip install -e ".[flow,dev]"
```

## Quick start

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

# Load results from the saved chain files
chain = sampler.load_chain()
cold_chain_samples = chain["samples"][0]  # shape (nsamples, ndim)
```

Chains, acceptance-rate reports, and checkpoints are written to `outdir`
(default `./chains`). Chain files are raw `float64` records (`chain_<i>.bin`,
`ndim + 4` columns) — ~18% faster to write and ~3x smaller than text. Pass
`chain_format="text"` for the human-readable `chain_<i>.txt` form; `load_chain()`
reads either.

## Vectorized likelihoods

For expensive models, evaluate all temperature chains in one batched call by
passing `vectorized=True`. Both the likelihood **and** the prior must then accept
a `(n, ndim)` array and return a length-`n` array:

```python
import numpy as np
from impulse import PTSampler

def log_likelihood_vec(x):
    # x has shape (n, ndim); return shape (n,)
    return -0.5 * np.sum(x**2, axis=1)

def log_prior_vec(x):
    return np.where(np.all(np.abs(x) <= 5, axis=1), 0.0, -np.inf)

sampler = PTSampler(
    ndim=2, lnlike=log_likelihood_vec, lnprior=log_prior_vec,
    ntemps=10, vectorized=True, outdir="./chains_vec",
)
sampler.sample(np.zeros(2), num_iterations=5000)
```

If the likelihood is JAX-jitted, also pass `jax=True`: the sampler then keeps the
batch shape constant on every call (masking invalid rows afterwards) so the JIT
cache is reused instead of recompiling.

## Parallel tempering features

- **Adaptive proposals** — a weighted mixture of adaptive Metropolis (`am_weight`),
  single-component adaptive Metropolis (`scam_weight`), and differential evolution
  (`de_weight`) that learn the target's covariance and history as sampling proceeds.
- **Min-fill-gated DE** — the `de` move activates once the history buffer
  holds `de_min_fill` rows (default 100, about 2,500 iterations at the default
  `buffer_thin=25`) instead of waiting for a completely full buffer, which keeps
  the history-based, ridge-following DE move available at realistic run
  lengths. Below the threshold `de` returns the current position unchanged (no
  hidden substitution).
- **Temperature-ladder adaptation** — the geometric ladder adapts toward uniform
  swap acceptance between neighbours (`adapt_t0`, `adapt_nu` control the schedule).
- **`inf_temp=True`** — replaces the hottest rung with a T = ∞ chain that samples
  the prior (`ntemps` total chains, `ntemps - 1` finite), improving hot-chain
  mixing and enabling prior-dominated moves to propagate down the ladder.
- **`num_adapt`** — freezes **all** adaptation (proposal covariances, DE buffers,
  ladder adaptation, flow refits) once the global
  iteration counter reaches `num_adapt`. The transition kernel is fixed from then
  on, so post-freeze samples are exactly Markovian. Recommended usage: set
  `num_adapt` to your intended warmup length and discard all pre-freeze samples as
  warmup. The default `None` adapts for the whole run — see
  [Adaptation and ergodicity](#adaptation-and-ergodicity) for what that does and
  does not guarantee.

```python
sampler = PTSampler(
    ndim=2, lnlike=log_likelihood, lnprior=log_prior,
    ntemps=8,                   # geometric ladder starting at min_temp
    inf_temp=True,              # hottest rung becomes T = inf (samples the prior)
    adapt_t0=100, adapt_nu=10,  # temperature-ladder adaptation schedule
    num_adapt=5000,             # freeze ALL adaptation at iteration 5000
    seed=123, outdir="./chains_pt",
)
sampler.sample(np.zeros(2), num_iterations=8000)  # discard the first 5000 as warmup
```

### Adaptation and ergodicity

The AM/SCAM covariance and the DE difference vectors are estimated from a
**finite** history buffer: `buffer_size` rows, one retained per `buffer_thin`
iterations, spanning the most recent

```
buffer_size × buffer_thin  =  2000 × 25  =  50,000 iterations
```

at the defaults. Older rows are evicted and stop contributing — the moments are
recomputed from the buffer each time, not accumulated over the whole run.

This is deliberate. A full-history estimator keeps burn-in forever at weight
`1/n`; the finite window throws it away, so the proposal stays matched to the
geometry the chain currently occupies. On a run started far from the mode, the
windowed covariance is accurate while the full-chain covariance is still visibly
inflated by the approach.

The trade-off is theoretical. Once the buffer starts evicting, the kernel keeps
changing by a non-vanishing amount, so the **diminishing-adaptation** condition
of Roberts & Rosenthal (2007) is not satisfied and their ergodicity theorem does
not apply. Below `buffer_size × buffer_thin` iterations the window is still
growing and the condition does hold.

In practice this has not been observed to bias results: on a unit Gaussian and on
an equal-weight bimodal target, both analysed strictly after eviction begins,
adapt-forever matched its `num_adapt`-frozen twin to within Monte Carlo error
(posterior width within 0.1%, mode weights 0.499 vs 0.500). Adapting for the
whole run is the default and is a reasonable choice.

When you need a chain that is exactly Markovian by construction rather than by
that argument — a formal convergence claim, or a target where the window may
"forget" a mode it has not visited in 50,000 iterations — set `num_adapt` and
discard the pre-freeze samples. Raising `buffer_thin` so the window never evicts
also restores the condition, but at the cost of readmitting burn-in and delaying
DE activation, so `num_adapt` is usually the better trade.

## Normalizing-flow proposals

With the `[flow]` extra installed, add a flow-based independence proposal that fits
itself to the cold chain's recent history (or wrap a pre-fitted `coppuccino` flow):

```python
from impulse import NormalizingFlowProposal  # needs the [flow] extra

sampler.add_custom_jump(NormalizingFlowProposal(min_samples=1000), weight=10)
```

`min_samples` counts rows in the history buffer, which keeps one row every
`buffer_thin` iterations (default 25), so the first fit above happens after
about 25,000 iterations, and a `min_samples` larger than `buffer_size` never
fits. Pass `buffer_thin=1` to the sampler for short runs.

## Custom proposals

Any callable with the signature `proposal(chain_stats) -> (new_sample, qxy)` can be
registered on every temperature chain via `add_custom_jump`. The `ChainStats`
argument provides `current_sample`, `rng`, `ndim`, learned covariances, and the
sample-history buffer.

**The `qxy` convention:** `qxy = log q(x|y) - log q(y|x)`, where `x` is the
**current** sample, `y` is the **proposed** sample, and `q(a|b)` is the density of
proposing `a` from `b`. It is **added** to the log-posterior ratio in the
Metropolis-Hastings acceptance, so positive `qxy` favors acceptance, and symmetric
proposals return `qxy = 0.0`.

Checkpoints store **no code**, so proposals are not serialized: to resume, you
re-register the same proposals, in the same order, with the same weights, and the
sampler verifies that against the checkpoint (raising `CheckpointMismatchError`
if they differ). Module-level functions and callable classes are the recommended
style; callable classes must define a `__name__` attribute, which keys the
acceptance-rate reports and is what the resume check matches on.

An asymmetric example — a multiplicative random walk, where `qxy` is the
log-Jacobian of the rescaling:

```python
import numpy as np
from impulse import PTSampler

class ScaleJump:
    """Rescale one random coordinate: y_i = x_i * exp(eps), eps ~ N(0, sigma^2)."""

    __name__ = "scale_jump"  # required for callable-class proposals

    def __init__(self, sigma=0.5):
        self.sigma = sigma

    def __call__(self, chain_stats):
        x = chain_stats.current_sample.copy()
        i = chain_stats.rng.integers(chain_stats.ndim)
        eps = chain_stats.rng.normal(0.0, self.sigma)
        x[i] *= np.exp(eps)
        # q(y|x) = N(eps; 0, s^2) / y_i and q(x|y) = N(-eps; 0, s^2) / x_i, so
        # qxy = log q(x|y) - log q(y|x) = log(y_i / x_i) = eps
        return x, eps

def pos_log_likelihood(x):
    return -0.5 * np.sum(np.log(x) ** 2)  # log-normal target, x > 0

def pos_log_prior(x):
    return 0.0 if np.all((x > 1e-6) & (x < 1e6)) else -np.inf

sampler = PTSampler(ndim=2, lnlike=pos_log_likelihood, lnprior=pos_log_prior,
                    ntemps=4, seed=3, outdir="./chains_custom")
sampler.add_custom_jump(ScaleJump(sigma=0.5), weight=25)
sampler.sample(np.ones(2), num_iterations=5000)
print(sampler.proposal_acceptance_rates()["scale_jump"]["rate"])
```

## Reversible jump: a variable number of components

`RJPTSampler` samples models with a variable number `k` of copies of one
component type (sinusoids, pulses, galactic binaries, ...) plus optional
shared parameters. Births and deaths change `k` and are accepted with Green's
reversible-jump ratio; you supply the density new components are drawn from,
and can add your own moves within and between models. Your functions take
`(x, k)`, where `x` holds the shared parameters and then `max_components`
slots of `n_component` parameters, NaN past slot `k`:

```python
import numpy as np

from impulse import IndependentComponentProposal, RJPTSampler

pulse_t = np.linspace(0.0, 10.0, 200)
pulse_data = np.exp(-0.5 * ((pulse_t - 3.0) / 0.3) ** 2)
pulse_data += 0.1 * np.random.default_rng(0).standard_normal(pulse_t.size)


def pulse_lnlike(x, k):  # up to 4 pulses of (amplitude, centre)
    comps = x.reshape(4, 2)[:k]
    model = sum(a * np.exp(-0.5 * ((pulse_t - c) / 0.3) ** 2) for a, c in comps)
    r = pulse_data - model
    return -0.5 * np.dot(r, r) / 0.1**2


def pulse_lnprior(x, k):  # uniform k; amplitude U(0.6, 2), centre U(0, 10)
    comps = x.reshape(4, 2)[:k]
    inside = np.all((comps >= [0.6, 0.0]) & (comps <= [2.0, 10.0]))
    return -k * np.log(14.0) if inside else -np.inf


pulse_births = IndependentComponentProposal(
    draw=lambda rng: rng.uniform([0.6, 0.0], [2.0, 10.0]),
    logpdf=lambda u: -np.log(14.0) if np.all((u >= [0.6, 0.0]) & (u <= [2.0, 10.0])) else -np.inf,
)
rj = RJPTSampler(n_shared=0, n_component=2, max_components=4,
                 lnlike=pulse_lnlike, lnprior=pulse_lnprior,
                 component_proposal=pulse_births, ntemps=4, seed=1, outdir="./rj_chains", verbose=False)
rj.sample(np.zeros(8), num_iterations=5000, initial_k=0)
print(np.bincount(rj.load_chain()["k"][0, 1000:], minlength=5))  # samples per k
```

See the [reversible-jump guide](https://aarondjohnson.github.io/impulse_mcmc/user_guide/reversible-jump.html)
and the [example notebook](https://github.com/AaronDJohnson/impulse_mcmc/blob/main/examples/rj_sinusoids.ipynb).

## Checkpoint and resume

Checkpoints are written automatically every `save_freq` iterations. Resuming is
**bit-exact**: an interrupted run resumed to `N`
total iterations produces chain files identical to a single uninterrupted
`N`-iteration run (every RNG stream is captured at an iteration boundary, and stale
chain-file rows are truncated on resume). `num_iterations` is a *global* target —
pass the total, not the increment:

```python
# The quick-start run above left a checkpoint in ./chains; continue it to 20000.
sampler = PTSampler(ndim=2, lnlike=log_likelihood, lnprior=log_prior,
                    ntemps=10, outdir="./chains", resume=True)
sampler.sample([0.0, 0.0], num_iterations=20000)
```

Checkpoints use a **no-code-execution format** by default — `sampler_checkpoint.npz`
(arrays) plus a schema-versioned `sampler_checkpoint.json` — so loading one is as
safe as reading a data file. Resume is *reconstruct then restore*: rebuild the
sampler the same way (same constructor and `add_custom_jump` calls), then
`resume=True` verifies the reconstruction against the checkpoint and restores
state into it. See the [checkpointing guide](docs/user_guide/checkpointing.md)
for what is verified, and [SECURITY.md](SECURITY.md) for the trust boundary.

## Migrating from 1.x

- 2.0.0 is a complete rewrite: the old `base.py` / `mhsampler.py` / `ptsampler.py`
  API is removed and replaced by `PTSampler`.
- Chains are bit-different from 1.x at the same seed, chain files are binary by
  default (`chain_format="text"` restores text), and matplotlib is no longer a
  dependency.
- See the full breaking-changes list in [CHANGELOG.md](CHANGELOG.md) under 2.0.0.

## Documentation and links

- **Documentation**: <https://aarondjohnson.github.io/impulse_mcmc/>
- **Examples**: notebooks in [`examples/`](examples/): a sinusoid quickstart,
  vectorized and JAX likelihoods, sampler features (tempering, periodic
  parameters, custom proposals, checkpointing), high-dimensional sampling,
  normalizing-flow proposals, and a reversible-jump search for an unknown number
  of sinusoids
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Security policy**: [SECURITY.md](SECURITY.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
- **Issues**: <https://github.com/AaronDJohnson/impulse_mcmc/issues>

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use impulse-mcmc in your research, please cite:

```bibtex
@software{impulse_mcmc,
  author = {Johnson, Aaron D.},
  title = {impulse-mcmc: A modular parallel tempering MCMC sampler},
  url = {https://github.com/AaronDJohnson/impulse_mcmc},
  version = {2.0.0},
  year = {2026}
}
```
