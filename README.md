# impulse-mcmc

A modular and efficient implementation of parallel tempering MCMC with adaptive proposals, vectorized likelihood evaluation, and comprehensive diagnostics.

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-github%20pages-blue.svg)](https://aarondjohnson.github.io/impulse_mcmc/)

## Overview

**impulse-mcmc** is a parallel tempering MCMC sampler for production-scale Bayesian
inference. It combines adaptive proposals (AM, SCAM, differential evolution,
normalizing flows), automatic temperature-ladder adaptation, product-space (birth-death) model
selection, and gradient-based NUTS transitions. Every sampler checkpoints and
resumes bit-exactly, in a format that executes no code on load.

Two supported samplers share one interface:

- **`PTSampler`** — parallel tempering with adaptive Metropolis-Hastings proposals
- **`NUTSSampler`** — standalone No-U-Turn Sampler for gradient-based problems

Product-space (birth-death) model selection and the NUTS-interleaved
**`HybridPTSampler`** live in **`impulse.experimental`**. They work and they are
tested — the trans-dimensional kernel is proven exact by finite-state
enumeration — but they are not covered by the 2.x API stability policy, so their
names and signatures may change in a minor release. Import them explicitly:

```python
from impulse.experimental import HybridPTSampler, BirthDeathProductSpace
```

## Installation

Requires Python ≥ 3.10. Core dependencies are just NumPy, SciPy, and tqdm.

```bash
pip install impulse-mcmc              # core
pip install "impulse-mcmc[plots]"     # + matplotlib for the impulse.validation SBC plots
pip install "impulse-mcmc[flow]"      # + coppuccino for normalizing-flow proposals
pip install "impulse-mcmc[dev]"       # + test and lint tooling
```

Or from source:

```bash
git clone https://github.com/AaronDJohnson/impulse_mcmc.git
cd impulse_mcmc
pip install -e ".[plots,flow,dev]"
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
- **Min-fill-gated DE** — the `de` move activates once the sample history buffer
  holds `min_fill` samples (default 100, configurable via the sampler's
  `de_min_fill` argument) instead of waiting for a completely full buffer. This
  keeps the history-based, ridge-following DE move available at realistic run
  lengths — essential in product-space model-selection runs, where per-model buffers never
  fill. Below the threshold `de` returns the current position unchanged (no
  hidden substitution). `EarlyDE` / `make_early_de` are backward-compatibility
  aliases for the same implementation:
  `sampler.add_custom_jump(make_early_de(min_fill), weight)` still works.
- **Temperature-ladder adaptation** — the geometric ladder adapts toward uniform
  swap acceptance between neighbours (`adapt_t0`, `adapt_nu` control the schedule).
- **`inf_temp=True`** — replaces the hottest rung with a T = ∞ chain that samples
  the prior (`ntemps` total chains, `ntemps - 1` finite), improving hot-chain
  mixing and enabling prior-dominated moves to propagate down the ladder.
- **`num_adapt`** — freezes **all** adaptation (proposal covariances, DE buffers,
  ladder adaptation, NUTS step sizes/mass matrices, flow refits) once the global
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

## Product-space model selection

> **Experimental.** Everything in this section lives in `impulse.experimental`
> and is not covered by the 2.x API stability policy.


`BirthDeathProductSpace` embeds up to `num_sources` identical "source" slots (each with
`num_params` parameters) plus a model index in one product space, and
`make_product_space_sampler` wires up the trans-dimensional kernels: one **combined
birth/death kernel** (separate birth and death jumps would violate detailed
balance), a uniform model-index jump, a source-swap (label-switching) move, and
the min-fill-gated `de` move (`de_min_fill`).

The prior contract differs from the likelihood contract:

- `loglikelihood(active_params)` receives only the **active** sources' parameters,
  shape `((nmodel + 1) * num_params,)`.
- `logprior(all_params)` receives **ALL** source slots — active and inactive —
  shape `(num_sources * num_params,)`. It must bound-check every slot, and (unless
  you pass `source_prior_logpdf` explicitly) it must be additive across slots,
  i.e. a product of independent per-source priors.

```python
import numpy as np
from impulse import bayes_factor_from_chain, model_visitation_stats
from impulse.experimental import (
    BirthDeathProductSpace, make_product_space_sampler)

# Data generated by ONE source of amplitude ~1
rng = np.random.default_rng(0)
data = 1.0 + 0.1 * rng.standard_normal(50)

MAX_SOURCES, NUM_PARAMS = 3, 1  # up to 3 sources, 1 parameter (amplitude) each

def rj_log_likelihood(active_params):
    # only the ACTIVE sources' parameters: shape ((nmodel + 1) * NUM_PARAMS,)
    return -0.5 * np.sum((data - active_params.sum()) ** 2) / 0.1**2

def rj_log_prior(all_params):
    # ALL source slots, active and inactive: shape (MAX_SOURCES * NUM_PARAMS,)
    if np.all((all_params >= 0.0) & (all_params <= 2.0)):
        return -all_params.size * np.log(2.0)  # independent Uniform(0, 2) per slot
    return -np.inf

def draw_source(rng):
    return rng.uniform(0.0, 2.0, size=NUM_PARAMS)  # one source's params from the prior

space = BirthDeathProductSpace(
    loglikelihood=rj_log_likelihood,
    logprior=rj_log_prior,
    num_sources=MAX_SOURCES,
    num_params=NUM_PARAMS,
    source_prior_draw=draw_source,
)

sampler = make_product_space_sampler(space, ntemps=8, seed=42, outdir="./chains_rj")
x0 = space.draw_initial_position(np.random.default_rng(42))
sampler.sample(x0, num_iterations=5000)

cold = sampler.load_chain()["samples"][0]      # model index is the last column
stats = model_visitation_stats(cold, num_models=MAX_SOURCES, burn=1000)
print("P(k+1 sources):", stats["posterior_probs"])
print("B(1 source vs 2):", bayes_factor_from_chain(cold, model_i=0, model_j=1, burn=1000))
```

`model_visitation_stats` also returns visit counts, the model transition matrix,
and mean dwell times — useful for judging how well the chain mixes across models.

## NUTS and HybridPTSampler

> `NUTSSampler` is supported. `HybridPTSampler` is **experimental**
> (`impulse.experimental`) and not covered by the 2.x API stability policy.


`NUTSSampler` is a standalone No-U-Turn Sampler driven by a single
`logp_and_grad(x) -> (logp, grad)` callable; `compose_logp_and_grad` builds one
from separate likelihood/prior (falling back to numerical gradients for any piece
you don't supply):

```python
import numpy as np
from impulse import NUTSSampler, compose_logp_and_grad

def nuts_lnlike(x):
    return -0.5 * np.sum(x**2)

def nuts_lnlike_grad(x):
    return -x

def nuts_lnprior(x):
    return 0.0

logp_and_grad = compose_logp_and_grad(nuts_lnlike, nuts_lnprior, nuts_lnlike_grad)
nuts = NUTSSampler(ndim=2, logp_and_grad=logp_and_grad, seed=42, outdir="./chains_nuts")
nuts.sample(np.zeros(2), num_iterations=1000)
```

`HybridPTSampler` interleaves MH proposals (including birth/death model moves), NUTS
transitions on the active continuous parameters, and PT swaps. Pass `lnlike_grad`
with signature `(active_params) -> (loglike, gradient)`; continuing the product-space model-selection
example above:

```python
from impulse.experimental import HybridPTSampler

def rj_log_likelihood_grad(active_params):
    resid = data - active_params.sum()
    return -0.5 * np.sum(resid**2) / 0.1**2, np.full(active_params.size, resid.sum() / 0.1**2)

hybrid = HybridPTSampler.from_product_space(
    space, lnlike_grad=rj_log_likelihood_grad, ntemps=4, seed=7, outdir="./chains_hybrid"
)
hybrid.sample(x0, num_iterations=1000)
```

Without `lnlike_grad`, `HybridPTSampler` skips NUTS and behaves like `PTSampler`.

## Normalizing-flow proposals

With the `[flow]` extra installed, add a flow-based independence proposal that fits
itself to the cold chain's recent history (or wrap a pre-fitted `coppuccino` flow):

```python
from impulse.flow_proposals import NormalizingFlowProposal  # pip install "impulse-mcmc[flow]"

sampler.add_custom_jump(NormalizingFlowProposal(min_samples=1000), weight=10)
```

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

## Checkpoint and resume

Checkpoints are written automatically every `save_freq` iterations. Resuming is
**bit-exact** for checkpoints written by 2.0: an interrupted run resumed to `N`
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
sampler the same way (same constructor / product-space / `add_custom_jump` calls),
then `resume=True` restores state into it. Legacy `sampler_checkpoint.pkl`
checkpoints still load (with a security/deprecation warning), but unpickling one can
execute arbitrary code — see [SECURITY.md](SECURITY.md) and the
[checkpointing guide](docs/user_guide/checkpointing.md) for the full trust boundary.

`NUTSSampler` follows the same contract, skipping warmup on resume and restoring
the adapted step size and mass matrix rather than re-adapting them.

## Migrating from 1.x

- 2.0.0 is a complete rewrite: the old `base.py` / `mhsampler.py` / `ptsampler.py`
  API is removed and replaced by `PTSampler`, `HybridPTSampler`, and `NUTSSampler`.
- Chains are bit-different from 1.x at the same seed; matplotlib moved behind the
  `[plots]` extra; `MassMatrix.from_covariance` now inverts its argument (Stan
  convention) — use `MassMatrix.from_precision` for Fisher matrices.
- See the full breaking-changes list in [CHANGELOG.md](CHANGELOG.md) under 2.0.0.

## Documentation and links

- **Documentation**: <https://aarondjohnson.github.io/impulse_mcmc/>
- **Examples**: complete notebooks in [`examples/`](examples/) (sinusoid fitting,
  high-dimensional sampling, product-space model selection, hybrid model selection + NUTS)
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
