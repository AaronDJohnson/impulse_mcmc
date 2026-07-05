# Custom proposals

Every Metropolis-Hastings move in impulse-mcmc — built-in or custom — is a
callable with one contract:

```text
proposal(chain_stats: ChainStats) -> (new_sample: np.ndarray, qxy: float)
```

- `chain_stats.current_sample` is the chain's current position (shape
  `(ndim,)`); return a *new* array, don't mutate it in place.
- `chain_stats.rng` is the chain's `numpy.random.Generator` — **always use
  it** (never `np.random` or your own generator), or you break
  reproducibility and checkpoint bit-exactness.
- `chain_stats` also exposes the adaptive state you may want: `groups`
  (parameter index groups), `proposal_L` (per-group Cholesky-like factors
  of the empirical covariance), and the sample history buffer used by the
  DE moves.
- `qxy` is the log proposal-density ratio described next.

Register the proposal on every temperature chain with a relative weight
using `sampler.add_custom_jump(proposal, weight)`, as in the worked
example below.

## The `qxy` convention

With `x` the **current** state and `y` the **proposed** state,

```text
qxy = log[ q(x | y) / q(y | x) ]
```

i.e. *reverse density over forward density*. The sampler **adds** `qxy` to
the tempered log-posterior ratio before the accept test:

```text
log alpha = [ log pi_T(y) - log pi_T(x) ] + qxy
```

so positive `qxy` favors acceptance. Symmetric proposals
(`q(y|x) = q(x|y)`, e.g. Gaussian random walks) return `qxy = 0.0`.

### A worked asymmetric example

A multiplicative random walk on a positive parameter: propose
`y_i = x_i * exp(eps)` with `eps ~ N(0, sigma^2)`. In log space the move is
symmetric, but the density in `x`-space carries a Jacobian `1/y_i`
(forward) versus `1/x_i` (reverse):

```text
q(y | x) = N(log y_i; log x_i, sigma^2) * (1 / y_i)
q(x | y) = N(log x_i; log y_i, sigma^2) * (1 / x_i)

qxy = log[q(x|y) / q(y|x)] = log(y_i) - log(x_i) = eps
```

Forgetting the Jacobian (returning `qxy = 0`) would bias the chain toward
small values. As a picklable callable class:

```python
import numpy as np

class LogScaleJump:
    """Multiplicative random walk on one strictly positive parameter."""

    __name__ = "log_scale_jump"   # used in acceptance-rate reports

    def __init__(self, index: int, sigma: float = 0.5):
        self.index = index
        self.sigma = sigma

    def __call__(self, chain_stats):
        x = chain_stats.current_sample.copy()
        eps = chain_stats.rng.normal(0.0, self.sigma)
        x[self.index] *= np.exp(eps)
        qxy = eps        # log[q(x|y)/q(y|x)] = log(y_i) - log(x_i)
        return x, qxy
```

And in use — parameter 0 has an Exponential(1) posterior (analytic mean
and standard deviation both 1), parameter 1 is a standard Gaussian:

```python
from impulse import PTSampler

def log_likelihood(x):
    return -x[0] - 0.5 * x[1] ** 2      # Exp(1) x N(0, 1), up to a constant

def log_prior(x):
    if 0.0 < x[0] < 50.0 and np.abs(x[1]) < 10.0:
        return 0.0
    return -np.inf

sampler = PTSampler(ndim=2, lnlike=log_likelihood, lnprior=log_prior,
                    ntemps=2, seed=3, outdir="./chains_custom")
sampler.add_custom_jump(LogScaleJump(index=0, sigma=0.8), weight=50)
sampler.sample(np.array([1.0, 0.0]), num_iterations=20_000)

cold = sampler.load_chain()["samples"][0][4_000:, 0]
print(cold.mean(), cold.std())          # both ~ 1.0
print(sampler.proposal_acceptance_rates()["log_scale_jump"]["rate"])
```

## Picklability rules

Checkpoints pickle the whole sampler, **including every registered
proposal** (see {doc}`checkpointing`). Therefore:

- **Use callable classes defined at module level**, as above — instances
  pickle their attributes and reimport the class by name.
- **Don't use closures or lambdas** as proposals: they can't be pickled,
  so the first checkpoint write (every `save_freq` iterations) fails.
- Give the class a `__name__` class attribute. It labels the proposal in
  acceptance-rate reports.
- Anything the instance holds must itself be picklable (arrays, floats,
  module-level functions are fine; open files and RNGs of your own are
  not — use `chain_stats.rng`).

## Adaptive custom proposals

A proposal may adapt internal state between calls (the built-in
normalizing-flow proposal refits itself from the history buffer, for
example). If it does, expose a `freeze_adaptation()` method: when the
sampler reaches `num_adapt` (see {doc}`parallel-tempering`), it calls
`freeze_adaptation()` on every proposal that has one, so the transition
kernel becomes exactly Markovian after the freeze. The freeze should be
idempotent and its state picklable — a frozen proposal must stay frozen
across checkpoint/resume.

## Trans-dimensional proposals

Custom moves that change the model index of a product-space model-selection run must
supply exact Hastings terms, including the probability of *selecting* the
forward and reverse moves. This is why birth and death ship as one
combined kernel ({meth}`~impulse.BirthDeathProductSpace.get_birth_death_proposal`)
rather than two registrable pieces — read {doc}`model-selection` before writing
your own.
