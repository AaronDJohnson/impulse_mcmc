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

## Checkpoint rules: re-register, don't pickle

Checkpoints do **not** serialize your proposals. The default `.npz` + `.json`
format stores no code at all (see {doc}`checkpointing`); resume works by
*reconstruct-then-restore*, so the contract is:

- **Register the same proposals, in the same order, with the same weights**
  before resuming. On resume the sampler compares the reconstructed proposal
  list against the names and weights recorded in the checkpoint and raises
  `CheckpointMismatchError` if they differ — naming the offending proposal
  rather than silently sampling from a different kernel.
- Give the class a `__name__` class attribute. It labels the proposal in
  acceptance-rate reports *and* is what the resume check matches on.
- **Callable classes at module level are still the recommended style**, since
  they give you a stable `__name__` and somewhere to hang state. A closure or
  lambda is not a checkpointing error — it will sample and checkpoint fine —
  but you must be able to re-create an equivalent object at resume time, and
  a lambda's `__name__` is `"<lambda>"`, which makes the mismatch message far
  less useful.

```{note}
Versions before 2.0.0 pickled the whole sampler, so proposals had to be
picklable and lambdas genuinely broke the first checkpoint write. That
restriction is gone with the `.npz` + `.json` format. Pickle checkpoints are
still readable but deprecated; see {doc}`checkpointing`.
```

## Adaptive custom proposals

A proposal may adapt internal state between calls (the built-in
normalizing-flow proposal refits itself from the history buffer, for
example). Two optional hooks matter:

**Persisting adapted state across resume.** Because nothing is pickled, a
proposal's internal state is only saved if it opts in by implementing
`get_checkpoint_state()` / `set_checkpoint_state(state)`:

```python
class AdaptiveJump:
    __name__ = "adaptive_jump"

    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, chain_stats):
        q = chain_stats.current_sample.copy()
        q += chain_stats.rng.standard_normal(chain_stats.ndim) * self.sigma
        return q, 0.0

    # --- optional: survive checkpoint/resume ---
    def get_checkpoint_state(self):
        return {"sigma": float(self.sigma)}          # JSON-friendly values

    def set_checkpoint_state(self, state):
        self.sigma = float(state["sigma"])
```

The sampler captures this from each stateful proposal when it checkpoints and
restores it on resume. Without these hooks a resumed run silently restarts your
proposal from its constructor defaults, discarding whatever it had learned.

**Freezing adaptation.** If the proposal adapts, also expose
`freeze_adaptation()`: when the sampler reaches `num_adapt` (see
{doc}`parallel-tempering`) it calls `freeze_adaptation()` on every proposal
that has one, so the transition kernel becomes exactly Markovian after the
freeze. The freeze should be idempotent, and the frozen flag should be part of
the state returned by `get_checkpoint_state()` — a frozen proposal must stay
frozen across checkpoint/resume.

## Trans-dimensional proposals

Custom moves that change the model index of a product-space model-selection run must
supply exact Hastings terms, including the probability of *selecting* the
forward and reverse moves. This is why birth and death ship as one
combined kernel ({meth}`~impulse.BirthDeathProductSpace.get_birth_death_proposal`)
rather than two registrable pieces — read {doc}`model-selection` before writing
your own.
