# Reversible-jump MCMC

impulse-mcmc does trans-dimensional inference through a **product-space
embedding**: instead of literally growing and shrinking the parameter
vector, the sampler works in a fixed-size space holding *all* possible
source slots plus a model index, and the index decides how many slots are
active.

For a space with `num_sources` maximum sources of `num_params` parameters
each, the parameter vector is

```text
[ src0_p0 ... src0_pk | src1_p0 ... src1_pk | ... | nmodel ]
```

with `ndim = num_sources * num_params + 1`. `nmodel = k` means sources
`0..k` are active (`k + 1` active sources).

## BirthDeathProductSpace: the contract

{class}`~impulse.BirthDeathProductSpace` wires your model into this embedding.

```python
import numpy as np
from impulse import PTSampler, BirthDeathProductSpace

rng = np.random.default_rng(0)
t = np.linspace(0.0, 1.0, 100)
sigma = 0.4

# synthetic data: TWO sinusoids
truth = [(1.6, 3.0), (1.0, 7.0)]                # (amplitude, frequency)
y = sum(a * np.sin(2 * np.pi * f * t) for a, f in truth)
y = y + sigma * rng.normal(size=t.size)

MAX_SOURCES = 3
NUM_PARAMS = 2   # per source: amplitude, frequency

def loglike(active):
    # receives ONLY the active sources: shape ((nmodel + 1) * NUM_PARAMS,)
    n_active = len(active) // NUM_PARAMS
    model = np.zeros_like(t)
    for k in range(n_active):
        amp, freq = active[NUM_PARAMS * k : NUM_PARAMS * (k + 1)]
        model = model + amp * np.sin(2 * np.pi * freq * t)
    return -0.5 * np.sum(((y - model) / sigma) ** 2)

def logprior(params):
    # receives ALL source slots (active AND inactive):
    # shape (MAX_SOURCES * NUM_PARAMS,) — bounds must hold on every slot
    amps = params[0::2]
    freqs = params[1::2]
    if np.all((amps >= 0.0) & (amps <= 5.0)) and np.all((freqs >= 1.0) & (freqs <= 10.0)):
        # log of the normalized per-source density, summed over slots
        return -(len(params) // NUM_PARAMS) * (np.log(5.0) + np.log(9.0))
    return -np.inf

def draw_source(rng):
    # one source's parameters, drawn from the per-source prior
    return np.array([rng.uniform(0.0, 5.0), rng.uniform(1.0, 10.0)])

space = BirthDeathProductSpace(
    loglikelihood=loglike,
    logprior=logprior,
    num_sources=MAX_SOURCES,
    num_params=NUM_PARAMS,
    source_prior_draw=draw_source,
)
```

The contract, precisely:

- **`loglikelihood(active)` receives only the active parameters** — the
  first `(nmodel + 1) * num_params` entries.
- **`logprior(params)` receives ALL `num_sources * num_params` source
  parameters**, active and inactive (this is where
  `BirthDeathProductSpace` differs from its base class
  {class}`~impulse.NestedProductSpace`, whose prior sees only active
  slots). Inactive slots stay inside the prior support and contribute the
  correct Occam factor to birth/death moves.
- **`source_prior_draw(rng)`** returns one source's parameters, shape
  `(num_params,)`, drawn from the per-source prior. It refreshes inactive
  slots and proposes newborn sources.
- When you don't pass `source_prior_logpdf`, the birth/death moves fall
  back to using `logprior` on a single source's `num_params`-length vector
  as the per-source density. That only works if `logprior` is **additive
  across slots** (independent per-source priors, as above) — the space
  probes this at proposal construction and raises if the probe fails.
  Supply `source_prior_logpdf` explicitly otherwise, and
  `source_proposal_logpdf` too if `source_prior_draw` samples something
  other than the prior.

## Building the sampler: `from_rjmcmc`

Don't wire proposals by hand — use
{meth}`PTSampler.from_rjmcmc <impulse.PTSampler.from_rjmcmc>` (or
`RJPTSampler.from_rjmcmc` to add NUTS, see {doc}`nuts`):

```python
sampler = PTSampler.from_rjmcmc(
    space,
    ntemps=8,
    seed=42,
    outdir="./chains_rj",
)

x0 = space.draw_initial_position(np.random.default_rng(42), nmodel=0)
sampler.sample(x0, num_iterations=20_000)
```

`from_rjmcmc` registers, on top of the standard continuous moves (AM,
SCAM):

- **One combined `birth_death` kernel** (weight
  `birth_weight + death_weight`). Birth adds the next source slot with a
  fresh prior draw; death removes the last active slot and refreshes it
  from the prior. The kernel internally chooses birth vs. death with the
  space's `prob_schedule` probabilities evaluated at the *current* model
  index, which is exactly what makes the Hastings ratios of the two moves
  correct.

  ```{warning}
  Never register birth and death as two separate constant-weight jumps
  (e.g. via `get_birth_proposal()` / `get_death_proposal()` +
  `add_custom_jump`). Constant-weight selection makes the forward/reverse
  selection probabilities state-independent, violates detailed balance,
  and demonstrably biases the model posterior toward fewer sources.
  Use the combined kernel — `from_rjmcmc` does this for you.
  ```

- **`nmodel_jump`** (weight `nmodel_weight`): a uniform draw of the model
  index, giving direct jumps between any two models.
- **`source_swap_proposal`** (weight `swap_weight`): swaps two source
  blocks, mixing over label permutations.
- **`de`** (weight `de_weight`): the min-fill-gated differential-evolution
  move. With per-model statistics the run's history is split across model
  indices, so a full-buffer gate would never open at realistic run
  lengths; `de` instead activates once the current model's buffer holds
  `de_min_fill` samples (default 100, a `from_rjmcmc` argument). This is
  the move that diffuses along within-model degeneracy ridges; without it
  model posteriors can be metastably wrong. Runs from earlier 2.0-dev
  builds reported this move under the acceptance key `early_de`.

Parameter groups are set to one group per source, **excluding the model
index**, so continuous moves never touch `nmodel`.

## Model posteriors and Bayes factors

The cold chain's last column is the model index; posterior model
probabilities are visit frequencies:

```python
from impulse import model_visitation_stats, bayes_factor_from_chain

chain = sampler.load_chain()
cold = chain["samples"][0]
burn = 5_000

stats = model_visitation_stats(cold, num_models=MAX_SOURCES, burn=burn)
print("P(k+1 sources | data):", stats["posterior_probs"])
print("mean dwell times:     ", stats["mean_dwell_times"])
print("transition matrix:\n", stats["transition_matrix"])

# Bayes factor for 2 sources (nmodel=1) vs 1 source (nmodel=0)
bf_21 = bayes_factor_from_chain(cold, model_i=1, model_j=0, burn=burn)
print("BF(2 sources : 1 source) =", bf_21)
```

How to read these diagnostics:

- `posterior_probs[k]` estimates $P(\text{model } k \mid \text{data})$;
  with a uniform model prior (the default embedding), ratios of these are
  Bayes factors — that is what
  {func}`~impulse.bayes_factor_from_chain` computes. If a model is never
  visited the estimate is 0 (or `inf`/`nan` for the ratio); that is a
  *mixing* statement, not evidence — run longer before trusting it.
- `transition_matrix[i, j]` is the observed frequency of `i -> j` moves.
  Weak off-diagonal mass means the chain rarely changes model; the model
  posterior then has few effective samples even if the run is long.
- `mean_dwell_times[k]` is the average consecutive stay in model `k`.
  Dwell times much longer than the autocorrelation time of the continuous
  parameters mean you should thin your confidence in the model posterior
  accordingly (effective sample size for `nmodel` is roughly
  `nsamples / mean dwell time`).
- Convergence checks: run multiple seeds and compare
  `posterior_probs`; treat disagreement beyond Monte-Carlo error as
  non-convergence of the trans-dimensional part.

A short run like the 20 000 iterations above demonstrates the mechanics;
production model selection typically needs hundreds of thousands of
iterations, and you should confirm stability of `posterior_probs` under
different seeds and longer runs.

## Notes and edge cases

- `space.model_posterior_probs(cold, burn=burn)` is a convenience wrapper
  for just the visit frequencies.
- For a single-model space (`num_sources == 1`), `from_rjmcmc` registers
  only the continuous moves — there is no trans-dimensional move to make.
- Resuming a checkpoint written before the 2.0 detailed-balance fix
  migrates the legacy separate birth/death wiring to the combined kernel
  automatically when possible, with a loud warning; pre-resume samples
  remain biased and should be discarded (see {doc}`checkpointing`).
