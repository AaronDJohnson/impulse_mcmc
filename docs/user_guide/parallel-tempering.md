# Parallel tempering

{class}`~impulse.PTSampler` runs `ntemps` coupled chains, each targeting a
tempered posterior

$$
\pi_T(x) \propto p(x)\, L(x)^{1/T},
$$

where the prior is *not* tempered. Hot chains ($T \gg 1$) move freely across
the posterior landscape; periodic swap moves let the cold chain ($T = 1$)
inherit their exploration. Only the cold chain samples your posterior — hot
chains are machinery for mixing.

## The temperature ladder

By default a geometric ladder is built from `min_temp=1.0` with spacing
$1 + \sqrt{2/\mathrm{ndim}}$, chosen to target roughly 25% swap acceptance
on Gaussian problems:

```python
import numpy as np
from impulse import PTSampler

def log_likelihood(x):
    return -0.5 * np.sum(x**2)

def log_prior(x):
    if np.all(np.abs(x) <= 5):
        return 0.0
    return -np.inf

# default ladder: geometric from min_temp with automatic spacing
sampler = PTSampler(ndim=2, lnlike=log_likelihood, lnprior=log_prior,
                    ntemps=8, seed=1, outdir="./chains_pt")
print(sampler.ptstate.ladder)
```

You can control the ladder in several ways:

```python
# geometric spacing solved to end at max_temp
sampler = PTSampler(ndim=2, lnlike=log_likelihood, lnprior=log_prior,
                    ntemps=8, max_temp=100.0, seed=1, outdir="./chains_pt")

# fully custom ladder (overrides everything else)
sampler = PTSampler(ndim=2, lnlike=log_likelihood, lnprior=log_prior,
                    ntemps=4, ladder=np.array([1.0, 3.0, 10.0, 30.0]),
                    seed=1, outdir="./chains_pt")
```

### The infinite-temperature chain

`inf_temp=True` replaces the top of the ladder with $T = \infty$. That
chain samples the *prior* (the likelihood contributes nothing at infinite
temperature), which guarantees globally distributed states keep feeding
down the ladder — useful for strongly multimodal posteriors, and required
if you want prior samples for evidence-style diagnostics:

```python
sampler = PTSampler(ndim=2, lnlike=log_likelihood, lnprior=log_prior,
                    ntemps=8, inf_temp=True, seed=1, outdir="./chains_pt")
print(sampler.ptstate.ladder[-1])   # inf
```

### Temperature adaptation

Between-swap acceptance rates are equalized during the run using the
adaptive scheme of Vousden et al. (2016, arXiv:1501.05823). The coldest and
hottest temperatures stay fixed; interior temperatures move with a
hyperbolic decay controlled by `adapt_t0` (default 100) and `adapt_nu`
(default 10). Swap attempts happen every `swap_steps` iterations
(default 1).

Check swap health after (or during) a run:

```python
sampler.sample(np.zeros(2), num_iterations=10_000)
report = sampler.chain_acceptance_rates()
print(report["temperatures"])   # final ladder
print(report["pt_swap"])        # per-neighbor-pair swap acceptance rates
```

Healthy runs have swap rates that are roughly uniform across pairs and not
near 0 (chains decoupled) or near 1 (ladder wastefully dense). The same
report is written to `<outdir>/chain_acceptance.json` at every save.

## Adaptive proposals

Each chain draws its Metropolis-Hastings move from a weighted mixture:

| Proposal | Default weight | What it does |
|----------|----------------|--------------|
| `am`     | `am_weight=15` | Adaptive Metropolis: correlated Gaussian jump from the chain's empirical covariance. |
| `scam`   | `scam_weight=30` | Single-Component AM: jump along one eigendirection of the covariance. |
| `de`     | `de_weight=50` | Differential evolution: difference of two random history samples; excellent along ridges and between modes. |

All three adapt: AM/SCAM from the recursively updated sample covariance
(refreshed every `cov_update` iterations), DE from a rolling history buffer
(`buffer_size`, default 50 000). `de` is min-fill-gated: the difference move
activates once the history buffer holds `min_fill` samples (default 100;
configurable via the sampler's `de_min_fill` argument), and before that the
proposal returns the current position unchanged. `EarlyDE` /
`make_early_de` are backward-compatibility aliases for the same
implementation (see {doc}`model-selection`).

Weights are relative, not percentages:

```python
sampler = PTSampler(ndim=2, lnlike=log_likelihood, lnprior=log_prior,
                    ntemps=8, am_weight=20, scam_weight=20, de_weight=60,
                    seed=1, outdir="./chains_pt")
```

Per-proposal acceptance statistics are available at any time:

```python
sampler.sample(np.zeros(2), num_iterations=10_000)
for name, stats in sampler.proposal_acceptance_rates().items():
    print(f"{name:12s} rate={stats['rate']:.3f} calls={stats['calls']}")
```

## Freezing adaptation: `num_adapt`

Adaptive proposals and ladder adaptation make the transition kernel
history-dependent. `num_adapt` freezes *everything* — covariance updates, the DE
buffer, ladder adaptation, and refits of adaptive custom proposals — after a
fixed number of iterations, leaving a transition kernel that is exactly
Markovian from that point on:

```python
sampler = PTSampler(ndim=2, lnlike=log_likelihood, lnprior=log_prior,
                    ntemps=8, num_adapt=5_000, seed=1, outdir="./chains_pt")
sampler.sample(np.zeros(2), num_iterations=20_000)

chain = sampler.load_chain()
post = chain["samples"][0][5_000:]   # discard the adaptive warmup
```

Recommended usage:

- Treat everything before `num_adapt` as warmup and discard it.
- A reasonable default is the first 25–50% of the planned run, once
  acceptance rates and the ladder have visibly stabilized.
- `num_adapt=None` (the default) adapts for the whole run — the historical
  behavior, and fine in practice; see the note below for what it does and does
  not guarantee.
- `num_adapt` counts *global* iterations and persists across
  checkpoint/resume; see {doc}`checkpointing` for the resume semantics.

### What adapting for the whole run does and does not guarantee

The AM/SCAM covariance and the DE difference vectors come from a **finite**
history buffer: `buffer_size` rows, one retained per `buffer_thin` iterations,
spanning the most recent `buffer_size * buffer_thin` iterations — 50,000 at the
defaults. Older rows are evicted and stop contributing;
`ChainStats.recursive_update` recomputes the moments from the buffer rather than
accumulating over all history.

That is intentional. A full-history estimator carries burn-in forever at weight
`1/n`, while the finite window discards it and keeps the proposal matched to the
geometry the chain currently occupies.

The consequence is that once the buffer begins evicting, the kernel keeps
changing by a non-vanishing amount, so the **diminishing-adaptation** condition
of Roberts & Rosenthal (2007) is not met and their ergodicity theorem does not
apply. Below `buffer_size * buffer_thin` iterations the window is still growing
and the condition does hold.

Empirically this has not been observed to bias results. On a unit Gaussian and
on an equal-weight bimodal target, analysed strictly after eviction begins,
adapt-forever matched its `num_adapt`-frozen twin to within Monte Carlo error
(posterior width within 0.1%; mode weights 0.499 against a true 0.500).

Use `num_adapt` when you want a chain that is exactly Markovian by construction
rather than by that argument — a formal convergence claim, or a target where the
window could plausibly "forget" a mode it has not visited within its span.
Raising `buffer_thin` until the window never evicts also restores the condition,
but readmits burn-in and delays DE activation, so freezing is usually the better
trade.

## Periodic parameters

Circular parameters (phases, angles) would otherwise random-walk to
$\pm\infty$ and inflate the adaptive covariance. Declare them with
`periodic`, mapping parameter index to a period or `(low, high)` interval:

```python
sampler = PTSampler(ndim=3, lnlike=lambda x: -0.5 * np.sum(x[:2]**2),
                    lnprior=lambda x: 0.0 if np.all(np.abs(x) <= 7) else -np.inf,
                    periodic={2: (-np.pi, np.pi)},   # dim 2 wraps into [-pi, pi)
                    ntemps=4, seed=1, outdir="./chains_periodic")
sampler.sample(np.zeros(3), num_iterations=2_000)
```

Stored positions are wrapped into the stated interval; your likelihood and
prior must be invariant under shifts by the period.
