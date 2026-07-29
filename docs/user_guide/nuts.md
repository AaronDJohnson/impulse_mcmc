# NUTS (gradient-based sampling)

Two entry points use gradients:

- {class}`~impulse.NUTSSampler` — a standalone No-U-Turn Sampler with
  Stan-style warmup, for fixed-dimension problems.
- {class}`~impulse.experimental.HybridPTSampler` — parallel tempering that *interleaves*
  NUTS transitions with the Metropolis-Hastings and birth/death model moves,
  when you pass `lnlike_grad`.

## Standalone: NUTSSampler

`NUTSSampler` takes a single callable `logp_and_grad(x) -> (logp, grad)`.
Build it from separate likelihood/prior pieces with
{func}`~impulse.compose_logp_and_grad`, which falls back to numerical
differentiation for any gradient you don't supply:

```python
import numpy as np
from impulse import NUTSSampler, compose_logp_and_grad

def lnlike(x):
    return -0.5 * np.sum(x**2)

def lnlike_grad(x):
    return -x

def lnprior(x):
    if np.all(np.abs(x) < 10):
        return 0.0
    return -np.inf

def lnprior_grad(x):
    return np.zeros_like(x)   # flat inside the support

logp_and_grad = compose_logp_and_grad(
    lnlike, lnprior, lnlike_grad=lnlike_grad, lnprior_grad=lnprior_grad
)

sampler = NUTSSampler(
    ndim=2,
    logp_and_grad=logp_and_grad,
    num_warmup=500,               # Stan-style three-phase warmup
    mass_matrix_type="diagonal",  # 'unit', 'diagonal', or 'dense'
    target_accept=0.8,
    seed=42,
    outdir="./chains_nuts",
)
sampler.sample(np.zeros(2), num_iterations=2_000)

chain = sampler.load_chain()
print(chain["samples"].shape)          # (2000, 2)
print(sampler.get_diagnostics())       # divergences, tree depth, step size, ...
```

If you have no gradient at all,
{func}`~impulse.make_logp_and_grad_numerical` wraps a scalar `logp` with
central differences — fine for smoke tests, slow (`2 * ndim` evaluations
per gradient) for real problems.

Watch `get_diagnostics()`:

- `num_divergent > 0` signals step sizes too large for the local
  curvature — raise `target_accept` (e.g. 0.9–0.99) or reparameterize.
- `num_max_depth` counting up means trajectories are being truncated —
  usually an under-informative mass matrix or unscaled parameters.

## Mass-matrix conventions

impulse follows **Stan's convention**: momenta are drawn
$p \sim N(0, M)$ and velocities are $M^{-1} p$, so the *inverse* metric
should equal the posterior covariance — i.e. the stored mass matrix is

$$
M = \Sigma^{-1}.
$$

The two constructors on {class}`~impulse.MassMatrix` differ in what they
invert, and mixing them up silently installs the inverse of the metric you
want:

```python
import numpy as np
from impulse import MassMatrix, MassMatrixType

posterior_cov = np.array([[1.0, 0.9],
                          [0.9, 1.0]])

# from_covariance INVERTS its argument: M = cov^{-1}
m1 = MassMatrix.from_covariance(posterior_cov, MassMatrixType.DENSE)

# from_precision stores its argument as-is: M = precision
fisher = np.linalg.inv(posterior_cov)   # Fisher approximates the precision
m2 = MassMatrix.from_precision(fisher, MassMatrixType.DENSE)

# both build the same metric
p = np.array([0.3, -0.2])
assert np.allclose(m1.kinetic_energy(p), m2.kinetic_energy(p))
```

Rule of thumb: **covariance estimates go through `from_covariance`;
Fisher/precision matrices go through `from_precision`.** A Fisher
information matrix approximates the posterior *precision*, which under
this convention is exactly the mass matrix — no inversion wanted.

```{versionchanged} 2.0.0
`from_covariance` now inverts its argument (it previously stored the
covariance as `M` directly). Code that passed a Fisher matrix to
`from_covariance` must switch to `from_precision`.
```

## Hybrid: HybridPTSampler with `lnlike_grad`

{class}`~impulse.experimental.HybridPTSampler` is a peer of `PTSampler` that adds a NUTS
transition after each MH step when `lnlike_grad` is provided. The gradient
callable has a different signature from the standalone sampler's — it
receives the **active continuous parameters** (no model index) and returns
both the value and the gradient:

```text
lnlike_grad(active_params) -> (loglike_value, gradient_array)
```

A fixed-dimension example (no model moves — `HybridPTSampler` works fine
as a plain PT+NUTS sampler):

```python
import numpy as np
from impulse.experimental import HybridPTSampler

def lnlike(x):
    return -0.5 * np.sum(x**2)

def lnprior(x):
    if np.all(np.abs(x) <= 10):
        return 0.0
    return -np.inf

def lnlike_grad(x):
    return -0.5 * np.sum(x**2), -x    # (value, gradient)

sampler = HybridPTSampler(
    ndim=2,
    lnlike=lnlike,
    lnprior=lnprior,
    lnlike_grad=lnlike_grad,
    ntemps=4,
    max_tree_depth=8,        # cold chain
    hot_chain_max_depth=4,   # cheaper trajectories for hot chains
    seed=42,
    outdir="./chains_rjpt",
)
sampler.sample(np.zeros(2), num_iterations=3_000)
print(sampler.get_diagnostics()["mean_accept_prob"])
```

Notes:

- The prior gradient is *not* requested: NUTS runs on the tempered
  likelihood plus prior, with the prior handled through its value (flat
  priors contribute zero gradient inside the support).
- Step sizes are adapted **per (chain, active dimension)** and mass
  matrices **per active dimension** — in product-space runs each model
  dimensionality gets its own adapted step size and mass matrix.
- Combine with a product space via `HybridPTSampler.from_product_space(space,
  lnlike_grad=...)`; the gradient then receives the
  `(nmodel + 1) * num_params` active source parameters, matching the
  space's `loglikelihood` contract (see {doc}`model-selection`).

## Injecting a Fisher-based mass matrix

If you can compute a Fisher matrix at (or near) the maximum a posteriori
point, inject it so NUTS starts with a good metric instead of adapting one
from scratch:

```python
from impulse import MassMatrix, MassMatrixType

n_active = 2   # number of active continuous parameters this matrix is for
fisher = np.array([[4.0, 0.0],
                   [0.0, 1.0]])

sampler = HybridPTSampler(ndim=2, lnlike=lnlike, lnprior=lnprior,
                      lnlike_grad=lnlike_grad, ntemps=4, seed=42,
                      outdir="./chains_rjpt_fisher")
sampler.set_mass_matrix(n_active, MassMatrix.from_precision(fisher, MassMatrixType.DENSE))
sampler.sample(np.zeros(2), num_iterations=2_000)
```

An injected mass matrix is preserved for the whole run — online mass-matrix
adaptation never overwrites it (only the dual-averaging step size continues to
re-tune against it). Use `from_precision` here — **never** `from_covariance`,
which would install the inverse of the intended metric.
