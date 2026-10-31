# Reversible jump: a variable number of components

`RJPTSampler` samples models made of a variable number `k` of copies of one
component type (sinusoids, pulses, galactic binaries, glitches, spline knots),
plus optional *shared* parameters that every model has, such as a noise level.
The state is `(k, θ_k)`, where `θ_k` holds the shared parameters and `k`
components of `p` parameters each. Moves that add or remove a component change
the number of parameters and are accepted with Green's reversible-jump ratio.

`RJPTSampler` is a `PTSampler`: tempering, the adaptive proposals, vectorized,
threaded and JAX likelihoods, and checkpoint/resume all work as described in
the rest of this guide. This page covers what is different.

## How the state is stored

Every stored row has the same width. With `K = max_components`,
`p = n_component` and `D = n_shared + K·p`:

| Columns | Content |
|---|---|
| `0` to `n_shared − 1` | shared parameters |
| `n_shared + i·p` to `n_shared + (i + 1)·p − 1` | component slot `i`, for `i = 0, …, K − 1` |
| `D` | `k` |

Slots `k` to `K − 1` hold NaN. Your likelihood and prior receive the `D`
parameter columns as `x` and the model index separately as `k`. The padding is
storage only: inactive slots are never sampled and carry no density.

## Quickstart: how many pulses?

Data with two Gaussian pulses in white noise, fitted with up to five pulses of
two parameters each (amplitude, centre). The amplitude prior starts at 0.5, so
a pulse too faint to detect is not an option. There are no shared parameters.

```python
import numpy as np

from impulse import IndependentComponentProposal, RJPTSampler

K, P = 5, 2  # up to 5 pulses, each (amplitude, centre)
WIDTH, SIGMA = 0.3, 0.1
LOW, HIGH = np.array([0.5, 0.0]), np.array([2.0, 10.0])
LOG_VOLUME = np.log(np.prod(HIGH - LOW))

rng = np.random.default_rng(0)
t = np.linspace(0.0, 10.0, 200)


def pulses(comps):
    return sum(a * np.exp(-0.5 * ((t - c) / WIDTH) ** 2) for a, c in comps)


data = pulses([(1.0, 3.0), (0.7, 6.5)]) + SIGMA * rng.standard_normal(t.size)


def lnlike(x, k):
    comps = x.reshape(K, P)[:k]  # x has shape (K * P,); slots past k are NaN
    r = data - pulses(comps)
    return -0.5 * np.dot(r, r) / SIGMA**2


def lnprior(x, k):
    comps = x.reshape(K, P)[:k]
    if np.any((comps < LOW) | (comps > HIGH)):
        return -np.inf
    return -k * LOG_VOLUME  # uniform p(k) is a constant; no k! term


births = IndependentComponentProposal(
    draw=lambda rng: rng.uniform(LOW, HIGH),
    logpdf=lambda u: -LOG_VOLUME if np.all((u >= LOW) & (u <= HIGH)) else -np.inf,
)

sampler = RJPTSampler(
    n_shared=0,
    n_component=P,
    max_components=K,
    lnlike=lnlike,
    lnprior=lnprior,
    component_proposal=births,
    ntemps=4,
    seed=1,
    outdir="rj_chains",
    verbose=False,
)
sampler.sample(np.zeros(K * P), num_iterations=5000, initial_k=0)

chain = sampler.load_chain()
ks = chain["k"][0, 1000:]  # cold chain after burn-in
print("p(k):", (np.bincount(ks, minlength=K + 1) / ks.size).round(3))
```

`births` is the *component proposal*: the density that birth moves draw new
components from. Here it is the component prior.

### The prior is over numbered slots

`lnprior(x, k)` is the joint density of `k` and the parameters in slots `0` to
`k − 1`. For interchangeable components it is

$$\log p(k) + \sum_{i<k} \log \pi(\theta_i) \;\; (+ \text{the shared prior}),$$

with **no** `k!` term: every ordering of the same set of components is a
separate state of the chain, and all of them are sampled. If you impose an
ordering instead (say, increasing centre), the prior gains `+ log k!` and a
birth is valid at only one insert position; relabelling after sampling, shown
below, is simpler.

## Vectorized likelihoods

With `vectorized=True`, `x` is `(batch, D)` and `k` is `(batch,)`. Mask the
inactive slots with `np.where` before using them: their values are NaN, and
`0 * NaN` is NaN.

```python
def lnlike_vectorized(x, k):
    comps = x.reshape(-1, K, P)  # (batch, K, P)
    active = np.arange(K)[None, :] < k[:, None]  # (batch, K)
    amplitude = np.where(active, comps[..., 0], 0.0)  # inactive pulses get amplitude 0
    centre = np.where(active, comps[..., 1], 0.0)
    shapes = np.exp(-0.5 * ((t - centre[..., None]) / WIDTH) ** 2)  # (batch, K, n_t)
    r = data - np.sum(amplitude[..., None] * shapes, axis=1)
    return -0.5 * np.sum(r * r, axis=1) / SIGMA**2


def lnprior_vectorized(x, k):
    comps = x.reshape(-1, K, P)
    active = np.arange(K)[None, :] < k[:, None]
    inside = np.where(active[..., None], (comps >= LOW) & (comps <= HIGH), True).all(axis=(1, 2))
    return np.where(inside, -k * LOG_VOLUME, -np.inf)


check_k = np.array([0, 1, 2, 5])
check_x = np.full((4, K * P), np.nan)
for row, kk in zip(check_x, check_k):
    row[: kk * P] = rng.uniform(LOW, HIGH, (kk, P)).ravel()
assert np.allclose(
    [lnlike(x, kk) for x, kk in zip(check_x, check_k)], lnlike_vectorized(check_x, check_k)
)
assert np.allclose(
    [lnprior(x, kk) for x, kk in zip(check_x, check_k)], lnprior_vectorized(check_x, check_k)
)

vec_sampler = RJPTSampler(
    n_shared=0,
    n_component=P,
    max_components=K,
    lnlike=lnlike_vectorized,
    lnprior=lnprior_vectorized,
    vectorized=True,
    component_proposal=births,
    ntemps=4,
    seed=1,
    outdir="rj_chains_vectorized",
    verbose=False,
)
vec_sampler.sample(np.zeros(K * P), num_iterations=1000, initial_k=0)
```

## Component proposals

Birth/death (reported as `birth` and `death`; weight `birth_death_weight`,
default 40) and the optional `component_redraw` move draw new components from
the `component_proposal`: an object with

- `draw(rng, x, k) -> u`: one component, shape `(p,)`;
- `logpdf(u, x, k) -> float`: the **normalized** log density of exactly what
  `draw` returns.

`(x, k)` is always the state of the *smaller* model: the current state for a
birth, and the state after removing the component for a death. A proposal that
depends on it (for example, one that avoids existing components) therefore
stays correct: the death evaluates `logpdf` at the same smaller state the
matching birth drew from.

Two requirements are yours to meet:

- **Coverage.** `logpdf` must be finite wherever the component prior is. A
  death whose component has `logpdf = −inf` cannot be reversed and is
  rejected. At the end of each `sample()` call a `RuntimeWarning` gives the
  number rejected in that call, with the running total in parentheses.
  Mixing a sharp density with the prior, as below, guarantees coverage.
- **A fixed density.** A density that changes during the run must stop
  changing at `num_adapt`, by implementing `freeze_adaptation()`.

A draw must also lie inside the declared periodic intervals
(`component_periodic`, see below), and `logpdf` must be the density of that
value. A proposal built on the real line, such as a Gaussian jitter on a phase,
has to wrap its draw with `np.mod` and report the wrapped density. A draw
outside the interval raises `ValueError`; if your draw uses a different
interval, for example `(−π, π)`, declare that interval in `component_periodic`.

`IndependentComponentProposal(draw, logpdf)` wraps state-free functions
`draw(rng)` and `logpdf(u)`, as in the quickstart. The prior is always a valid
birth density, but a poor one when the likelihood is sharp: most births land
where the data rule a component out. A density built from the data does
better. For galactic binaries that is an F-statistic map; for these pulses, a
matched filter over the centre:

```python
edges = np.linspace(LOW[1], HIGH[1], 101)  # 100 centre cells of width 0.1
centres = 0.5 * (edges[:-1] + edges[1:])
templates = np.exp(-0.5 * ((t[None, :] - centres[:, None]) / WIDTH) ** 2)
matched = np.maximum(templates @ data, 0.0) ** 2  # large where a pulse fits the data


class GriddedProposal:
    """Centre from a gridded density mixed with the prior; amplitude from the prior."""

    def __init__(self, edges, weights, mix=0.5):
        self.edges = np.asarray(edges, dtype=float)
        widths = np.diff(self.edges)
        weights = np.asarray(weights, dtype=float)
        self.cell_prob = mix * weights / weights.sum() + (1 - mix) * widths / widths.sum()
        self.log_density = np.log(self.cell_prob / widths)

    def draw(self, rng, x, k):
        cell = rng.choice(self.cell_prob.size, p=self.cell_prob)
        centre = rng.uniform(self.edges[cell], self.edges[cell + 1])
        return np.array([rng.uniform(LOW[0], HIGH[0]), centre])

    def logpdf(self, u, x, k):
        amplitude, centre = u
        if not (LOW[0] <= amplitude <= HIGH[0] and self.edges[0] <= centre <= self.edges[-1]):
            return -np.inf
        cell = min(np.searchsorted(self.edges, centre, side="right") - 1, self.cell_prob.size - 1)
        return self.log_density[cell] - np.log(HIGH[0] - LOW[0])


gridded = RJPTSampler(
    n_shared=0,
    n_component=P,
    max_components=K,
    lnlike=lnlike,
    lnprior=lnprior,
    component_proposal=GriddedProposal(edges, matched),
    ntemps=4,
    seed=1,
    outdir="rj_chains_gridded",
    verbose=False,
)
gridded.sample(np.zeros(K * P), num_iterations=5000, initial_k=0)

for name, run in [("prior births", sampler), ("matched-filter births", gridded)]:
    cold_k = run.load_chain()["k"][0]
    print(f"{name}: cold chain first has k = 2 at saved iteration {np.argmax(cold_k == 2)}")
```

## Custom moves

### Within a model: `add_custom_jump`

A within-model move has the `PTSampler` signature (see
{doc}`custom-proposals`), `move(chain_stats) -> (row, qxy)`, with two
differences: `chain_stats.current_sample` is the full row `(D + 1,)` with `k`
last, and the returned row must keep the same `k` and NaN in the same slots
(the sampler raises `ValueError` naming the move otherwise).
`current_sample` is a read-only view of the live chain state: writing to it
raises `ValueError`, and so does returning it (or any array sharing its
memory). Return a new array, for example `current_sample.copy()` with changes.
`chain_stats.layout` is an {class}`~impulse.RJLayout`, which knows where each
slot is:

```python
class NudgeCentre:
    """Shift one random component's centre by N(0, 0.05^2): symmetric, so qxy = 0."""

    __name__ = "nudge_centre"

    def __call__(self, cs):
        layout, x = cs.layout, cs.current_sample
        k = layout.k(x)
        if k == 0:
            return x.copy(), 0.0
        y = x.copy()
        centre = layout.slot(int(cs.rng.integers(k))).start + 1  # parameter 1 of that slot
        y[centre] += 0.05 * cs.rng.standard_normal()
        return y, 0.0
```

The sampler wraps periodic parameters after a within-model move. That is
harmless for a symmetric random-walk step like this one (`qxy = 0`), but a move
with any other `qxy` must return periodic values already wrapped into their
intervals, and compute `qxy` for those values.

A move with no active coordinates (for example `n_shared = 0` and `k = 0`)
returns the current state, which counts as accepted. Within-model acceptance
rates therefore read high when the chain spends time at low `k`.

`weight` must be finite and at least 0. Moves named `birth` or `death` are
refused: those are the names of the built-in birth/death entries in
`proposal_acceptance_rates()`. Give a callable class a distinct `__name__`.

### Between models: `add_model_jump`

A move that changes `k` has the same signature and returns the new row, whose
last entry is the new `k`. Register a move **and its reverse as one
proposal**: the move picks a direction itself, and its `qxy` includes the
probability of picking the reverse.

For a move from `x` to `y` that draws random numbers `u` with density `g`,
whose reverse draws `u'` with density `g'`,

$$\mathrm{qxy} = \log \frac{j(y \to x)\, g'(u')}{j(x \to y)\, g(u)}
+ \log \left| \det \frac{\partial (y, u')}{\partial (x, u)} \right|,$$

where `j` is the probability of choosing that direction. When the Jacobian is
the same constant for every birth (as when a birth maps `u` to the new
component by a fixed affine map), pass its log as `log_jacobian` and leave it
out of `qxy`: the sampler adds it when `k` grows and subtracts it when `k`
shrinks.

Here a birth draws the new pulse as `θ = μ + σ·u` with `u ~ N(0, I)`, so
`log_jacobian = Σ log σ`. The position factors `1/(k + 1)` (where to insert,
which to remove) cancel between a move and its reverse:

```python
class AffineBirthDeath:
    """Birth: theta = mu + sigma * u, u ~ N(0, I), at a random position. Death: the reverse."""

    __name__ = "affine_birth_death"

    def __init__(self, mu, sigma):
        self.mu, self.sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)

    @staticmethod
    def birth_probability(layout, k):
        if k >= layout.max_components:
            return 0.0
        return 1.0 if k <= layout.min_components else 0.5

    @staticmethod
    def log_normal(u):
        return float(np.sum(-0.5 * u * u - 0.5 * np.log(2 * np.pi)))

    def __call__(self, cs):
        layout, rng, x = cs.layout, cs.rng, cs.current_sample
        k = layout.k(x)
        b = self.birth_probability(layout, k)
        if rng.random() < b:  # birth, at one of k + 1 positions
            u = rng.standard_normal(layout.n_component)
            y = layout.insert(x, int(rng.integers(k + 1)), self.mu + self.sigma * u)
            d_next = 1.0 - self.birth_probability(layout, k + 1)
            return y, np.log(d_next) - np.log(b) - self.log_normal(u)
        y, theta = layout.remove(x, int(rng.integers(k)))  # death, of one of k
        u = (theta - self.mu) / self.sigma
        b_prev = self.birth_probability(layout, k - 1)
        return y, np.log(b_prev) - np.log(1.0 - b) + self.log_normal(u)


SIGMA_BIRTH = np.array([0.5, 2.5])
custom = RJPTSampler(
    n_shared=0,
    n_component=P,
    max_components=K,
    lnlike=lnlike,
    lnprior=lnprior,
    component_proposal=births,
    birth_death_weight=0,  # the built-in birth/death is replaced by the jump below
    ntemps=4,
    seed=2,
    outdir="rj_chains_custom",
    verbose=False,
)
custom.add_model_jump(
    AffineBirthDeath(mu=[1.0, 5.0], sigma=SIGMA_BIRTH),
    weight=40,
    log_jacobian=float(np.sum(np.log(SIGMA_BIRTH))),
)
custom.add_custom_jump(NudgeCentre(), weight=10)
custom.sample(np.zeros(K * P), num_iterations=2000, initial_k=0)
print(sorted(custom.proposal_acceptance_rates()))
```

The same rules apply to model jumps: the returned row must be a new array, and
the sampler checks it (shape, `k` within the limits, finite values before slot
`k`, NaN after it) and raises `ValueError` naming the jump otherwise. Returned
rows must already be wrapped into the periodic intervals, with `qxy` computed
for the wrapped values, unless `qxy` is `−inf`, which means "reject" and is
not checked. `log_jacobian` must be finite.

### Redrawing a component

`component_redraw_weight` (default 0) adds `component_redraw`: replace one
random component with a fresh draw from the component proposal, keeping `k`.
With a good component proposal it lets a component jump straight to another
mode, which local moves can't.

## Periodic parameters

`component_periodic={j: period}` marks parameter `j` of every component as
periodic: for components `(log10 A, f, φ)`, `component_periodic={2: 2 * np.pi}`
wraps every phase. `periodic` takes indices into the padded parameter vector
and is for shared parameters; the two may not overlap.

Component proposals and model jumps must produce values already inside these
closed intervals, with densities and `qxy` computed for those values (see
above); otherwise the sampler raises `ValueError`.

## Reading the chains

`load_chain()` returns the `PTSampler` keys, with:

- `samples`: `(ntemps, n, D)`, NaN past each row's `k`;
- `k`: `(ntemps, n)` integers;
- `shared`: `(ntemps, n, n_shared)` and `components`: `(ntemps, n, K, p)`,
  views of `samples`.

### Label switching

Slot 0 in one sample need not be the same pulse as slot 0 in the next: births
insert at random positions, and nothing in the posterior tells the slots
apart. Per-slot means, traces and R-hat are therefore meaningless. Use
summaries that don't depend on the labels, such as the distribution of all
active centres, or relabel each sample (here, by sorting on the centre) before
summarizing:

```python
comps = chain["components"][0, 1000:]  # (n, K, P), NaN past k
centre_values = comps[..., 1][~np.isnan(comps[..., 1])]  # every active centre, any slot
near = (np.abs(centre_values - 3.0) < 0.2) | (np.abs(centre_values - 6.5) < 0.2)
print("share of centres near a true pulse:", near.mean().round(3))

two = comps[ks == 2][:, :2]  # samples with k = 2: (m, 2, P)
order = np.argsort(two[..., 1], axis=1)  # sort each sample's pulses by centre
relabelled = np.take_along_axis(two, order[..., None], axis=1)
print("mean (amplitude, centre) per pulse:", relabelled.mean(axis=0).round(2).tolist())
```

`grubin` and `effective_sample_size` are meaningful for the shared parameters
and for `k`.

## Checkpoints and resume

Resume works as for `PTSampler` (see {doc}`checkpointing`): rebuild the
sampler the same way, with the same constructor arguments, the same
`add_custom_jump` and `add_model_jump` calls in the same order, and a component
proposal of the same class, then pass `resume=True`. On top of the `PTSampler`
checks, a changed `n_shared`, `n_component`, `max_components`,
`min_components`, component proposal class, or model jump `log_jacobian`
raises `CheckpointMismatchError`.

The sampler cannot tell whether a component proposal was rebuilt from
different data or settings. A proposal can guard against that by returning a
fingerprint from `get_checkpoint_state()` and comparing it in
`set_checkpoint_state(state)`. The returned state must be JSON-serializable
(checkpoints are `.npz` + JSON, so a NumPy array or scalar needs converting
first, as the hexdigest string is here):

```{code-block} python
import hashlib


class CheckedGriddedProposal(GriddedProposal):
    def get_checkpoint_state(self):
        return {"map": hashlib.sha256(self.log_density.tobytes()).hexdigest()}

    def set_checkpoint_state(self, state):
        if state["map"] != self.get_checkpoint_state()["map"]:
            raise ValueError("the birth map differs from the checkpointed run")
```

## JAX

With `vectorized=True, jax=True` the sampler always sends the full batch, so
the function compiles once; `k` changes values, not shape. Mask with
`jnp.where`, and enable 64-bit floats before creating any JAX array (the
{doc}`vectorized likelihoods notebook <../examples/vectorized_likelihoods>`
shows why):

```{code-block} python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
t_jax, data_jax = jnp.asarray(t), jnp.asarray(data)


@jax.jit
def lnlike_jax(x, k):
    comps = x.reshape(-1, K, P)
    active = jnp.arange(K)[None, :] < k[:, None]
    amplitude = jnp.where(active, comps[..., 0], 0.0)  # where, not multiply: 0 * NaN is NaN
    centre = jnp.where(active, comps[..., 1], 0.0)
    shapes = jnp.exp(-0.5 * ((t_jax - centre[..., None]) / WIDTH) ** 2)
    r = data_jax - jnp.sum(amplitude[..., None] * shapes, axis=1)
    return -0.5 * jnp.sum(r * r, axis=1) / SIGMA**2


jax_sampler = RJPTSampler(
    n_shared=0,
    n_component=P,
    max_components=K,
    lnlike=lnlike_jax,
    lnprior=lnprior_vectorized,
    vectorized=True,
    jax=True,
    component_proposal=births,
    ntemps=4,
    outdir="rj_chains_jax",
)
```

## Storage

Each saved row holds `D + 5` values: the padded parameters, `k`, `lnlike`,
`lnprob`, `accepted` and `temperature`. With `K = 20`, `p = 8` and no shared
parameters that is 165 values, about 1.3 kB per row, or 132 MB per
temperature for 100,000 saved rows. `thin` is the main lever.
