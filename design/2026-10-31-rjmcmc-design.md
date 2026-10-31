# Reversible-jump sampler (`RJPTSampler`): design

Status: approved in discussion 2026-10-30/31; awaiting spec review.
Branch: `rjmcmc`, based on `refactor` (PR #12, the PTSampler-only 2.0.0).

## Goal and scope

Add a reversible-jump MCMC sampler with parallel tempering for problems with a
variable number `k` of copies of one component type (sinusoids, galactic
binaries, glitches, spline knots), plus optional shared parameters that every
model has. The state is `(k, θ_k)` with `θ_k` of length `n_shared + k·p`;
moves between models change the dimension and are accepted with Green's
ratio.

In scope: the variable-count model space, built-in birth/death, an optional
component-redraw move, user-defined within-model and between-model moves,
storage, checkpoint/resume, tests and docs.

Out of scope now (see "Future work"): named model sets with different
parameter vectors, a gridded-density/F-statistic helper (stays in the user's
galactic-binary code), saving only the cold chain.

## Architecture

`RJPTSampler` subclasses `PTSampler`. The model index `k` is stored as an
extra **last column** of the position array, so swaps, chain files,
checkpoints and the MH step work unchanged. The padded parameter vector has
`D = n_shared + K·p` entries (`K = max_components`): shared parameters first,
then `K` component slots of `p` parameters each. Slots past `k` hold NaN. The
padding is storage only: inactive slots are never sampled, never passed to the
prior as values, and carry no density.

New code lives in one new module, `impulse/reversible_jump.py`; every public
name is exported from top-level `impulse`. PTSampler's own code paths
(`am`, `scam`, `de`, `ChainStats`, `vectorized_mh_step`, `pt_step`) are not
modified, so its chains stay bit-identical to v2 (golden comparison 52/52).
Shared helpers may be factored out of `proposals.py` / `chain_stats.py` only
if that leaves PTSampler bit-exact.

Units:

| Unit | Responsibility |
|---|---|
| `RJLayout` | Owns the slot convention: `k(row)`, `params(row)`, `components(row)`, `active_mask(k)`, `insert(row, j, u)`, `remove(row, j) -> (row, u)`. `insert` and `remove` take and return full rows (`(D + 1,)`) and update the k column. The only place that knows the layout. |
| `MaskedChainStats` (ChainStats subclass) | Masked covariance from NaN-padded history, per-k active-block factors, exposes `layout`. |
| `rj_am`, `rj_scam`, `rj_de` | Within-model moves on the active coordinates. |
| `BirthDeath`, `ComponentRedraw` | Built-in moves using the component proposal. |
| `IndependentComponentProposal` | Wraps state-free `draw(rng)` / `logpdf(u)` functions as a component proposal. |
| `RJPTSampler` | Wiring, validation, function wrappers, storage and checkpoint extras. |

## Public API

```python
from impulse import RJPTSampler, IndependentComponentProposal

K, p = 8, 3

def lnlike(x, k):                  # x: (2 + K*p,), slots past k are NaN
    shared, comps = x[:2], x[2:].reshape(K, p)[:k]
    ...

def lnprior(x, k):                 # full log prior over numbered slots, incl. log p(k)
    ...

sampler = RJPTSampler(
    n_shared=2, n_component=p, max_components=K, min_components=0,
    lnlike=lnlike, lnprior=lnprior,
    component_proposal=IndependentComponentProposal(draw, logpdf),
    birth_death_weight=40,         # default; ~30% of moves with default AM/SCAM/DE weights
    component_redraw_weight=0,     # default off
    component_periodic=None,       # e.g. {6: 2*np.pi, 7: np.pi}, applied to every slot
    ntemps=8, outdir="./rj_chains", seed=1,   # any PTSampler keyword
)
sampler.add_custom_jump(within_model_move, weight=10)
sampler.add_model_jump(between_model_move, weight=10, log_jacobian=0.0)
sampler.sample(x0, num_iterations=100_000, initial_k=0)

chain = sampler.load_chain()
```

### User functions

- `lnlike(x, k)` and `lnprior(x, k)`. Per-row mode: `x` is `(D,)`, `k` an
  `int`. Vectorized and JAX mode: `x` is `(batch, D)`, `k` is `(batch,)`.
  `loglargs`/`loglkwargs`/`logpargs`/`logpkwargs` and `threads` work as in
  PTSampler. JAX users mask with `jnp.where(mask, ..., 0)`, since `0 * NaN`
  is NaN.
- `lnprior(x, k)` is the joint prior density over **numbered slots**. For
  exchangeable components it is `log p(k) + Σ_{i<k} log π(θ_i)` (+ shared
  prior), with no `k!`. If the user imposes an ordering to avoid label
  switching, the prior needs `+ log k!` and births succeed only at the one
  order-preserving insert position; the guide recommends relabelling after
  sampling instead.
- The prior is untempered and the likelihood tempered, as in PTSampler.

### Component proposal

An object with `draw(rng, x, k) -> u` (shape `(p,)`) and
`logpdf(u, x, k) -> float`, where `(x, k)` is always the **smaller** model's
state (padded `(D,)` vector without the k column, and its `k`). Requirements,
documented and partly checked:

- `logpdf` is the normalized log density of exactly what `draw` returns.
- Coverage: `logpdf` must be finite wherever the component prior is.
- A density that changes during sampling is an adaptive proposal and must
  implement `freeze_adaptation()` (called at `num_adapt`).

Optional `get_checkpoint_state()` / `set_checkpoint_state(state)` hooks are
honoured as for other stateful proposals.

### Custom moves

- `add_custom_jump(move, weight)`: within-model. `move(chain_stats) ->
  (row, qxy)`; `row` is `(D + 1,)` with `row[-1] == k`. Must not change `k`
  or write to inactive slots (checked; see Errors).
- `add_model_jump(move, weight, log_jacobian=0.0)`: between-model. Same
  signature; `row[-1]` is the new `k`. `qxy` is Green's ratio without the
  Jacobian when `log_jacobian` is used:
  `qxy = log[ j(y→x) g'(u') / ( j(x→y) g(u) ) ] (+ log|det J| if state-dependent)`.
  The sampler adds `+log_jacobian` when `k` increases and `−log_jacobian`
  when it decreases. A move and its reverse are registered as **one**
  proposal, which computes `j(y→x)` itself.
- Moves read the layout as `chain_stats.layout` (an `RJLayout`), whose
  `insert` / `remove` do the slot shifting.
- Registered moves are wrapped; the wrapper preserves `__name__`,
  `get_checkpoint_state` / `set_checkpoint_state` and `freeze_adaptation`.

### `sample()` and initial state

`sample(initial_position, num_iterations, thin=1, initial_k=None)`.
`initial_position` is `(D,)` or `(ntemps, D)` (padded, no k column).
`initial_k` is an int or one value per temperature; default
`min_components`. Entries past `initial_k` are set to NaN. Shared values and
the first `initial_k` slots must be finite, and `lnlike`/`lnprior` at the
start must be finite (existing check).

### `load_chain()`

PTSampler keys, with:

- `samples`: `(ntemps, n, D)`, NaN-padded, no k column;
- `k`: `(ntemps, n)` integers;
- `shared`: view `(ntemps, n, n_shared)`;
- `components`: view `(ntemps, n, K, p)`.

## Moves

### Move selection

Each chain selects its own move from the weights with its own RNG stream
(existing `JumpProposals` behaviour; required by the user).

### Built-in birth/death (one proposal, `birth_death`)

In the formulas below, `x` and `y` are rows; `draw` and `logpdf` receive
`layout.params(row)` and the row's `k`.

- `b_k = d_k = 1/2` for `min < k < max`; `b_min = 1`, `d_max = 1`. If
  `min_components == max_components` the move is not registered.
- Birth: `j ~ U{0..k}`, `u ~ draw(rng, x, k)`, `y = layout.insert(x, j, u)`;
  `qxy = log d_{k+1} − log b_k − logpdf(u, x, k)`.
- Death: `j ~ U{0..k−1}`, `(y, u) = layout.remove(x, j)`;
  `qxy = log b_{k−1} − log d_k + logpdf(u, y, k−1)`.
- The `1/(k+1)` position factors cancel. The map is a slot insertion, so the
  Jacobian is exactly 1 and is not computed.
- Acceptance is the existing MH step:
  `log α = (lnL(y) − lnL(x))/T + lnprior(y, k') − lnprior(x, k) + qxy`.
- A death with `logpdf(u) = −inf` is rejected; a single `RuntimeWarning`
  reports how often it happened.
- Acceptance reports list `birth` and `death` separately, per chain.

### `component_redraw` (optional, default weight 0)

For `k ≥ 1`: `j ~ U{0..k−1}`, `(x₋ⱼ, xⱼ) = layout.remove(x, j)`,
`u ~ draw(rng, x₋ⱼ, k−1)`, `y = layout.insert(x₋ⱼ, j, u)`;
`qxy = logpdf(xⱼ, x₋ⱼ, k−1) − logpdf(u, x₋ⱼ, k−1)`. At `k = 0` it returns the
current state.

### Within-model moves (masked full vector)

- One history buffer per chain stores padded rows (NaN in inactive slots).
  The k column is excluded from all statistics and groups.
- Masked covariance: entry `(a, b)` uses only stored rows where both `a` and
  `b` are active (pairwise-complete). Entries with fewer than 10 such rows
  take the initial covariance's entry (`sample_cov`, default identity). The
  matrix is then made positive semi-definite by clipping eigenvalues at the
  existing ridge (`COV_RIDGE_REL × initial scale`).
- Per-k factors: for group `g` at model `k`, the factor of
  `Σ[active, active]` (active = `g ∩ active_mask(k)`) via `svd_groups`,
  computed on first use and cached until the next covariance update.
- `rj_am` / `rj_scam`: as `am` / `scam` on the active block, with
  `n_eff` = number of active coordinates in the group. Groups with no active
  coordinates at the current `k` are skipped; if none remain the move returns
  the current state with `qxy = 0`.
- `rj_de`: two stored rows; the difference on the group's active coordinates,
  with 0 where either row lacks the slot; scale uses `n_eff`. Symmetric
  because swapping the rows negates the difference with the same mask.
- Weights: the usual `am_weight`, `scam_weight`, `de_weight`, `de_min_fill`.
- `groups` index the padded parameter vector (`0..D−1`).

### Swaps

Unchanged `pt_step`: whole rows swap, including k; acceptance uses only
log-likelihoods and temperatures. Ladder adaptation unchanged.

## Storage, resume, errors

### Chain files

Unchanged `ShortChain`, both formats. Row: `D` padded parameters, k, lnlike,
lnprob, accepted, temperature. Binary stores NaN natively; text writes `nan`.
Size: K=20, p=8 gives 165 columns, 1.3 KB per row, 132 MB per temperature per
100k saved rows; `thin` is the lever.

### Checkpoints

Same `.npz` + JSON format and reconstruct-then-restore. The NaN-padded
buffer and masked covariance are stored as is; the per-k factor cache is not
stored and is rebuilt deterministically, so interrupted-and-resumed runs stay
bit-identical to uninterrupted ones. Added to the verified run-shaping
settings (mismatch raises `CheckpointMismatchError`): `n_shared`,
`n_component`, `max_components`, `min_components`, each model jump's
`log_jacobian`, and the component proposal's class name. The existing
`sampler_class` check separates PTSampler and RJPTSampler checkpoints.
Limitation (documented): a component proposal rebuilt from different data or
settings is not detected unless it implements `get_checkpoint_state`, e.g.
returning a hash of its map.

### Construction-time checks (`ValueError`)

- `n_component ≥ 1`, `0 ≤ min_components ≤ max_components`, `n_shared ≥ 0`.
- `sample_cov` is `(D, D)`.
- `groups` within `0..D−1`.
- `periodic` keys within `0..D−1`; `component_periodic` keys within
  `0..p−1`, expanded to every slot; the two may not overlap.
- `component_proposal` has callable `draw` and `logpdf`.
- `initial_k` within limits; finite active entries in `initial_position`.

### Run-time checks

- Model jump returns k out of range, wrong NaN pattern, or a non-finite
  active value: `ValueError` naming the jump.
- Custom within-model jump changes k or writes an inactive slot: `ValueError`
  naming the jump.
- Built-in moves are correct by construction and not re-checked.
- NaN from `lnprior` or `lnlike` counts as −inf (existing behaviour).

### Diagnostics

`grubin` and `effective_sample_size` apply to shared parameters and to k.
Per-slot statistics are not meaningful because of label switching
(documented).

## Tests

Statistical (fixed seeds; long runs marked `slow`, short smoke versions in the
fast suite):

1. Prior only: `lnlike = 0`, truncated-Poisson `p(k)`. k histogram matches
   `p(k)` (χ²); active components match their prior (KS). Once with the prior
   as birth density, once with a narrow density mixed with the prior.
2. Analytic posterior on k: `y_n ~ N(Σ_{i<k} a_i, σ²)`, `a_i ~ N(0, s²)`,
   closed-form `Z_k`; `ntemps > 1`.
3. Constant Jacobian: built-in birth/death off; one model jump implementing
   an affine birth/death pair (`u ~ N(0, 1)`, `θ = μ + σu`,
   `log_jacobian = Σ log σ`); recovers test 2's posterior.
4. State-dependent component proposal centred on the existing components'
   mean; prior-only invariance.
5. `component_redraw` at fixed k (`min = max`) against an analytic posterior.
6. Power checks: tests 1–3 each also run a deliberately broken kernel
   (always kill the last component; omit `log d/b`, which only matters at
   the k limits, so that test puts prior mass there; `log_jacobian = 0` where
   wrong) and assert the test detects it.

Unit:

- Masked covariance equals brute-force `np.cov` on co-active rows; fallback
  below 10 rows; PSD after repair. Per-k factor equals the active block's
  factor. `rj_de` difference symmetry; `n_eff` uses the active count.
- `RJLayout.insert` / `remove` are inverses; `active_mask`.
- Every construction-time and run-time check raises, naming the jump where
  applicable; the `logpdf = −inf` warning fires once.
- Chain I/O round-trips NaN in both formats; `load_chain` shapes and views.

Reproducibility and modes:

- Same seed, bit-identical chains. Interrupted-and-resumed equals
  uninterrupted, bit for bit, with a model jump and `component_periodic`.
- Each new run-shaping setting raises `CheckpointMismatchError` on change.
- Vectorized mode gives the same chain as per-row mode.
- JAX test (`pytest.importorskip("jax")`, x64 enabled) with the masked
  likelihood pattern.
- PTSampler golden comparison against v2 still 52/52; existing suite passes.

## Docs

- `docs/user_guide/reversible-jump.md`: the `(k, θ_k)` state; the
  numbered-slot prior (no `k!`); a quickstart small enough for
  `scripts/check_doc_snippets.py`; component proposals (exact density,
  coverage, smaller-state conditioning) with an F-statistic-shaped example;
  custom model jumps (Green's ratio, one proposal per move pair, an
  affine-birth example with `log_jacobian`); resume rules; label switching;
  JAX masking.
- API pages: `RJPTSampler` in `samplers.rst`; `IndependentComponentProposal`,
  `RJLayout` in `proposals.rst`.
- README: short reversible-jump section.
- CHANGELOG 2.0.0: update the "PTSampler only" scope paragraph; add an Added
  entry.
- `examples/rj_sinusoids.ipynb` (unknown number of sinusoids in noise),
  rendered on the examples page.
- Doc-snippet check and Sphinx `-W` pass.

## Delivery

Separate PR from `rjmcmc`. If PR #12 has merged by then, the PR targets `v2`;
otherwise it targets `ptsampler-2.0`. 2.0.0 is released after this lands.

## Future work

- Named model sets: the index column becomes a model id and `RJLayout` maps
  each id to its active columns. Masked statistics, storage and checkpoints are
  unchanged; moves between named models are model jumps.
- Saving only the cold chain (a PTSampler option).
