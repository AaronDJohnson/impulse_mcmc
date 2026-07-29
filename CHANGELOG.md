# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-07-26

Version 1.0.0 on PyPI is the pre-rewrite package (the old `base.py` /
`mhsampler.py` / `ptsampler.py` API, since removed). 2.0.0 is a complete
rewrite and shares no API with it.

### Breaking Changes

- Complete rewrite of the 1.0.0 API: the old `base.py`, `mhsampler.py`, and
  `ptsampler.py` modules are removed. The package is now organized around
  `PTSampler`, `HybridPTSampler`, and `NUTSSampler` with adaptive proposals,
  product-space (birth-death) model selection, checkpoint/resume, and SBC
  validation utilities.
- `DeathProposal` / `make_death_proposal` now require a `draw_from_prior`
  argument (the vacated slot is refreshed with a fresh prior draw).
- `MassMatrix.from_covariance` now inverts its argument (builds
  `M = cov^-1`, matching Stan's convention). Use `MassMatrix.from_precision`
  when supplying Fisher/precision matrices directly.
- The history-buffer defaults changed from `buffer_size=50_000, buffer_thin=1`
  to `buffer_size=2_000, buffer_thin=25`. The retained HORIZON is unchanged —
  both span 50,000 iterations — but only every 25th state is stored, because
  consecutive MCMC states are highly autocorrelated (measured ~89% redundancy
  at ACT≈9). Memory drops ~25x: 2100 MB → 84 MB at the documented
  product-space defaults (`num_models=5, num_params=10, ntemps=21`).

  Verified that this does NOT introduce adaptation bias, which is the reason
  thinning is used instead of simply shrinking the buffer. Eigen-whitened
  `E[z^2]` on a 6-D correlated Gaussian, 8 seeds x 60k iterations from exact
  stationary starts (truth 1.0):

  | buffer_size | thin | horizon | width error |
  |---|---|---|---|
  | 50000 | 1 | 50000 | +0.02% |
  | 2000 | 25 | 50000 | −0.10% |
  | 2000 | 1 | **2000** | **−0.60%** |

  Bias tracks the horizon, not the row count: at identical memory, the thinned
  buffer has 6x less bias. Chains are bit-different from earlier 2.0.0-dev
  builds at the same seed. Set `buffer_thin=1` for the previous behavior.

  Note `de_min_fill` counts STORED rows, so at the new defaults the
  differential-evolution move activates after `100 * 25 = 2500` iterations
  rather than 100. Short runs may want `buffer_thin=1`.
- Checkpoints are written with `np.savez` instead of `np.savez_compressed`.
  Compression measured 360 ms against 4.5 ms on a realistic payload — an 80x
  cost, most of it spent deflating the zero padding in the history buffers —
  and checkpoint writes were ~41% of wall time at the default `save_freq`. A
  20,000-iteration run at `ntemps=8, ndim=10` went from 10.12 s to 4.68 s.
  Checkpoint files are correspondingly larger and an uncompressed `.npz` can
  now exceed the equivalent pickle; with the thinned buffers the absolute size
  is small (2.2 MB at the defaults above).
- `CHECKPOINT_SCHEMA_VERSION` is now 2. When per-model statistics are active,
  `update_sample` makes the chain's `_buffer` the *same object* as the current
  model's buffer, and the checkpoint was writing it under both `"buffer"` and
  `"m{k}.buffer"` — storing the dominant array twice (measured 25% of the
  buffer payload). The duplicate is dropped and the alias recorded in metadata.
  Version 1 checkpoints still load; version 2 checkpoints will not load on
  older impulse, which is what the version gate is for.
- `grubin` now takes parameters **only**, shape `(T, D)` — the same contract as
  `effective_sample_size`, and exactly what `load_chain()` returns in
  `chain["samples"][k]`. It previously dropped the last two columns of its
  input, a convention that matched no format this library produces: chain files
  carry four trailing columns (lnlike, lnprob, accepted, temperature) and
  `load_chain` returns none, so R-hat was computed either on two too few
  parameters or on lnlike/lnprob promoted to parameters — silently, with no
  warning. Passing a non-2-D array now raises `ValueError`. If you read a raw
  `chain_*.txt`, slice it yourself: `grubin(data[:, :ndim])`.
- `PTSampler.from_product_space` / `HybridPTSampler.from_product_space` register one combined
  `birth_death` jump instead of separate birth and death jumps; the
  `birth_proposal` / `death_proposal` keys in acceptance-rate reports are
  replaced by a single `birth_death` key.
- The "RJMCMC"/"reversible-jump" misnomer is gone from the public API: this
  package does product-space (composite-model-space) model selection with
  birth/death moves, not dimension-changing reversible jump in the Green
  (1995) sense. The 2.0.0-dev names are renamed and **removed with no
  aliases** (nothing has shipped — 1.0.0 on PyPI is the unrelated pre-rewrite
  package): `RJMCMCProductSpace` → `BirthDeathProductSpace`, `RJPTSampler` →
  `HybridPTSampler`, `PTSampler.from_rjmcmc` / `HybridPTSampler.from_rjmcmc` →
  `from_product_space`, and `load_rjpt_checkpoint` → `load_hybrid_checkpoint`.
- The internal module files were renamed to match: `impulse.rjmcmc` →
  `impulse.birth_death`, `impulse.rjmcmc_proposals` →
  `impulse.birth_death_proposals`, and `impulse.rjpt_sampler` →
  `impulse.hybrid_sampler`. There are no import-path shims; import the public
  classes and functions from the top-level `impulse` package.
- Chains are bit-different from 1.x runs at the same seed (RNG stream
  changes from the combined birth/death kernel and the min-fill-gated
  `de` move). Default-configuration chains also differ from earlier
  2.0.0-dev builds at the same seed: `de` now actually runs once its
  history buffer holds `de_min_fill` samples (default 100), where
  earlier builds silently substituted a plain Gaussian for every `de`
  selection until the buffer was completely full (more than
  `buffer_size` samples, 50,000 by default — never reached at realistic
  run lengths).
- `impulse.sampler_step.pt_step` drops its unused `lnlike_fn` /
  `lnprior_fn` parameters (PT swaps only permute cached values; nothing
  was ever recomputed), and the never-used `ChainStats.proposals_ready`
  property is removed.
- matplotlib is no longer a hard dependency. The plotting helpers in
  `impulse.validation` now require the `plots` extra
  (`pip install impulse-mcmc[plots]`); the numeric SBC functions remain
  matplotlib-free.
- The default checkpoint format changed from a pickle
  (`sampler_checkpoint.pkl`) to the no-code-execution `sampler_checkpoint.npz`
  + `sampler_checkpoint.json` pair (see Added). `PTSampler` / `HybridPTSampler`
  now write the new format; `resume=True` prefers it and falls back to a
  legacy `.pkl` only when no new-format checkpoint is present. Sampling
  behavior is unchanged and resume stays bit-exact, but the on-disk files
  differ, and the reconstruct-then-restore contract now *requires* the same
  proposals to be re-registered before resuming (previously the pickle
  restored them for you).

### Deprecated

- The pickle checkpoint format (`sampler_checkpoint.pkl`) is deprecated in
  favor of the no-code-execution `.npz` + `.json` format and is slated for
  removal in a future 2.x release. `load_checkpoint`, `load_hybrid_checkpoint`,
  and `load_nuts_checkpoint` still read pickles but now emit a loud
  security/deprecation warning (unpickling can execute arbitrary code; see
  [SECURITY.md](SECURITY.md)). `NUTSSampler` checkpointing remains on pickle
  for now (its checkpointing is separate from the PT engine).

### Removed

- `impulse/parallel.py` and its `ParallelLikelihood` class are removed. They
  were never exported from `impulse`, never used by any sampler, never
  documented, and carried 0% test coverage; the shared-memory worker pool also
  deadlocked reliably, which is why the suite had to be run with
  `--ignore=tests/test_parallel.py` in every CI job. The parallelism the
  samplers actually expose is unaffected: pass `threads=` to use the
  `ThreadPoolExecutor` path in the likelihood wrapper, or `vectorized=True` to
  evaluate a batch yourself. Removing the module lifted measured coverage from
  84% to 87% and let CI drop its exclusions.

### Fixed

- `resume=True` with no usable checkpoint no longer silently appends a fresh
  cold-start run onto an existing chain (a completed 3000-row chain became 6000
  rows with an unconverged transient spliced into the middle, with no warning).
  It now raises `RuntimeError`. **This is a behavior change**: a run that
  previously "succeeded" by corrupting its own output now stops. `resume=True`
  on an empty output directory still starts fresh.
- The adaptive covariance could collapse into an absorbing state: `am`/`scam`
  take their entire step scale from `proposal_L`, so a posterior narrower than
  the initial `sample_cov` started at the mode rejected every early proposal,
  filled the history buffer with identical rows, and drove `proposal_L` to
  exactly zero — after which no default proposal could move that coordinate
  again. The chain reported a point mass with `sd = 0.0` while its acceptance
  rate looked healthy (0.82). Fixed with a Haario et al. (2001) `epsilon * I`
  ridge scaled to the initial covariance.
- Adaptation was starved whenever `cov_update > save_freq`: the history ring was
  sized `save_freq` but the adaptation refresh asks for `cov_update` samples, so
  `get_recent_samples` silently clamped. At `save_freq=100, cov_update=2000` the
  covariance saw 101 of 4000 samples. Reported by @thompsonphys (issue #11).
- Chain thinning restarted its phase at every flush, so `thin=3` with
  `save_freq=10` wrote iterations 0,3,6,9,10,13,... (gaps 3,3,3,1) instead of a
  uniform 3 — a periodic artifact of period `save_freq` in any autocorrelation
  or ESS estimate computed from the saved file. **Changes on-disk chain contents
  for any `thin > 1` run.**
- `loglargs`/`loglkwargs`/`logpargs`/`logpkwargs` were dropped on the NUTS path
  of `HybridPTSampler`: the NUTS transition called the raw user callables, so
  the MH step targeted the intended density while NUTS targeted the function's
  defaults. With a defaulted extra argument this silently sampled the wrong
  distribution (measured sd 5.05 against a true 4.0); with a required one the
  run died mid-sampling with a `TypeError`.
- Parameter `groups` that do not cover every index now warn at construction:
  `am`/`scam`/`de` only propose within a group, so an uncovered coordinate stays
  frozen and the chain samples a *conditional* of the posterior rather than the
  marginal. The new `unmanaged_indices` argument declares indices that another
  proposal moves (the product-space model index, moved by birth/death).

- Resuming a run with a different `save_freq`, `buffer_size`, `cov_update` or
  `swap_steps` was silently accepted even though the checkpoint recorded all
  four. A changed `save_freq` discarded chain rows (600 of 2000 in a measured
  case, no warning); a changed `buffer_size` broke the
  `len(_buffer) == buffer_size` invariant and crashed the differential-evolution
  proposal with an out-of-bounds index. All seven run-shaping scalars are now
  verified on resume with an error that names the field and both values, and
  `ChainStats.set_checkpoint_state` restores `buffer_size` alongside the buffer.
- 1-parameter models (`ndim=1`) crashed at the first covariance update with
  `IndexError: too many indices for array: array is 0-dimensional`, because
  `np.cov` returns a scalar rather than a 1x1 matrix for a single column. Both
  the global and per-model covariance paths now promote with `np.atleast_2d`.
- Differential evolution never ran at realistic run lengths: the stock
  `de` was gated on a completely full history buffer (more than
  `buffer_size` samples, 50,000 by default) and the jump selector
  silently substituted a plain Gaussian for every `de` selection — the
  substitution is what hid the dead move. `de` is now min-fill-gated
  (the difference move activates once the buffer holds `de_min_fill`
  samples, default 100, and the proposal returns the current position
  unchanged below the threshold), the hidden Gaussian substitution is
  removed entirely (the selected proposal always runs as registered),
  and `from_product_space` no longer registers a weight-0 stock `de`
  placeholder — the unified `de` carries `de_weight` directly.
- Birth-death detailed balance: the default birth/death configuration biased
  model posteriors toward boundary models and fewer sources. Birth and
  death are now a single combined kernel with exact Hastings terms for the
  move schedule; death removes the last active slot and refreshes it from
  the prior with the matching proposal-ratio term; the death-side
  correction for non-prior source proposals is restored. The production
  proposal mixture is verified pi-invariant by finite-state enumeration of
  the real kernels, guarded by regression tests.
- Product-space configuration validation: per-source prior fallbacks are probed at
  construction (provably wrong setups raise), schedules are validated, and
  single-model spaces skip trans-dimensional registration. Legacy
  checkpoints with the old separate birth/death wiring are migrated on
  resume when reconstruction is safe.
- NUTS mass-matrix adaptation used the inverse of Stan's convention, making
  adaptation harmful on anisotropic targets; the dense path no longer
  double-inverts, and constructors copy caller arrays. The `energy_error`
  diagnostic now reports the true Delta-H at the selected leaf.
- Parallel-tempering log-probabilities were stale for one acceptance step
  after temperature-ladder adaptation; the infinite-temperature chain no
  longer produces `0 * (-inf) = NaN` silent rejections.
- The non-vectorized function wrapper allocates float64 output, fixing an
  `OverflowError` when an int-returning prior meets `-inf` rows.
- Checkpoint/resume reproducibility: checkpoints are written at exact
  iteration boundaries, covariance-update cadence is persisted, the resume
  flag and checkpoint path survive state restoration, and chain files
  truncate to the checkpointed row count on resume.

### Added

- No-code-execution checkpoint format (now the default for `PTSampler` /
  `HybridPTSampler`): array state in `sampler_checkpoint.npz`
  (`numpy.savez`) plus a schema-versioned `sampler_checkpoint.json`
  metadata sidecar (`schema_version` starts at 1). Loading uses
  `numpy.load(..., allow_pickle=False)` and `json.load`, so resuming a
  checkpoint executes no code — it is as safe as reading a data file, including
  the automatic `resume=True` load from a shared `outdir`. The format is
  written uncompressed for write speed (see above) and carries an explicit
  schema version in place of implicit pickle-layout compatibility. Resume is
  *reconstruct then restore*: rebuild the sampler exactly as the original run
  did (same constructor / `from_product_space` / `add_custom_jump` calls), then
  `resume=True` verifies the reconstruction matches the checkpoint (class,
  `ndim`, `ntemps`, and the ordered proposal names and weights) and restores
  state into it. Bit-exact resume is preserved. New helpers in
  `impulse.resume`: `save_state_checkpoint`, `restore_state_checkpoint`,
  `load_state_checkpoint`, and `CHECKPOINT_SCHEMA_VERSION`; a `format` keyword
  on `checkpoint_sampler` (default new format).
- `MassMatrix.from_precision` for injecting Fisher/precision matrices.
- `num_adapt` sampler option: freezes covariance updates, temperature-ladder
  adaptation, mass-matrix and dual-averaging updates (finalized to the
  smoothed step size), and flow refits after the given iteration. Default
  `None` preserves the historical adapt-forever behavior; resume keeps the
  checkpointed value unless `num_adapt` is passed explicitly.
- Min-fill-gated differential evolution: `de` draws from the filled tail
  of the (per-model, in RJ configurations) history buffer and activates
  once it holds `min_fill` samples, configurable per sampler via the new
  `de_min_fill` constructor argument on `PTSampler` / `HybridPTSampler` and
  their `from_product_space` constructors. This gives product-space model-selection runs a
  ridge-following move at realistic run lengths and fixes metastable
  continuous mixing. `EarlyDE` / `make_early_de` remain as
  backward-compatibility aliases for the same implementation
  (`impulse.proposals.DEProposal` is the configurable carrier class).
- Reproducibility guarantees, enforced by tests: identical seeds produce
  bit-identical chains, and an interrupted-then-resumed run matches an
  uninterrupted one bit-exactly for both samplers.
- `plots` optional-dependency extra providing matplotlib for the
  `impulse.validation` plotting helpers.
- `pytest-timeout` in the `dev` extra, so a local run can bound a hanging
  test with `--timeout=N`. (The specific hang it was originally added for
  lived in `tests/test_parallel.py`, which this release removes; see
  *Removed*.)
- Releases now use PyPI trusted publishing (OIDC). Before the next release,
  trusted publishers must be registered on both pypi.org (environment
  `pypi`) and test.pypi.org (environment `testpypi`) for this repository's
  `deploy.yml` workflow — publishing fails until then.
- This changelog.

### Changed (packaging)

- Version is 2.0.0 and is now read dynamically from `impulse.__version__`
  (single source of truth).
- Dependency floors relaxed to `numpy>=1.24`, `scipy>=1.10`, `tqdm>=4.60` —
  the previous floors were latest-at-authoring pins, not real requirements.
  A floor-pinned CI job verifies these minimums on push.
- Build backend floor raised to `setuptools>=77`, required for the PEP 639
  SPDX `license = "MIT"` expression (builds fail on older setuptools).
- Classifiers: Python 3.10-3.13, `Development Status :: 4 - Beta`, and
  `Typing :: Typed` (the package ships `py.typed`).
- Added `MANIFEST.in`. Without it setuptools fell back to the legacy distutils
  `test*.py` glob, which collects `tests/test_*.py` but drops
  `tests/conftest.py` — so the shipped source distribution contained a test
  suite that could not run (relevant to downstream packagers: conda-forge,
  Debian, Spack). The sdist now also ships `CHANGELOG.md`, `CITATION.cff`,
  `CONTRIBUTING.md`, `SECURITY.md`, and the documentation sources. Note this
  defect is invisible when building in a working tree that contains a stale
  `*.egg-info/SOURCES.txt`; verify sdist contents from a clean clone.
- The project description and `CITATION.cff` no longer say "reversible-jump".
  This string becomes immutable PyPI metadata per version, and the term was
  retired from the API in this same release.

### Changed (documentation)

- The requirement that custom proposals be **picklable** is withdrawn — the
  `.npz` + `.json` checkpoint format stores no code, so proposals are not
  serialized at all and a lambda proposal now checkpoints and resumes fine.
  What replaces it is the reconstruct-then-restore contract: re-register the
  same proposals, in the same order, with the same weights, which the loader
  verifies (`CheckpointMismatchError` on mismatch). Callable classes are still
  recommended, but for a stable `__name__` — which keys acceptance reports and
  the resume check — not for picklability.
- Documented the `get_checkpoint_state()` / `set_checkpoint_state()` protocol,
  which is how an adaptive proposal persists internal state under the default
  format. It was previously undocumented, so a proposal that adapted between
  calls silently restarted from its constructor defaults on resume.
