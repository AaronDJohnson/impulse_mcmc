# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-07-04

Version 1.0.0 on PyPI is the pre-rewrite package (the old `base.py` /
`mhsampler.py` / `ptsampler.py` API, since removed). 2.0.0 is a complete
rewrite and shares no API with it.

### Breaking Changes

- Complete rewrite of the 1.0.0 API: the old `base.py`, `mhsampler.py`, and
  `ptsampler.py` modules are removed. The package is now organized around
  `PTSampler`, `RJPTSampler`, and `NUTSSampler` with adaptive proposals,
  reversible-jump model selection, checkpoint/resume, and SBC validation
  utilities.
- `DeathProposal` / `make_death_proposal` now require a `draw_from_prior`
  argument (the vacated slot is refreshed with a fresh prior draw).
- `MassMatrix.from_covariance` now inverts its argument (builds
  `M = cov^-1`, matching Stan's convention). Use `MassMatrix.from_precision`
  when supplying Fisher/precision matrices directly.
- `PTSampler.from_rjmcmc` / `RJPTSampler.from_rjmcmc` register one combined
  `birth_death` jump instead of separate birth and death jumps; the
  `birth_proposal` / `death_proposal` keys in acceptance-rate reports are
  replaced by a single `birth_death` key.
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
  + `sampler_checkpoint.json` pair (see Added). `PTSampler` / `RJPTSampler`
  now write the new format; `resume=True` prefers it and falls back to a
  legacy `.pkl` only when no new-format checkpoint is present. Sampling
  behavior is unchanged and resume stays bit-exact, but the on-disk files
  differ, and the reconstruct-then-restore contract now *requires* the same
  proposals to be re-registered before resuming (previously the pickle
  restored them for you).

### Deprecated

- `RJMCMCProductSpace` is renamed to `BirthDeathProductSpace`, which
  accurately describes what it is: a product-space (composite-model-space)
  sampler with birth/death model moves, not dimension-changing reversible
  jump in the Green (1995) sense. `RJMCMCProductSpace` remains importable as
  a deprecated alias and may be removed in a future release.
- The pickle checkpoint format (`sampler_checkpoint.pkl`) is deprecated in
  favor of the no-code-execution `.npz` + `.json` format and is slated for
  removal in a future 2.x release. `load_checkpoint`, `load_rjpt_checkpoint`,
  and `load_nuts_checkpoint` still read pickles but now emit a loud
  security/deprecation warning (unpickling can execute arbitrary code; see
  [SECURITY.md](SECURITY.md)). `NUTSSampler` checkpointing remains on pickle
  for now (its checkpointing is separate from the PT engine).

### Fixed

- Differential evolution never ran at realistic run lengths: the stock
  `de` was gated on a completely full history buffer (more than
  `buffer_size` samples, 50,000 by default) and the jump selector
  silently substituted a plain Gaussian for every `de` selection — the
  substitution is what hid the dead move. `de` is now min-fill-gated
  (the difference move activates once the buffer holds `de_min_fill`
  samples, default 100, and the proposal returns the current position
  unchanged below the threshold), the hidden Gaussian substitution is
  removed entirely (the selected proposal always runs as registered),
  and `from_rjmcmc` no longer registers a weight-0 stock `de`
  placeholder — the unified `de` carries `de_weight` directly.
- RJMCMC detailed balance: the default birth/death configuration biased
  model posteriors toward boundary models and fewer sources. Birth and
  death are now a single combined kernel with exact Hastings terms for the
  move schedule; death removes the last active slot and refreshes it from
  the prior with the matching proposal-ratio term; the death-side
  correction for non-prior source proposals is restored. The production
  proposal mixture is verified pi-invariant by finite-state enumeration of
  the real kernels, guarded by regression tests.
- RJMCMC configuration validation: per-source prior fallbacks are probed at
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
  `RJPTSampler`): array state in `sampler_checkpoint.npz`
  (`numpy.savez_compressed`) plus a schema-versioned `sampler_checkpoint.json`
  metadata sidecar (`schema_version` starts at 1). Loading uses
  `numpy.load(..., allow_pickle=False)` and `json.load`, so resuming a
  checkpoint executes no code — it is as safe as reading a data file, including
  the automatic `resume=True` load from a shared `outdir`. The format is
  smaller than the old pickle (savez compression) and carries an explicit
  schema version in place of implicit pickle-layout compatibility. Resume is
  *reconstruct then restore*: rebuild the sampler exactly as the original run
  did (same constructor / `from_rjmcmc` / `add_custom_jump` calls), then
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
  `de_min_fill` constructor argument on `PTSampler` / `RJPTSampler` and
  their `from_rjmcmc` constructors. This gives reversible-jump runs a
  ridge-following move at realistic run lengths and fixes metastable
  continuous mixing. `EarlyDE` / `make_early_de` remain as
  backward-compatibility aliases for the same implementation
  (`impulse.proposals.DEProposal` is the configurable carrier class).
- Reproducibility guarantees, enforced by tests: identical seeds produce
  bit-identical chains, and an interrupted-then-resumed run matches an
  uninterrupted one bit-exactly for both samplers.
- `plots` optional-dependency extra providing matplotlib for the
  `impulse.validation` plotting helpers.
- `pytest-timeout` in the `dev` extra (guards local full-suite runs against
  the known `tests/test_parallel.py` hang; CI skips that module).
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
