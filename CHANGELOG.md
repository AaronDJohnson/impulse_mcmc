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
  changes from the combined birth/death kernel and the new EarlyDE
  proposal).
- matplotlib is no longer a hard dependency. The plotting helpers in
  `impulse.validation` now require the `plots` extra
  (`pip install impulse-mcmc[plots]`); the numeric SBC functions remain
  matplotlib-free.

### Fixed

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

- `MassMatrix.from_precision` for injecting Fisher/precision matrices.
- `num_adapt` sampler option: freezes covariance updates, temperature-ladder
  adaptation, mass-matrix and dual-averaging updates (finalized to the
  smoothed step size), and flow refits after the given iteration. Default
  `None` preserves the historical adapt-forever behavior; resume keeps the
  checkpointed value unless `num_adapt` is passed explicitly.
- EarlyDE: a min-fill-gated differential-evolution proposal drawing from
  the filled tail of the per-model buffer, registered by `from_rjmcmc`
  (`de_min_fill`). This gives reversible-jump runs a ridge-following move
  at realistic run lengths and fixes metastable continuous mixing.
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
