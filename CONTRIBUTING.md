# Contributing to impulse-mcmc

Thanks for contributing! This is a research code for parallel-tempering MCMC,
NUTS, and reversible-jump model selection — correctness of the sampler kernels
is the top priority, and the sections below explain how we defend it.

## Development setup

Requires Python >= 3.10.

```bash
git clone https://github.com/AaronDJohnson/impulse_mcmc.git
cd impulse_mcmc
pip install -e ".[dev]"
```

The `dev` extra installs pytest (with coverage and timeout plugins), black,
isort, and flake8. Plotting helpers need matplotlib (`pip install -e ".[plots]"`).

## Running the tests

```bash
pytest tests/ -m "not slow" --ignore=tests/test_parallel.py
```

- The `slow` marker (declared in `pyproject.toml`) gates long statistical
  runs; CI runs the fast suite on every push and the slow suite in a separate
  job. Run `pytest tests/ -m slow` before submitting changes to sampler kernels.
- `tests/test_parallel.py` exercises multiprocessing and is known to hang on
  some platforms. It is excluded from CI and carries a 120 s per-test timeout
  locally (via `pytest-timeout`, included in the `dev` extra) — run it
  deliberately, not as part of the default sweep.
- If matplotlib is installed and your home directory is read-only (e.g. on a
  cluster), point its cache somewhere writable before running the plot tests:
  `export MPLCONFIGDIR=$TMPDIR/mpl`.

## Formatting and linting

black and isort are enforced in CI (`black --check`, `isort --check-only`),
with line length 100 and isort's black profile (configured in `pyproject.toml`):

```bash
black impulse tests
isort impulse tests
```

## Typing

The package ships `py.typed`. Keep `mypy impulse` clean (install mypy
separately if you don't already have it).

## Statistical tests and new proposals

Sampler bugs often don't crash — they silently bias posteriors. Two rules:

1. **Detailed-balance regression tests must pass.** The tests in
   `tests/test_rjmcmc_detailed_balance.py` — including the deterministic
   exact-enumeration test (`test_exact_enumeration_stationarity`), which
   enumerates the real kernels with zero Monte Carlo noise — are the guard
   against transdimensional bias. Never weaken their tolerances to make a
   change pass; a failure there means the kernel is wrong.
2. **New proposals need a `qxy` correctness argument.** Proposals return
   `(sample, qxy)` with `qxy = log[q(x|y) / q(y|x)]`. A PR adding or changing
   a proposal must include a short written derivation of why its `qxy` is
   correct (in the docstring or PR description), plus a test — ideally an
   extension of the detailed-balance suite.

## API stability (2.x)

The 2.x series follows semver intent:

- **Public API** is what `impulse/__init__.py` exports in `__all__`. Removing
  or breaking a public name gets a deprecation warning at least one minor
  release before removal.
- **Internals** — underscore-prefixed names and module paths not re-exported
  in `__all__` — may change in any release without warning.
- **Checkpoint compatibility** is best-effort across patch and minor
  versions: checkpoints are pickles of sampler objects (see
  [SECURITY.md](SECURITY.md)), so internal refactors can break old
  checkpoints. Don't rely on resuming a long run across an upgrade.

User-visible changes should get an entry in `CHANGELOG.md` (Keep a Changelog
format).
