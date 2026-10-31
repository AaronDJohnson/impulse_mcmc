# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - Unreleased

Version 1.0.0 on PyPI is the pre-rewrite package (the old `base.py` /
`mhsampler.py` / `ptsampler.py` API, since removed). 2.0.0 is a complete
rewrite and shares no API with it.

2.0.0 ships one sampler, `PTSampler`. The development builds on the `v2`
branch also carried NUTS, product-space model selection and SBC validation
tools; those are not part of this release (see *Removed*). Entries below that
compare against "earlier 2.0.0-dev builds" matter only if you ran that branch.

### Breaking Changes

- **Chain files are now raw binary by default.** Each temperature writes
  `chain_<i>.bin` — fixed-width `float64`, `ndim + 4` values per row, no
  header or delimiters — instead of `chain_<i>.txt`. Decimal formatting was
  ~18% of wall time on a 21-temperature run; the binary files are also ~3x
  smaller.

  `load_chain()` detects and reads either encoding, so code that goes through
  it is unaffected. Code that reads the files directly must change:

  ```python
  data = np.loadtxt("chains/chain_0.txt")                    # before
  data = np.fromfile("chains/chain_0.bin").reshape(-1, ndim + 4)   # after
  ```

  Pass `chain_format="text"` to keep the previous files. Both encodings store
  identical values (`%.18e` round-trips a `float64` exactly), so this is purely
  a speed/size/readability trade.

- Complete rewrite of the 1.0.0 API: the old `base.py`, `mhsampler.py`, and
  `ptsampler.py` modules are removed. The package is now organized around
  `PTSampler` with adaptive proposals and checkpoint/resume.
- The history-buffer defaults changed from `buffer_size=50_000, buffer_thin=1`
  to `buffer_size=2_000, buffer_thin=25`. The retained HORIZON is unchanged —
  both span 50,000 iterations — but only every 25th state is stored, because
  consecutive MCMC states are highly autocorrelated (measured ~89% redundancy
  at ACT≈9). History-buffer memory drops ~25x.

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
  Checkpoint files are correspondingly larger, but small in absolute terms with
  the thinned buffers (5.8 MB at `ntemps=21, ndim=10` and the default
  `save_freq`).
- `grubin` now takes parameters **only**, shape `(T, D)` — the same contract as
  `effective_sample_size`, and exactly what `load_chain()` returns in
  `chain["samples"][k]`. It previously dropped the last two columns of its
  input, a convention that matched no format this library produces: chain files
  carry four trailing columns (lnlike, lnprob, accepted, temperature) and
  `load_chain` returns none, so R-hat was computed either on two too few
  parameters or on lnlike/lnprob promoted to parameters — silently, with no
  warning. Passing a non-2-D array now raises `ValueError`. If you read a raw
  chain file, slice it yourself: `grubin(data[:, :ndim])`.
- Chains are bit-different from 1.x runs at the same seed. Default-configuration
  chains also differ from earlier 2.0.0-dev builds at the same seed: `de` now
  actually runs once its history buffer holds `de_min_fill` samples (default
  100), where earlier builds silently substituted a plain Gaussian for every
  `de` selection until the buffer was completely full.
- matplotlib is no longer a dependency.
- Checkpoints are no longer pickles. `PTSampler` writes the no-code-execution
  `sampler_checkpoint.npz` + `sampler_checkpoint.json` pair (see Added), and
  `sampler_checkpoint.pkl` files are neither written nor read. Resume stays
  bit-exact, but the reconstruct-then-restore contract now *requires* the same
  proposals to be re-registered before resuming (previously the pickle
  restored them for you).

### Removed

- **Everything outside `PTSampler` that the 2.0.0-dev builds carried.** None
  of it is in 2.0.0, and there are no compatibility aliases:
  - `NUTSSampler`, `compose_logp_and_grad`, `make_logp_and_grad_numerical`,
    `MassMatrix` and `MassMatrixType` (the `impulse.nuts` package).
  - `impulse.experimental`: `HybridPTSampler`, `BirthDeathProductSpace`,
    `make_product_space_sampler` and `load_hybrid_checkpoint`; and from the
    core, `NestedProductSpace`, `model_visitation_stats` and
    `bayes_factor_from_chain`.
  - `impulse.validation`: the SBC functions (`run_sbc_continuous`,
    `run_sbc_model_selection`, `compute_sbc_rank`, `compute_sbc_quantile`,
    `compute_model_pit`, `ecdf`) and plots (`sbc_ecdf_plot`, `coverage_plot`,
    `rank_histogram`), along with the `[plots]` extra.
  - The pickle checkpoint readers `load_checkpoint` and `load_nuts_checkpoint`.
  - `EarlyDE` / `make_early_de`; use `DEProposal(min_fill)` or the sampler's
    `de_min_fill` argument.
  - The `unmanaged_indices` constructor argument, which existed for the
    product-space model index.
- The internal modules were consolidated into `impulse.samplers`,
  `impulse.proposals`, `impulse.chain_stats`, `impulse.chain_io`,
  `impulse.resume` and `impulse.diagnostics`. `impulse.flow_proposals`,
  `impulse.sampler_state`, `impulse.sampler_step`, `impulse.online_updates`,
  `impulse.file_io`, `impulse.utils`, `impulse.wrapping` and
  `impulse.input_function_wrapper` no longer exist; import public names from
  the top-level `impulse` package (`from impulse import NormalizingFlowProposal`).
- `impulse/parallel.py` and its `ParallelLikelihood` class are removed. They
  were never exported from `impulse`, never used by any sampler, never
  documented, and carried 0% test coverage; the shared-memory worker pool also
  deadlocked reliably, which is why the suite had to be run with
  `--ignore=tests/test_parallel.py` in every CI job. The parallelism the
  sampler actually exposes is unaffected: pass `threads=` to use the
  `ThreadPoolExecutor` path in the likelihood wrapper, or `vectorized=True` to
  evaluate a batch yourself.

### Fixed

- **Resuming with a different `chain_format` silently split the run across two
  files.** The encoding lives in the chain files on disk, not in the
  checkpoint, and it was not among the verified run-shaping fields. Resuming a
  `chain_format="binary"` run with `chain_format="text"` raised nothing, wrote
  a second set of `chain_<i>.txt` files, and left `load_chain()` returning only
  the original `.bin` half — so the resumed iterations were invisible.
  `chain_format` is now checkpointed and verified, raising
  `CheckpointMismatchError` on a mismatch.

- **The adaptive proposals were documented as satisfying diminishing
  adaptation. They do not.** The AM/SCAM covariance and the DE difference
  vectors come from a finite history buffer spanning
  `buffer_size * buffer_thin` iterations (50,000 at the defaults); once it
  begins evicting, the kernel keeps changing by a non-vanishing amount and the
  Roberts & Rosenthal (2007) ergodicity theorem no longer applies. Below that
  length the window is still growing and the condition does hold. The claim has
  been removed; the README and the parallel tempering guide now state the
  buffer span, why the window is deliberately finite (a full-history estimator
  carries burn-in forever at weight `1/n`), and that `num_adapt` is how to
  obtain an exactly Markovian kernel. No bias was detectable in testing: on a
  unit Gaussian and an equal-weight bimodal target, analysed strictly after
  eviction begins, adapt-forever matched its `num_adapt`-frozen twin to within
  Monte Carlo error.

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
- Parameter `groups` that do not cover every index now warn at construction:
  `am`/`scam`/`de` only propose within a group, so an uncovered coordinate stays
  frozen and the chain samples a *conditional* of the posterior rather than the
  marginal.
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
  `np.cov` returns a scalar rather than a 1x1 matrix for a single column. The
  covariance update now always produces an `(ndim, ndim)` matrix.
- Differential evolution never ran at realistic run lengths: the stock
  `de` was gated on a completely full history buffer (more than
  `buffer_size` samples) and the jump selector silently substituted a plain
  Gaussian for every `de` selection — the substitution is what hid the dead
  move. `de` is now min-fill-gated (the difference move activates once the
  buffer holds `de_min_fill` samples, default 100, and the proposal returns the
  current position unchanged below the threshold), and the hidden Gaussian
  substitution is removed entirely (the selected proposal always runs as
  registered).
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

- `chain_format` keyword on `PTSampler`: `"binary"` (default) or `"text"`. See
  Breaking Changes. The implementation lives in the new `impulse.chain_io`
  module.

- `ChainStats.update_from_window` / `MultiChainStats.update_from_window`, the
  accurate name for what was called `recursive_update`: the method recomputes
  the mean, covariance and per-group SVD from the current history window rather
  than accumulating over the whole run. `recursive_update` remains as an alias.
  Note that code monkeypatching the alias on an instance (a test spy, a
  profiling wrapper) will no longer intercept the sampler's calls — patch
  `update_from_window` instead.

- `PTSampler`'s previously undocumented constructor arguments now appear in the
  API reference: `chain_format`, `periodic`, `buffer_thin` and `threads`.

- `verbose` keyword on `PTSampler` (default `True`). `verbose=False`
  suppresses the tqdm progress bar — useful for batch/cluster jobs, notebooks,
  and nested loops. It is presentation-only: it is never checkpointed or
  verified on resume, and a run resumed with `verbose=False` stays quiet even
  when the checkpoint came from a verbose run. Pinned by
  `tests/test_verbose.py`, which also asserts the chain is byte-identical with
  and without it.

- No-code-execution checkpoint format: array state in `sampler_checkpoint.npz`
  (`numpy.savez`) plus a metadata sidecar `sampler_checkpoint.json` carrying
  `schema_version` 2. Loading uses `numpy.load(..., allow_pickle=False)` and
  `json.load`, so resuming a checkpoint executes no code — it is as safe as
  reading a data file, including the automatic `resume=True` load from a shared
  `outdir`. The format is written uncompressed for write speed (see above).
  Resume is *reconstruct then restore*: rebuild the sampler exactly as the
  original run did (same constructor and `add_custom_jump` calls), then
  `resume=True` verifies the reconstruction matches the checkpoint (class, the
  seven run-shaping scalars, `chain_format`, and the ordered proposal names and
  weights on every chain) and restores state into it. Both files carry a shared
  write token, so a pair torn by a crash between the two writes is detected and
  ignored. Helpers exported from `impulse`: `save_state_checkpoint`,
  `restore_state_checkpoint`, `load_state_checkpoint`, `checkpoint_sampler`,
  `check_for_checkpoint`, `CheckpointMismatchError` and `TornCheckpointError`.
- `num_adapt` sampler option: freezes covariance updates, the DE history
  buffer, temperature-ladder adaptation, and flow refits after the given
  iteration. Default `None` preserves the historical adapt-forever behavior;
  resume keeps the checkpointed value unless `num_adapt` is passed explicitly.
- Min-fill-gated differential evolution: `de` draws from the filled tail of the
  history buffer and activates once it holds `min_fill` samples, configurable
  per sampler via the `de_min_fill` constructor argument
  (`impulse.DEProposal` is the configurable carrier class).
- The proposal building blocks are exported from `impulse`: `am`, `scam`, `de`,
  `gaussian`, `DEProposal`, `NormalizingFlowProposal`, `JumpProposals` and
  `ProposalBundle`.
- Reproducibility guarantees, enforced by tests: identical seeds produce
  bit-identical chains, and an interrupted-then-resumed run matches an
  uninterrupted one bit-exactly.
- `pytest-timeout` in the `dev` extra, so a local run can bound a hanging
  test with `--timeout=N`. (The specific hang it was originally added for
  lived in `tests/test_parallel.py`, which this release removes; see
  *Removed*.)
- Releases now use PyPI trusted publishing (OIDC). Before the next release,
  trusted publishers must be registered on both pypi.org (environment
  `pypi`) and test.pypi.org (environment `testpypi`) for this repository's
  `deploy.yml` workflow — publishing fails until then.
- This changelog.

### Changed (performance)

- 1.5-1.8x faster than the 2.0.0-dev builds on cheap targets, with
  bit-identical chains at the same seed. The gain comes from consolidating the
  modules plus a bit-exact pass (no random-bit draw for single-group proposals,
  one batched draw for the swap uniforms, one mean pass in the covariance
  update). Wall time on a 10-D Gaussian, default settings, median of three
  interleaved runs:

  | Run | 2.0.0-dev | 2.0.0 | Speedup |
  |---|---|---|---|
  | per-row, 21 temperatures, 20k iterations | 6.16 s | 3.99 s | 1.54x |
  | vectorized, 21 temperatures, 20k iterations | 4.84 s | 2.74 s | 1.77x |
  | per-row, 4 temperatures, 40k iterations | 3.68 s | 2.50 s | 1.47x |

  The gain matters only when the likelihood is cheap: at 21 temperatures a
  whole step, likelihood included, now costs ~9.5 µs per chain per-row and
  ~140 µs per iteration vectorized.

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
- The `[flow]` extra (coppuccino) carries a `python_version >= '3.11'` marker,
  matching coppuccino's own requirement, so `pip install "impulse-mcmc[flow]"`
  still resolves on 3.10 (without the flow proposal).
- The project description and `CITATION.cff` describe the `PTSampler`-only
  scope. The description becomes immutable PyPI metadata per version.

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
- New example notebooks, rendered in the docs: `sinusoid_quickstart`,
  `vectorized_likelihoods` (NumPy, JAX and threads, with timings, and the
  silent float32 conversion JAX makes unless `jax_enable_x64` is set) and
  `sampler_features` (tempering, periodic parameters, `sample_cov`, `groups`,
  a custom proposal, checkpoint/resume). They replace `sinusoidal_model.ipynb`,
  `sinusoid.ipynb` and the `sine_model.py` helper.
