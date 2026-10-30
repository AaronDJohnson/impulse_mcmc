# Checkpointing and resuming

Long runs should never be lost to a wall-clock limit or a crash.
`PTSampler` checkpoints alongside its chain files and resumes from the
checkpoint **bit-exactly**.

The checkpoint is a **no-code-execution** pair of files —
`sampler_checkpoint.npz` (array state) and `sampler_checkpoint.json`
(metadata) — so loading a checkpoint is as safe as reading a data file (see
[Loading is safe](#loading-is-safe-the-no-code-execution-format) below).

## Basic usage

Pass `resume=True` and re-run the *same* construction and `sample()` call:

```python
import numpy as np
from impulse import PTSampler

def log_likelihood(x):
    return -0.5 * np.sum(x**2)

def log_prior(x):
    if np.all(np.abs(x) <= 5):
        return 0.0
    return -np.inf

def make_sampler():
    return PTSampler(ndim=2, lnlike=log_likelihood, lnprior=log_prior,
                     ntemps=4, seed=42, save_freq=1000,
                     outdir="./chains_resume", resume=True)

# First invocation: no checkpoint exists yet, so this starts fresh and
# checkpoints every save_freq iterations. Suppose it is killed early...
make_sampler().sample(np.zeros(2), num_iterations=5_000)

# ...the next invocation finds <outdir>/sampler_checkpoint.json and
# continues from it up to the SAME global target.
make_sampler().sample(np.zeros(2), num_iterations=5_000)
```

The semantics to remember:

- **`num_iterations` is a global target, not an increment.** A resumed run
  continues from the checkpointed iteration counter up to
  `num_iterations`. To extend a finished 5 000-iteration run to 20 000,
  resume with `num_iterations=20_000`.
- A checkpoint is written at the **end** of every iteration `jj` with
  `jj > 0 and jj % save_freq == 0`, capturing the sampler after that
  iteration fully completed (post PT-swap, post adaptation), including
  every RNG stream.
- `resume=True` with no checkpoint and empty chain files simply starts
  fresh — so the same script works for both the first submission and every
  requeue. `resume=True` with **no** usable checkpoint but chain files that
  already hold data raises `RuntimeError` instead of appending a cold-start
  run onto the existing chain.

## The resume contract: reconstruct, then restore

The checkpoint does **not** contain your likelihood, prior, or any proposal
code — callables cannot be stored in a data file. Resume therefore works in
two steps:

1. **Reconstruct** the sampler exactly as the original run did — the same
   constructor arguments and the same `add_custom_jump` registrations, in the
   same order.
2. **Restore** the saved state into that reconstructed sampler (`resume=True`
   does this for you).

Before restoring, the loader **verifies** the reconstruction matches the
checkpoint metadata — sampler class; the run-shaping scalars `ndim`, `ntemps`,
`swap_steps`, `cov_update`, `save_freq`, `buffer_size` and `buffer_thin`;
`chain_format`; and the ordered proposal names *and* weights on every chain.
If anything differs it raises `CheckpointMismatchError` naming the first
mismatch (a missing custom jump, a changed weight, a changed `save_freq`),
rather than silently restoring into the wrong sampler:

```python
from impulse import CheckpointMismatchError

class StepJump:
    __name__ = "step_jump"

    def __call__(self, chain_stats):
        x = chain_stats.current_sample.copy()
        x += chain_stats.rng.normal(0.0, 0.5, size=x.shape)
        return x, 0.0

def make_custom_sampler(weight):
    sampler = PTSampler(ndim=2, lnlike=log_likelihood, lnprior=log_prior,
                        ntemps=4, seed=1, outdir="./chains_custom_resume",
                        resume=True)
    sampler.add_custom_jump(StepJump(), weight)   # same jump, same order
    return sampler

make_custom_sampler(10).sample(np.zeros(2), num_iterations=2_000)   # first run
make_custom_sampler(10).sample(np.zeros(2), num_iterations=4_000)   # resume

try:
    make_custom_sampler(20).sample(np.zeros(2), num_iterations=6_000)
except CheckpointMismatchError as err:
    print(err)   # names 'step_jump' and both weights
```

## The bit-exact guarantee

Resuming is exact: an interrupted (or prematurely stopped) run resumed to `N`
total iterations produces chain files *bit-identical* to a single
uninterrupted `N`-iteration run. On resume the chain files are truncated back
to the checkpointed row count and every subsequent iteration is regenerated
from the checkpointed RNG streams. This is enforced by the test suite
(`tests/test_reproducibility.py`).

Bit-exactness holds because the JSON sidecar stores every generator's
`bit_generator.state` (exact integers) and the `.npz` stores all floating-point
state losslessly.

Boundary conditions — the guarantee holds only if:

- you reconstruct the sampler with the **same configuration and functions**
  (the likelihood and prior are not stored; a changed likelihood silently
  changes the resumed chain), and re-register the **same proposals with the
  same weights** (the verification above enforces this);
- you don't change adaptation-relevant options mid-run. `num_adapt` is
  the deliberate exception: when you don't pass it, the resumed run keeps
  the checkpointed value; passing it explicitly (including an explicit
  `None`) overrides the checkpointed value with a warning when they
  differ. Proposals whose frozen state was serialized (e.g. a frozen
  normalizing flow) stay frozen either way.

## Loading is safe: the no-code-execution format

The checkpoint is two files in `outdir`:

- `sampler_checkpoint.npz` — all array state (positions, log-densities, the
  temperature ladder, per-chain adaptive statistics, DE history buffers, and
  the in-memory chain buffer), written with `numpy.savez`. It is deliberately
  UNCOMPRESSED: compression measured 80x slower on a realistic payload and
  made checkpoint writes ~41% of wall time, so the files are larger but the
  writes are cheap.
- `sampler_checkpoint.json` — a schema-versioned metadata sidecar: the
  `schema_version`, the impulse version, the sampler class and constructor
  echo, every RNG bit-generator state, the ordered proposal names and weights,
  and the scalar bookkeeping.

Loading uses `numpy.load(..., allow_pickle=False)` and `json.load` — **no
pickle is involved**, so a tampered or corrupted checkpoint cannot execute
code. The worst a bad checkpoint can do is fail to load. This matters because
`resume=True` auto-loads from `outdir`, which is often shared cluster scratch;
that auto-load is as safe as reading a data file.

The two files are written atomically: both go to temp files and are renamed
into place with the JSON sidecar committed **last**. Each write stamps both
files with the same token, so a crash between the two renames (a fresh `.npz`
beside a stale `.json`) is detected as torn and the checkpoint is ignored on
the next resume.

## Schema versioning and compatibility

The JSON sidecar carries a `schema_version` (currently `2`). The loader reads
the current schema and **refuses a newer one** with a clear error rather than
misreading it — so a checkpoint written by a future impulse will not be
silently mis-restored by an older one. Schema bumps are documented in
`CHANGELOG.md`. Checkpoint compatibility across versions is best-effort:
because resume is reconstruct-then-restore, a refactor of constructor wiring
or of a component's serialized state can break old checkpoints, so don't rely
on resuming a long run across an upgrade.

Pickle checkpoints (`sampler_checkpoint.pkl`, written by older versions) are
neither written nor read. If `resume=True` finds one
next to existing chain data, the `RuntimeError` above says so; finish that run
with the version that wrote it.

## Manual load (advanced)

`resume=True` handles discovery, verification, restoration, and chain-file
truncation for you, and is the recommended path. If you need the pieces
directly:

```python
from impulse import check_for_checkpoint, restore_state_checkpoint

path = check_for_checkpoint("./chains_resume")   # None if no usable checkpoint
if path is not None:
    sampler = make_sampler()                     # reconstruct with the SAME wiring
    restore_state_checkpoint(sampler, path)      # verifies + restores in place
    print("checkpointed iteration:", sampler.short_chain.iteration)
```

`restore_state_checkpoint` verifies the reconstructed sampler against the
metadata (raising `CheckpointMismatchError` on a mismatch) and restores state
into it. It does not truncate the chain files; `resume=True` does that before
sampling continues.
