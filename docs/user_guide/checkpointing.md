# Checkpointing and resuming

Long runs should never be lost to a wall-clock limit or a crash. `PTSampler`
and `HybridPTSampler` checkpoint alongside their chain files and resume from the
checkpoint **bit-exactly**.

The default checkpoint format is a **no-code-execution** pair of files —
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
- `resume=True` with no checkpoint present simply starts fresh — so the
  same script works for both the first submission and every requeue.

## The resume contract: reconstruct, then restore

The checkpoint does **not** contain your likelihood, prior, or any proposal
code — callables cannot be stored in a data file, and the product space and
custom proposals are code. Resume therefore works in two steps:

1. **Reconstruct** the sampler exactly as the original run did — the same
   constructor arguments, the same `from_product_space` call, and the same
   `add_custom_jump` registrations, in the same order.
2. **Restore** the saved state into that reconstructed sampler (`resume=True`
   does this for you).

Before restoring, the loader **verifies** the reconstruction matches the
checkpoint metadata — sampler class; the run-shaping scalars `ndim`, `ntemps`,
`swap_steps`, `cov_update`, `save_freq`, `buffer_size` and `buffer_thin`; and the ordered
proposal names *and* weights on every chain. If anything differs it raises a
clear error naming the first mismatch (e.g. a missing custom jump, a changed
weight, or a changed `save_freq`), rather than silently restoring into the wrong
sampler:

```python
import numpy as np
from impulse.experimental import BirthDeathProductSpace, HybridPTSampler

NUM_PARAMS, NUM_SOURCES = 2, 3
LO, HI = np.array([0.0, 0.0]), np.array([5.0, 3.0])
data = np.zeros(32)

def loglike(params):
    return -0.5 * float(np.sum(data**2))

def logprior(params):
    p = np.asarray(params).reshape(-1, NUM_PARAMS)
    return 0.0 if np.all((p >= LO) & (p <= HI)) else -np.inf

def draw(rng):
    return rng.uniform(LO, HI)

space = BirthDeathProductSpace(loglike, logprior, num_sources=NUM_SOURCES,
                               num_params=NUM_PARAMS, source_prior_draw=draw)

def make_rj_sampler():
    # Same space, same weights, and the same custom jumps as the original run:
    return HybridPTSampler.from_product_space(space, ntemps=8, seed=1,
                                              outdir="./chains_rj", resume=True)

x0 = space.draw_initial_position(np.random.default_rng(1))
make_rj_sampler().sample(x0, num_iterations=2_000)   # first run
make_rj_sampler().sample(x0, num_iterations=4_000)   # resume — same wiring
```

This is how resume always worked in practice — the pickle format stored
functions *by reference*, so you always needed the same importable
definitions. The new format just makes the requirement explicit and checked.

## The bit-exact guarantee

Resuming is exact: an interrupted (or prematurely stopped) run resumed to `N`
total iterations produces chain files *bit-identical* to a single
uninterrupted `N`-iteration run. On resume the chain files are truncated back
to the checkpointed row count and every subsequent iteration is regenerated
from the checkpointed RNG streams. This is enforced by the test suite
(`tests/test_reproducibility.py`) — the format swap is invisible to it.

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
  temperature ladder, per-chain and per-model adaptive statistics, DE history
  buffers, NUTS mass matrices and sample buffers), written with
  `numpy.savez`. It is deliberately UNCOMPRESSED: compression measured 80x
  slower on a realistic payload and made checkpoint writes ~41% of wall
  time, so the files are larger but the writes are cheap.
- `sampler_checkpoint.json` — a schema-versioned metadata sidecar: the
  `schema_version`, the impulse version, the sampler class and constructor
  echo, every RNG bit-generator state, the ordered proposal names and weights,
  and the scalar bookkeeping.

Loading uses `numpy.load(..., allow_pickle=False)` and `json.load` — **no
pickle is involved**, so a tampered or corrupted checkpoint cannot execute
code. The worst a bad checkpoint can do is fail to load. This matters because
`resume=True` auto-loads from `outdir`, which is often shared cluster scratch;
with the new format that auto-load is as safe as reading a data file.

The two files are written atomically: both go to temp files and are renamed
into place with the JSON sidecar committed **last**, so an interrupted write
(a `.npz` with no `.json`) is detected as torn and ignored on the next resume.

## Schema versioning and compatibility

The JSON sidecar carries a `schema_version` (currently `2`). The loader reads
the current schema and **refuses a newer one** with a clear error rather than
misreading it — so a checkpoint written by a future impulse will not be
silently mis-restored by an older one. Schema bumps are documented in
`CHANGELOG.md`. Checkpoint compatibility across versions is best-effort:
because resume is reconstruct-then-restore, a refactor of constructor wiring
or of a component's serialized state can break old checkpoints, so don't rely
on resuming a long run across an upgrade.

## Legacy pickle checkpoints

Older checkpoints are a single `sampler_checkpoint.pkl` — a Python pickle of
the whole sampler. **Unpickling can execute arbitrary code.** `resume=True`
falls back to a `.pkl` only when no new-format checkpoint is present, and
`load_checkpoint` / `load_hybrid_checkpoint` / `load_nuts_checkpoint` emit a loud
security/deprecation warning when they read one:

- Only resume from pickle checkpoints you (or a pipeline you trust) wrote.
- Keep `outdir` somewhere only you can write. A run resumed from a `.pkl`
  keeps writing `.pkl` for the rest of that run; to move fully to the new
  format, start a fresh run (`resume=False`, or a new `outdir`).
- `NUTSSampler` still checkpoints via pickle (its checkpointing is separate
  from the PT engine).

The pickle format is deprecated and slated for removal in a future 2.x
release. See the project's
[security policy](https://github.com/AaronDJohnson/impulse_mcmc/blob/main/SECURITY.md)
for the full trust boundary.

### Legacy birth-death checkpoints (pre-2.0 detailed-balance fix)

Birth-death *pickle* checkpoints written before the 2.0 detailed-balance
fix registered birth and death as two separate constant-weight jumps — wiring
that biases the model posterior toward fewer sources. On resume from such a
pickle, impulse-mcmc detects this and either **migrates** it automatically to
the combined `birth_death` kernel (with a `UserWarning` telling you that
*pre-resume* samples remain biased and should be discarded) or, when it cannot
migrate safely, warns loudly and leaves the checkpoint untouched — in which
case start a fresh run for correct model posteriors.

## Manual load (advanced)

`resume=True` handles discovery, verification, restoration, and chain-file
truncation for you, and is the recommended path. If you need the pieces
directly:

```python
from impulse.resume import restore_state_checkpoint, check_for_checkpoint

path = check_for_checkpoint("./chains_resume")   # None if no checkpoint
if path is not None and path.endswith(".json"):
    sampler = make_sampler()          # reconstruct with the SAME wiring
    restore_state_checkpoint(sampler, path)   # verifies + restores in place
    print("checkpointed iteration:", sampler.short_chain.iteration)
```

`restore_state_checkpoint` verifies the reconstructed sampler against the
metadata (raising `impulse.resume.CheckpointMismatchError` on a mismatch) and
restores state into it. The legacy pickle loaders
(`load_checkpoint` / `load_hybrid_checkpoint` / `load_nuts_checkpoint`) return
the restored sampler object directly and rebind the stripped callables, but
they unpickle — only use them on checkpoints you trust.
```
