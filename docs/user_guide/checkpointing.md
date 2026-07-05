# Checkpointing and resuming

Long runs should never be lost to a wall-clock limit or a crash. All
samplers write a pickle checkpoint alongside their chain files, and
`PTSampler` / `RJPTSampler` resume from it **bit-exactly**.

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

# ...the next invocation finds <outdir>/sampler_checkpoint.pkl and
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

## The bit-exact guarantee

For checkpoints **written by impulse-mcmc 2.0**, resuming is exact: an
interrupted (or prematurely stopped) run resumed to `N` total iterations
produces chain files *bit-identical* to a single uninterrupted
`N`-iteration run. On resume the chain files are truncated back to the
checkpointed row count and every subsequent iteration is regenerated from
the checkpointed RNG streams. This is enforced by the test suite
(`tests/test_reproducibility.py`).

Boundary conditions — the guarantee holds only if:

- the checkpoint was written by version 2.0 (see the legacy note below);
- you reconstruct the sampler with the **same configuration and
  functions** — the likelihood and prior are *not* stored in the
  checkpoint (they are stripped before pickling and rebound on load), so
  a changed likelihood silently changes the resumed chain;
- you don't change adaptation-relevant options mid-run. `num_adapt` is
  the deliberate exception: when you don't pass it, the resumed run keeps
  the checkpointed value; passing it explicitly (including an explicit
  `None`) overrides the checkpointed value with a warning when they
  differ. Proposals whose frozen state was pickled (e.g. a frozen
  normalizing flow) stay frozen either way.

## Checkpoints are pickles — trust boundary

Checkpoints are Python pickles of whole sampler objects, and **unpickling
can execute arbitrary code**. Loading a checkpoint is equivalent to
running a script from the same source:

- Only resume from checkpoints that you, or a pipeline you trust, wrote.
- `resume=True` auto-loads `sampler_checkpoint.pkl` from `outdir`. If
  `outdir` is on shared or world-writable storage (cluster scratch, group
  project space), anyone who can write there can run code as you on your
  next resume. Keep `outdir` somewhere only you can write, or check
  permissions before resuming.
- Never load checkpoints downloaded from the internet or attached to bug
  reports.

See the project's
[security policy](https://github.com/AaronDJohnson/impulse_mcmc/blob/main/SECURITY.md)
for the full statement of this trust boundary.

This is also why custom proposals must be *picklable* — every registered
proposal is serialized into the checkpoint (see {doc}`custom-proposals`).

## Legacy checkpoints (pre-2.0 RJ runs)

Reversible-jump checkpoints written before the 2.0 detailed-balance fix
registered birth and death as two separate constant-weight jumps — wiring
that biases the model posterior toward fewer sources. On resume,
impulse-mcmc detects this and:

1. **migrates** the checkpoint automatically when reconstruction is safe,
   replacing the pair with the combined `birth_death` kernel (with a
   `UserWarning` telling you that *pre-resume* samples remain biased and
   should be discarded), or
2. warns loudly and leaves the checkpoint untouched when it cannot migrate
   — in that case start a fresh run for correct model posteriors.

Either way: model posteriors built from samples drawn *before* the resume
are biased; regenerate them from post-resume samples.

## Manual load (advanced)

`load_checkpoint` / `load_rjpt_checkpoint` / `load_nuts_checkpoint` give
you the restored sampler object directly, rebinding the functions that
were stripped at checkpoint time:

```python
from impulse import check_for_checkpoint, load_checkpoint

path = check_for_checkpoint("./chains_resume")   # None if no checkpoint
if path is not None:
    sampler = load_checkpoint(path, lnlike=log_likelihood, lnprior=log_prior)
    print("checkpointed iteration:", sampler.short_chain.iteration)
```

Note that `load_checkpoint` expects the functions in the form the sampler
stored them (for `PTSampler` these are its internal wrapped versions when
resuming mid-`sample`), so for ordinary use prefer `resume=True`, which
handles the rebinding and file truncation for you.
