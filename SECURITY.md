# Security Policy

## Supported versions

Only the 2.x series receives security fixes. The 1.0.0 release on PyPI is the
pre-rewrite package and is unsupported.

## Checkpoints: the default format executes no code on load

`PTSampler` and `RJPTSampler` checkpoint to a **no-code-execution format** by
default: `sampler_checkpoint.npz` (array state) plus `sampler_checkpoint.json`
(metadata). Loading uses `numpy.load(..., allow_pickle=False)` and
`json.load` (`impulse/resume.py`), so **loading a checkpoint is as safe as
reading a data file** — a tampered checkpoint cannot execute code. The most a
corrupted checkpoint can do is fail to load or restore wrong numbers; it
cannot run a payload. `resume=True` auto-loads this format from `outdir`, and
that is safe even when `outdir` is shared/world-writable cluster scratch.

The format carries an explicit `schema_version` (in the JSON sidecar). The
loader accepts the current schema and refuses a newer one with a clear error,
rather than misreading it.

## Legacy pickle checkpoints — treat them as executable code

Older checkpoints (`sampler_checkpoint.pkl`) are Python pickles of whole
sampler objects. **Unpickling a file can execute arbitrary code**, so loading
a legacy pickle checkpoint is equivalent to running a script from the same
source. This trust boundary applies **only** to the legacy pickle format:

- Only resume from pickle checkpoints that you, or a pipeline you trust, wrote.
- `resume=True` falls back to `sampler_checkpoint.pkl` **only** when no
  new-format checkpoint is present, and emits a loud security/deprecation
  warning when it does. If `outdir` is on shared or world-writable storage
  (cluster scratch, group project space) and holds a pickle checkpoint,
  anyone who can write there can run code as you on your next resume. Keep
  `outdir` somewhere only you (or your pipeline) can write. To move off the
  pickle format, start a fresh run (`resume=False`, or a new `outdir`): a run
  resumed from a `.pkl` keeps writing `.pkl` for the rest of that run.
- `NUTSSampler` still checkpoints via pickle (its checkpointing is separate
  from the PT engine); the same caution applies.
- Never load pickle checkpoints downloaded from the internet or attached to
  bug reports.

This is inherent to the pickle format, not a bug; a maliciously crafted pickle
checkpoint is out of scope as a vulnerability, but sandbox escapes or code
execution that does *not* require loading untrusted input are in scope. The
pickle format is deprecated and slated for removal in a future 2.x release.

## Reporting a vulnerability

Please report vulnerabilities privately, not in public issues:

- GitHub: [Security advisories](https://github.com/AaronDJohnson/impulse_mcmc/security/advisories/new)
  ("Report a vulnerability" on the repo's Security tab), or
- Email: aaron9035@gmail.com

Include a minimal reproduction if you can. This is a volunteer-maintained
research code; reports are triaged on a best-effort basis.
