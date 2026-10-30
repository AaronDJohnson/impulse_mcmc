# Security Policy

## Supported versions

Only the 2.x series receives security fixes. The 1.0.0 release on PyPI is the
pre-rewrite package and is unsupported.

## Checkpoints: the default format executes no code on load

`PTSampler` checkpoints to a **no-code-execution format**:
`sampler_checkpoint.npz` (array state) plus `sampler_checkpoint.json`
(metadata). Loading uses `numpy.load(..., allow_pickle=False)` and
`json.load` (`impulse/resume.py`), so **loading a checkpoint is as safe as
reading a data file** — a tampered checkpoint cannot execute code. The most a
corrupted checkpoint can do is fail to load or restore wrong numbers; it
cannot run a payload. `resume=True` auto-loads this format from `outdir`, and
that is safe even when `outdir` is shared/world-writable cluster scratch.

The format carries an explicit `schema_version` (in the JSON sidecar). The
loader accepts the current schema and refuses a newer one with a clear error,
rather than misreading it.

## Pickle checkpoints are not loaded

Older versions wrote `sampler_checkpoint.pkl`, a Python pickle of the whole
sampler. Unpickling can execute arbitrary code, so 2.0.0 neither writes nor
reads pickle checkpoints: `resume=True` ignores a `.pkl` file. (With no usable
checkpoint beside existing chain data it raises, and the message points out the
pickle so you can finish that run with the version that wrote it.) Never
unpickle checkpoint files downloaded from the internet or attached to bug
reports.

## Reporting a vulnerability

Please report vulnerabilities privately, not in public issues:

- GitHub: [Security advisories](https://github.com/AaronDJohnson/impulse_mcmc/security/advisories/new)
  ("Report a vulnerability" on the repo's Security tab), or
- Email: aaron9035@gmail.com

Include a minimal reproduction if you can. This is a volunteer-maintained
research code; reports are triaged on a best-effort basis.
