# Security Policy

## Supported versions

Only the 2.x series receives security fixes. The 1.0.0 release on PyPI is the
pre-rewrite package and is unsupported.

## Checkpoints are pickles — treat them as executable code

Sampler checkpoints (`sampler_checkpoint.pkl`) are Python pickles of whole
sampler objects (`impulse/resume.py`). **Unpickling a file can execute
arbitrary code**, so loading a checkpoint is equivalent to running a script
from the same source:

- Only resume from checkpoints that you, or a pipeline you trust, wrote.
- Note that `resume=True` automatically loads `sampler_checkpoint.pkl` from
  the sampler's `outdir`. If that directory is on shared or world-writable
  storage (cluster scratch, group project space), anyone who can write there
  can run code as you on your next resume. Keep `outdir` somewhere only you
  (or your pipeline) can write, or check permissions before resuming.
- Never load checkpoints downloaded from the internet or attached to bug
  reports.

This is inherent to the pickle format, not a bug; a maliciously crafted
checkpoint is out of scope as a vulnerability, but sandbox escapes or code
execution that does *not* require loading untrusted input are in scope.

## Reporting a vulnerability

Please report vulnerabilities privately, not in public issues:

- GitHub: [Security advisories](https://github.com/AaronDJohnson/impulse_mcmc/security/advisories/new)
  ("Report a vulnerability" on the repo's Security tab), or
- Email: aaron9035@gmail.com

Include a minimal reproduction if you can. This is a volunteer-maintained
research code; reports are triaged on a best-effort basis.
