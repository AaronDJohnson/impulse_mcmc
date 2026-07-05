---
name: Bug report
about: Something crashed, or a sampler produced results you believe are wrong
labels: bug
---

## Environment

- impulse-mcmc version (`python -c "import impulse; print(impulse.__version__)"`):
- Python version:
- numpy version:
- OS / platform (laptop, cluster, ...):

## Minimal script

The smallest script that reproduces the problem (small `ndim`, few
iterations, toy likelihood if possible):

```python
# your script here
```

## Expected vs observed

- What you expected to happen:
- What actually happened (paste the full traceback if it crashed):

## If the bug is statistical (biased posterior, wrong Bayes factor, ...)

- Random seed(s) used and whether the result persists across seeds:
- Number of iterations / temperatures, and any convergence diagnostics you
  checked (Gelman-Rubin, effective sample size, acceptance rates):
- Please do **not** attach checkpoint `.pkl` files — they are pickles and we
  won't load them. Attach chain text output or a script instead.
