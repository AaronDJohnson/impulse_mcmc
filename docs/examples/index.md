# Examples

The repository ships complete, runnable Jupyter notebooks in
[`examples/`](https://github.com/AaronDJohnson/impulse_mcmc/tree/main/examples).
The three below are rendered here with their **stored outputs**; they are
never executed during documentation builds. They share one model, a sinusoid
in noise, and are meant to be read in order:

```{toctree}
:maxdepth: 1

sinusoid_quickstart
vectorized_likelihoods
sampler_features
```

- **Quickstart**: likelihood and prior, running the sampler, reading the
  chains, and checking convergence.
- **Vectorized likelihoods**: batched NumPy, JAX (and why to enable 64-bit
  floats), threads, and timings showing when each pays off.
- **Sampler features**: tempering on a two-mode posterior, periodic
  parameters, starting covariances and groups, a custom proposal, and
  checkpoint/resume with `num_adapt`.

## More notebooks in the repository

These additional notebooks are stored without outputs; run them locally:

- [`high_dimensional_test.ipynb`](https://github.com/AaronDJohnson/impulse_mcmc/blob/main/examples/high_dimensional_test.ipynb)
  — a 52-parameter Bayesian linear regression stress test (50,000 iterations
  across 30 temperatures).
- [`normalizing_flow_proposal.ipynb`](https://github.com/AaronDJohnson/impulse_mcmc/blob/main/examples/normalizing_flow_proposal.ipynb)
  — the optional flow-based proposal (`pip install "impulse-mcmc[flow]"`).
