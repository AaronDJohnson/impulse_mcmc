# Examples

The repository ships complete, runnable Jupyter notebooks in
[`examples/`](https://github.com/AaronDJohnson/impulse_mcmc/tree/main/examples).
The two below are rendered here with their **stored outputs** — they are
never executed during documentation builds (full runs take tens of
thousands of iterations).

```{toctree}
:maxdepth: 1

sinusoidal_model
rjmcmc_sinusoids
```

## More notebooks in the repository

These additional notebooks are stored without outputs; run them locally:

- [`high_dimensional_test.ipynb`](https://github.com/AaronDJohnson/impulse_mcmc/blob/main/examples/high_dimensional_test.ipynb)
  — a 52-parameter Bayesian linear regression stress test (50,000 iterations
  across 30 temperatures).
- [`rjpt_sinusoids.ipynb`](https://github.com/AaronDJohnson/impulse_mcmc/blob/main/examples/rjpt_sinusoids.ipynb)
  — product-space model selection with `HybridPTSampler`.
- [`rjmcmc_nuts_sinusoids.ipynb`](https://github.com/AaronDJohnson/impulse_mcmc/blob/main/examples/rjmcmc_nuts_sinusoids.ipynb)
  — hybrid model selection + NUTS + parallel tempering.
- [`product_space_sinusoids.ipynb`](https://github.com/AaronDJohnson/impulse_mcmc/blob/main/examples/product_space_sinusoids.ipynb)
  — the lower-level product-space embedding.
- [`sbc_gaussian.ipynb`](https://github.com/AaronDJohnson/impulse_mcmc/blob/main/examples/sbc_gaussian.ipynb)
  and
  [`sbc_rjmcmc.ipynb`](https://github.com/AaronDJohnson/impulse_mcmc/blob/main/examples/sbc_rjmcmc.ipynb)
  — simulation-based calibration of the continuous and model-selection
  pipelines.
- [`normalizing_flow_proposal.ipynb`](https://github.com/AaronDJohnson/impulse_mcmc/blob/main/examples/normalizing_flow_proposal.ipynb)
  — the optional flow-based proposal (`pip install "impulse-mcmc[flow]"`).
