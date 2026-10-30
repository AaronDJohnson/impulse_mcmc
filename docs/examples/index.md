# Examples

The repository ships complete, runnable Jupyter notebooks in
[`examples/`](https://github.com/AaronDJohnson/impulse_mcmc/tree/main/examples).
The one below is rendered here with its **stored outputs** — it is never
executed during documentation builds (a full run takes tens of thousands of
iterations).

```{toctree}
:maxdepth: 1

sinusoidal_model
```

## More notebooks in the repository

These additional notebooks are stored without outputs; run them locally:

- [`high_dimensional_test.ipynb`](https://github.com/AaronDJohnson/impulse_mcmc/blob/main/examples/high_dimensional_test.ipynb)
  — a 52-parameter Bayesian linear regression stress test (50,000 iterations
  across 30 temperatures).
- [`normalizing_flow_proposal.ipynb`](https://github.com/AaronDJohnson/impulse_mcmc/blob/main/examples/normalizing_flow_proposal.ipynb)
  — the optional flow-based proposal (`pip install "impulse-mcmc[flow]"`).
