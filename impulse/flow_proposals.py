"""Normalizing-flow independence-Metropolis proposal.

Fits a normalizing flow (via the optional ``coppuccino`` package) to the
recent sample buffer of the cold chain and uses that flow as an
independence proposal for Metropolis-Hastings. Because draws are
independent of the current state, the proposal ratio is

    qxy = log q(x_current) - log q(x_proposal)

where ``q`` is the flow density. Positive ``qxy`` favors acceptance, in
keeping with the rest of the library.

The dependency is optional. The class raises ``ImportError`` with an
install hint at construction time if ``coppuccino`` (and its JAX-based
dependencies) are not available.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from impulse.chain_stats import ChainStats

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "NormalizingFlowProposal requires the optional `coppuccino` package "
    "(https://github.com/AaronDJohnson/coppuccino). Install with "
    "`pip install coppuccino`, or `pip install -e .` from a local checkout."
)


def _require_coppuccino():
    try:
        from coppuccino import normalizing_flows_fit, sample, log_prob  # type: ignore
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e
    return normalizing_flows_fit, sample, log_prob


class NormalizingFlowProposal:
    """Independence proposal using a normalizing flow.

    Two modes:

    1. **Adaptive (default)** — fits a copula flow to the cold chain's
       recent samples and refits periodically as the chain evolves.
    2. **Fixed** — pass a pre-fitted ``coppuccino`` flow via ``flow=...``.
       The flow is used as-is for the entire run; no fitting is performed.
       Useful when you already have a good density estimate (e.g. from a
       previous run, importance-sampled draws, or simulator output).

    Parameters
    ----------
    flow : optional
        A pre-fitted ``coppuccino`` flow (the object returned by
        ``coppuccino.normalizing_flows_fit`` or ``coppuccino.load_flow``).
        When provided, refitting is disabled and the proposal is active
        from iteration 0 (no warm-up needed).
    refit_interval : int
        Number of cold-chain calls between flow refits (default 2000).
        Ignored when ``flow`` is provided.
    min_samples : int
        Minimum cold-chain samples buffered before the first fit. Until
        reached, the proposal is a stay-put no-op. Ignored when ``flow``
        is provided.
    max_epochs : int
        Forwarded to ``coppuccino.normalizing_flows_fit``. Ignored when
        ``flow`` is provided.
    prior_bounds : np.ndarray, optional
        ``(ndim, 2)`` bounds forwarded to the flow fitter. Recommended
        when the prior is bounded. Ignored when ``flow`` is provided.
    rng_seed : int
        Seed for flow fitting reproducibility. Ignored when ``flow`` is
        provided.
    fit_kwargs : dict, optional
        Extra kwargs forwarded to ``normalizing_flows_fit``.
    cold_chain_only : bool
        If True (default), only the cold (T=1) chain triggers refits.
        Hot chains reuse the cold flow as their proposal.

    Notes
    -----
    - In **fixed** mode, the flow is dropped during pickle (JAX function
      references cannot be pickled reliably). On resume from checkpoint,
      re-attach with :meth:`set_flow` before continuing — otherwise the
      proposal silently becomes a stay-put no-op.
    - The optional ``coppuccino`` import is resolved at construction.
      Without it, instantiation raises ``ImportError`` with an install hint.
    """

    __name__ = "nf_flow"

    def __init__(
        self,
        *,
        flow=None,
        refit_interval: int = 2000,
        min_samples: int = 500,
        max_epochs: int = 200,
        prior_bounds: Optional[np.ndarray] = None,
        rng_seed: int = 0,
        fit_kwargs: Optional[dict] = None,
        cold_chain_only: bool = True,
    ):
        # Fail fast if the optional dep is missing.
        self._fit_fn, self._sample_fn, self._logprob_fn = _require_coppuccino()

        self.refit_interval = int(refit_interval)
        self.min_samples = int(min_samples)
        self.max_epochs = int(max_epochs)
        self.prior_bounds = (
            None if prior_bounds is None else np.asarray(prior_bounds, dtype=float)
        )
        self.rng_seed = int(rng_seed)
        self.fit_kwargs = dict(fit_kwargs or {})
        self.cold_chain_only = bool(cold_chain_only)

        # Fixed-flow mode: a pre-fitted flow disables refitting.
        self.fixed = flow is not None
        self.flow = flow
        self._call_count = 0
        self._last_fit_at = -10**9
        self._fit_count = 0

    def set_flow(self, flow) -> None:
        """Attach (or replace) a pre-fitted flow and disable refitting.

        Call this after restoring a fixed-flow proposal from a checkpoint,
        since the flow itself is dropped during pickle.
        """
        self.flow = flow
        self.fixed = True

    def __getstate__(self):
        # Drop the bound coppuccino functions and the JAX-backed flow.
        # The flow contains function references (e.g. softplus) that can't
        # pickle reliably across JAX versions. On restore, we refit lazily
        # from the chain's sample buffer on the next call.
        state = self.__dict__.copy()
        state["_fit_fn"] = None
        state["_sample_fn"] = None
        state["_logprob_fn"] = None
        state["flow"] = None
        state["_last_fit_at"] = -10**9  # force refit on next call
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-resolve the optional dep at load time.
        self._fit_fn, self._sample_fn, self._logprob_fn = _require_coppuccino()

    def _maybe_fit(self, chain_stats: ChainStats) -> None:
        if self.fixed:
            return  # pre-fitted flow — never refit
        if self.cold_chain_only and chain_stats.chain_index != 0:
            return

        n_avail = min(chain_stats.sample_total, chain_stats.buffer_size)
        if n_avail < self.min_samples:
            return
        if (self._call_count - self._last_fit_at) < self.refit_interval and self.flow is not None:
            return

        buf = np.asarray(chain_stats._buffer[-n_avail:], dtype=np.float64)
        seed = self.rng_seed + self._fit_count
        try:
            self.flow = self._fit_fn(
                buf,
                max_epochs=self.max_epochs,
                prior_bounds=self.prior_bounds,
                rng_seed=seed,
                **self.fit_kwargs,
            )
            self._fit_count += 1
            self._last_fit_at = self._call_count
            logger.info("NF proposal refit #%d on %d samples", self._fit_count, n_avail)
        except Exception as e:
            # If a fit fails we keep the previous flow and skip until the next
            # interval. Common cause: degenerate samples or marginal collapse.
            logger.warning("NF proposal refit failed (%s); keeping previous flow", e)

    def __call__(self, chain_stats: ChainStats):
        if chain_stats.chain_index == 0:
            self._call_count += 1
        self._maybe_fit(chain_stats)

        if self.flow is None:
            # No flow yet — return current sample with qxy=0, a no-op that's
            # always accepted but does not move the chain. Burn-in will fill
            # the buffer; the first fit will succeed once min_samples is hit.
            return chain_stats.current_sample.copy(), 0.0

        # Independence sample from the flow.
        seed = int(chain_stats.rng.integers(0, 2**31 - 1))
        proposal = np.asarray(self._sample_fn(self.flow, n_samples=1, rng_seed=seed))[0]
        proposal = np.asarray(proposal, dtype=np.float64)

        # qxy = log q(current) - log q(proposal)
        x_curr = np.asarray(chain_stats.current_sample, dtype=np.float64)
        batch = np.stack([x_curr, proposal])
        lp = np.asarray(self._logprob_fn(self.flow, batch))
        qxy = float(lp[0] - lp[1])

        # If either density is NaN (e.g. flow extrapolating past its support),
        # force rejection with qxy = -inf so the chain stays at x_curr.
        if not np.isfinite(qxy):
            return x_curr.copy(), -np.inf

        return proposal, qxy
