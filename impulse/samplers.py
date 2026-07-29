"""The main parallel-tempering sampler.

:class:`PTSampler` runs adaptive parallel-tempering MCMC: a weighted mixture
of AM/SCAM/DE (and user-registered) proposals per temperature chain,
neighbour swaps with automatic temperature-ladder adaptation, periodic
chain saving, and bit-exact checkpoint/resume via the no-code-execution
``.npz`` + ``.json`` format.

This module carries no model-selection knowledge. To wire a ``PTSampler`` for
product-space (birth-death) model selection, use the experimental
:func:`impulse.experimental.make_product_space_sampler`.

Module-level helpers — :func:`setup_seeds`,
:func:`setup_chain_stats`, :func:`setup_standard_jumps`, and
:func:`setup_initial_position` — build the per-chain RNGs, statistics,
proposal mixtures, and initial positions, and are shared with
:class:`impulse.experimental.hybrid_sampler.HybridPTSampler`.

The engine shared with :class:`~impulse.experimental.hybrid_sampler.HybridPTSampler` lives in
the internal :mod:`impulse._pt_base` module; the setup helpers are defined
there and re-exported here to keep their historical public import paths.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

from impulse._pt_base import (  # noqa: F401  (setup_* re-exported public API)
    _UNSET,
    _PTSamplerBase,
    setup_chain_stats,
    setup_initial_position,
    setup_seeds,
    setup_standard_jumps,
)
from impulse.resume import load_checkpoint


class PTSampler(_PTSamplerBase):
    """
    Parallel Tempering Markov Chain Monte Carlo sampler.

    A sophisticated MCMC sampler that uses parallel tempering to improve mixing
    and exploration of complex posterior distributions. Supports adaptive proposals,
    checkpoint/resume functionality, and vectorized likelihood evaluations.

    Parameters
    ----------
    ndim : int
        Dimensionality of the parameter space.
    lnlike : callable
        Log-likelihood function that accepts parameter arrays.
    lnprior : callable
        Log-prior function that accepts parameter arrays.
    buffer_size : int, default 2000
        Size of internal buffer for storing samples and computing statistics.
    sample_mean : np.ndarray, optional
        Initial estimate of parameter means for adaptive proposals.
    sample_cov : np.ndarray, optional
        Initial covariance matrix estimate for adaptive proposals.
    groups : list, optional
        Parameter groups for block updates. If None, treats all parameters as one group.
    loglargs : tuple, optional
        Additional positional arguments for likelihood function.
    loglkwargs : dict, optional
        Additional keyword arguments for likelihood function.
    logpargs : tuple, optional
        Additional positional arguments for prior function.
    logpkwargs : dict, optional
        Additional keyword arguments for prior function.
    cov_update : int, default 100
        Frequency of covariance matrix updates (in iterations).
    save_freq : int, default 1000
        Frequency of saving chains to disk (in iterations).
    scam_weight : float, default 30
        Relative weight for single-component adaptive Metropolis proposals.
    am_weight : float, default 15
        Relative weight for adaptive Metropolis proposals.
    de_weight : float, default 50
        Relative weight for differential evolution proposals.
    de_min_fill : int, default 100
        Minimum number of history-buffer samples before the differential
        evolution move activates; below the threshold ``de`` returns the
        current position unchanged. See :func:`impulse.proposals.de`.
    seed : int, optional
        Random seed for reproducible sampling.
    outdir : str, default './chains'
        Directory for saving chain files and checkpoints.
    ntemps : int, default 21
        Number of temperature chains.
    swap_steps : int, default 1
        Frequency of temperature swap attempts.
    min_temp : float, default 1.0
        Minimum (cold) temperature.
    max_temp : float, optional
        Maximum (hot) temperature. If None, determined automatically.
    temp_step : float, optional
        Temperature spacing parameter. If None, determined automatically.
    ladder : np.ndarray, optional
        Custom temperature ladder. Overrides automatic temperature selection.
    inf_temp : bool, default False
        Whether to include an infinite temperature chain.
    adapt_t0 : int, default 100
        Initial adaptation period for temperature ladder.
    adapt_nu : int, default 10
        Adaptation frequency for temperature ladder.
    resume : bool, default False
        Whether to resume from existing checkpoint.
    vectorized : bool, default False
        Whether likelihood and prior functions support vectorized evaluation.
    jax : bool, default False
        Set True when the likelihood is JAX-traced/JIT-compiled. The MH step
        will then evaluate the likelihood on the full proposal batch on every
        iteration (masking invalid rows afterwards) so the input shape stays
        constant and the JIT cache is reused instead of recompiling.

        This does *not* skip computation for rows that fall outside the prior
        — the likelihood is still computed for every row in the batch, and the
        invalid rows are zeroed out only after the call. Use this flag when
        the cost of JAX recompilation dominates the cost of evaluating a few
        extra rows (almost always true for JIT'd likelihoods). When the
        likelihood is plain vectorized NumPy and the prior-rejection rate is
        high, leave ``jax=False`` so the step can genuinely skip invalid
        rows.
    num_adapt : int, optional
        Number of iterations during which adaptation is allowed. Once the
        global iteration counter (which persists across checkpoint resume)
        reaches ``num_adapt``, all adaptation freezes: the covariance/mean/SVD
        recomputes feeding the AM/SCAM proposals, the DE sample buffer,
        temperature-ladder adaptation, and refits of adaptive custom
        proposals (e.g. normalizing flows). The DE buffer is frozen too —
        not just its covariance contribution — because a rolling buffer
        would keep the kernel history-dependent; DE continues proposing
        from the frozen buffer. The transition kernel is therefore fixed
        from iteration ``num_adapt`` on, so later samples are exactly
        Markovian; samples drawn before the freeze are warmup and should
        be discarded for strict asymptotic guarantees. ``None`` adapts
        forever, preserving historical behavior.
    verbose : bool, default True
        Show the tqdm sampling progress bar. Set ``False`` to silence it
        (batch/cluster jobs, nested loops such as SBC, notebooks). Presentation
        only: it does not affect the chain, is never checkpointed, and is not
        verified on resume -- a run resumed with ``verbose=False`` stays quiet
        even if the checkpoint came from a verbose run.

        Resume semantics: when ``num_adapt`` is not passed (the default),
        resuming keeps the checkpointed value — un-freezing on resume by
        default would produce a half-frozen kernel, because proposals
        whose frozen state is pickled (e.g. a frozen normalizing flow)
        stay frozen while everything else adapts again. An explicitly
        passed value — including an explicit ``None`` — overrides the
        checkpointed value, with a warning when they differ. Fresh (non
        -resumed) runs treat the default exactly like ``None``.

    Attributes
    ----------
    state : SamplerState
        Current state of all temperature chains.
    ptstate : PTState
        Parallel tempering specific state (temperature ladder, swap statistics).
    multi_chain_stats : MultiChainStats
        Statistics tracking for adaptive proposals.
    proposal_bundle : ProposalBundle
        Collection of proposal distributions for all chains.

    Examples
    --------
    >>> import numpy as np
    >>> from impulse import PTSampler
    >>>
    >>> # Define a simple 2D Gaussian likelihood
    >>> def log_likelihood(x):
    ...     return -0.5 * np.sum(x**2)
    >>>
    >>> # Uniform prior on [-5, 5]^2
    >>> def log_prior(x):
    ...     return 0.0 if np.all(np.abs(x) <= 5) else -np.inf
    >>>
    >>> # Create sampler
    >>> sampler = PTSampler(ndim=2, lnlike=log_likelihood, lnprior=log_prior,
    ...                    ntemps=10, seed=42)
    >>>
    >>> # Run sampling
    >>> initial_pos = np.array([0.0, 0.0])
    >>> sampler.sample(initial_pos, num_iterations=10000)
    Sampling: 100%|██████████| 10000/10000 [00:45<00:00, 220.11it/s]

    >>> # Access results
    >>> print(f"Final acceptance rate: {sampler.state.accepted.mean():.3f}")
    >>> print(f"Temperature swaps accepted: {sampler.ptstate.swap_accept.sum()}")

    Notes
    -----
    - The sampler automatically saves chains and checkpoints during sampling
    - Temperature ladder adaptation helps optimize parallel tempering efficiency
    - Adaptive proposals improve as the sampler learns the target distribution
    - Vectorized functions can significantly improve performance for expensive likelihoods
    """

    # __init__ is inherited unchanged from _PTSamplerBase (identical
    # signature and behavior to the historical PTSampler constructor).

    _logger = logger

    def _load_checkpoint(self, path: str):
        """Load a PTSampler checkpoint, rebinding the wrapped callables."""
        return load_checkpoint(path, lnlike=self.lnlike, lnprior=self.lnprior)

    def add_custom_jump(self, proposal, weight):
        """
        Add a custom proposal distribution to all temperature chains.

        Parameters
        ----------
        proposal : callable
            Proposal with signature ``proposal(chain_stats: ChainStats) ->
            (new_sample: np.ndarray, qxy: float)``, where ``qxy`` is the
            log proposal-density ratio

                ``qxy = log q(x | y) - log q(y | x)``,

            with ``x`` the CURRENT sample, ``y`` the PROPOSED sample, and
            ``q(a | b)`` the density of proposing ``a`` from ``b``.
            ``qxy`` is ADDED to the log-posterior ratio in the
            Metropolis-Hastings acceptance, so positive ``qxy`` favors
            acceptance. Symmetric proposals (``q(y|x) == q(x|y)``, e.g. a
            Gaussian random walk) must return ``qxy = 0.0``; for an
            asymmetric example (a multiplicative random walk whose ``qxy``
            is the log-Jacobian of the rescaling) see the "Custom
            proposals" section of the README and docs.

            Checkpoints store no code, so the proposal is not serialized.
            To resume, re-register the same proposals in the same order
            with the same weights; the sampler verifies this against the
            checkpoint and raises ``CheckpointMismatchError`` otherwise.
            Callable classes must define a ``__name__`` attribute; it keys
            acceptance-rate reports and is what the resume check matches
            on. A proposal that adapts internal state can persist it by
            implementing ``get_checkpoint_state`` / ``set_checkpoint_state``.
        weight : float
            Relative weight for this proposal type (normalized against all
            registered proposals).

        Examples
        --------
        >>> def custom_proposal(chain_stats):
        ...     x = chain_stats.current_sample.copy()
        ...     x += 0.1 * chain_stats.rng.standard_normal(chain_stats.ndim)
        ...     return x, 0.0  # symmetric proposal => qxy = 0
        >>> sampler.add_custom_jump(custom_proposal, weight=25)
        """
        super().add_custom_jump(proposal, weight)

    def sample(self, initial_position: np.ndarray, num_iterations: int, thin: int = 1):
        """
        Run parallel tempering MCMC sampling.

        Performs the main sampling loop, handling proposal generation, acceptance/rejection,
        temperature swaps, adaptive updates, and periodic saves.

        Parameters
        ----------
        initial_position : array_like
            Starting position(s) for the chains. See setup_initial_position for formats.
        num_iterations : int
            Total number of MCMC iterations to perform.
        thin : int, default 1
            Thinning factor for saved samples. Only every thin-th sample is saved.

        Raises
        ------
        ValueError
            If initial likelihood or prior values are not finite.

        Examples
        --------
        >>> sampler = PTSampler(2, log_likelihood, log_prior)
        >>> sampler.sample([0.0, 0.0], num_iterations=10000)
        >>> # Chains are automatically saved to ./chains/ directory

        Notes
        -----
        - Progress is displayed via tqdm progress bar
        - Checkpoints are saved periodically for resuming interrupted runs
        - Temperature swaps and covariance updates occur at specified intervals
        - All chains are saved to disk at save_freq intervals
        - When ``num_adapt`` is set, all adaptation stops once the global
          iteration counter reaches it; samples before the freeze are warmup
          and should be discarded for strict asymptotic guarantees

        Resume semantics: ``num_iterations`` is a GLOBAL iteration target —
        a resumed run continues from the checkpointed iteration counter up
        to ``num_iterations``, so pass the total, not the increment.  A
        checkpoint is written at the END of every iteration ``jj`` with
        ``jj > 0`` and ``jj % save_freq == 0``, capturing the sampler after
        that iteration fully completed (post PT-swap, post adaptation),
        including every RNG stream.  On resume the chain files are truncated
        back to the checkpointed flushed-row count and all iterations after
        the checkpoint are re-generated bit-identically, so an interrupted
        (or prematurely stopped) run resumed to ``N`` total iterations
        produces chain files identical to a single uninterrupted ``N``
        -iteration run.
        """
        return super().sample(initial_position, num_iterations, thin)

    def proposal_acceptance_rates(self) -> dict:
        """Per-proposal acceptance statistics aggregated across chains.

        Returns
        -------
        dict
            ``{name: {calls, accepts, rate, per_chain: [...]}}``
        """
        return super().proposal_acceptance_rates()

    def chain_acceptance_rates(self) -> dict:
        """Per-chain MH acceptance summary, plus PT swap rates.

        Returns
        -------
        dict
            ``temperatures`` (list of T per chain),
            ``mh`` (list per chain: ``{calls, accepts, rate, per_proposal}``),
            ``pt_swap`` (np.ndarray of length ``ntemps - 1`` with the
            accept rate for each neighbour-pair swap, or empty array if
            ``ntemps == 1``).
        """
        return super().chain_acceptance_rates()

    def save_chain_acceptance_rates(self, path: Optional[str] = None) -> str:
        """Write a JSON snapshot of chain acceptance rates to disk.

        Includes per-chain MH rate, per-proposal × per-chain rate,
        aggregate per-proposal rate, and PT swap acceptance per pair.

        Parameters
        ----------
        path : str, optional
            Output path. Defaults to ``<outdir>/chain_acceptance.json``.

        Returns
        -------
        str
            Resolved path written.
        """
        return super().save_chain_acceptance_rates(path)

    def load_chain(self):
        """
        Load saved chain files from disk.

        Reads the chain files written by the sampler and returns them as
        a dictionary of arrays stacked across temperature chains.

        Returns
        -------
        dict
            Dictionary with the following keys:
            - ``samples`` : np.ndarray, shape (ntemps, nsamples, ndim)
            - ``lnlike`` : np.ndarray, shape (ntemps, nsamples)
            - ``lnprob`` : np.ndarray, shape (ntemps, nsamples)
            - ``accepted`` : np.ndarray, shape (ntemps, nsamples)
            - ``temperature`` : np.ndarray, shape (ntemps, nsamples)

        Raises
        ------
        FileNotFoundError
            If any expected chain file does not exist.

        Examples
        --------
        >>> sampler = PTSampler(ndim=2, lnlike=log_likelihood, lnprior=log_prior)
        >>> sampler.sample([0.0, 0.0], num_iterations=10000)
        >>> chain = sampler.load_chain()
        >>> print(chain['samples'].shape)
        (21, 10000, 2)
        """
        return super().load_chain()
