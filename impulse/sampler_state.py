from dataclasses import dataclass

import numpy as np


def tempered_lnprobs(lnlikes: np.ndarray, lnpriors: np.ndarray, temps: np.ndarray) -> np.ndarray:
    """
    Compute tempered log-posteriors ``beta * lnlike + lnprior`` with ``beta = 1/T``.

    Handles infinite temperatures explicitly: at ``T = inf`` the chain
    samples the prior, so the result is exactly ``lnprior``.  A naive
    ``1/T * lnlike`` would produce ``0 * (-inf) = NaN`` for out-of-support
    likelihoods on the infinite-temperature chain.

    Parameters
    ----------
    lnlikes : np.ndarray
        Log-likelihood values, shape (ntemps,).
    lnpriors : np.ndarray
        Log-prior values, shape (ntemps,).
    temps : np.ndarray
        Temperature values, shape (ntemps,). May contain ``np.inf``.

    Returns
    -------
    np.ndarray
        Tempered log-posteriors, shape (ntemps,). Rows with ``-inf``
        log-prior remain ``-inf``; results are bit-identical to
        ``1/temps * lnlikes + lnpriors`` for finite temperatures.
    """
    with np.errstate(invalid="ignore"):
        beta = np.where(np.isinf(temps), 0.0, 1.0 / temps)
        # the beta == 0 branch of beta * lnlikes still evaluates eagerly
        # (hence the errstate guard); np.where discards its NaNs
        return np.where(beta == 0.0, lnpriors, beta * lnlikes + lnpriors)


@dataclass
class SamplerState:
    """
    Complete state information for all temperature chains in parallel tempering.

    Stores positions, log-probabilities, and metadata for all chains in a
    vectorized format to enable efficient batch operations.

    Parameters
    ----------
    positions : np.ndarray, shape (ntemps, ndim)
        Current parameter positions for each temperature chain.
    lnlikes : np.ndarray, shape (ntemps,)
        Log-likelihood values at current positions.
    lnpriors : np.ndarray, shape (ntemps,)
        Log-prior values at current positions.
    lnprobs : np.ndarray, shape (ntemps,)
        Log-posterior values (tempered): lnprob = lnprior + lnlike / temp.
    accepted : np.ndarray, shape (ntemps,)
        Binary indicators of proposal acceptance in last step (0 or 1).
    temps : np.ndarray, shape (ntemps,)
        Temperature values for each chain.

    Attributes
    ----------
    ntemps : int
        Number of temperature chains.
    ndim : int
        Dimensionality of parameter space.

    Examples
    --------
    >>> import numpy as np
    >>> positions = np.random.randn(5, 3)  # 5 chains, 3 parameters
    >>> lnlikes = np.random.randn(5)
    >>> lnpriors = np.zeros(5)
    >>> temps = np.array([1.0, 1.5, 2.0, 3.0, 4.0])
    >>> lnprobs = lnpriors + lnlikes / temps
    >>> accepted = np.ones(5, dtype=int)
    >>>
    >>> state = SamplerState(positions, lnlikes, lnpriors, lnprobs, accepted, temps)
    >>> print(f"Number of chains: {state.ntemps}")
    >>> print(f"Parameter dimensions: {state.ndim}")

    Notes
    -----
    - All arrays must have consistent leading dimensions (ntemps)
    - Temperatures should be ordered from cold (1.0) to hot (high values)
    - Log-probabilities are tempered by division: β = 1/T
    """

    positions: np.ndarray  # shape (ntemps, ndim)
    lnlikes: np.ndarray  # shape (ntemps,)
    lnpriors: np.ndarray  # shape (ntemps,)
    lnprobs: np.ndarray  # shape (ntemps,)
    accepted: np.ndarray  # shape (ntemps,)
    temps: np.ndarray  # shape (ntemps,)

    @property
    def ntemps(self) -> int:
        """Number of temperature chains."""
        return int(self.positions.shape[0])

    @property
    def ndim(self) -> int:
        """Dimensionality of parameter space."""
        return int(self.positions.shape[1])


@dataclass
class PTState:
    """
    State management for parallel tempering temperature ladder and swap statistics.

    Handles temperature ladder construction, adaptation, and tracking of
    swap acceptance rates for monitoring parallel tempering efficiency.

    Parameters
    ----------
    ndim : int
        Dimensionality of parameter space.
    ntemps : int
        Number of temperature chains.
    swap_steps : int, default 1
        Frequency of swap attempts between temperature chains.
    min_temp : float, default 1.0
        Minimum (coldest) temperature, typically 1.0 for the target distribution.
    max_temp : float, optional
        Maximum (hottest) temperature. If None, determined automatically.
    temp_step : float, optional
        Geometric spacing parameter for temperature ladder. If None, computed automatically.
    nswaps : int, default 1
        Total number of swap attempts (initialized to 1 to avoid division by zero).
    ladder : np.ndarray, optional
        Custom temperature ladder. If None, constructed automatically.
    inf_temp : bool, default False
        Whether to include an infinite temperature chain for improved mixing.
    adapt_t0 : float, default 100
        Initial adaptation period before temperature adaptation begins.
    adapt_nu : float, default 10
        Frequency of temperature ladder adaptation.

    Attributes
    ----------
    swap_accept : np.ndarray
        Number of accepted swaps between each adjacent pair of temperatures.

    Examples
    --------
    >>> ptstate = PTState(ndim=3, ntemps=10, max_temp=100.0)
    >>> print(f"Temperature ladder: {ptstate.ladder}")
    >>> print(f"Swap acceptance rates: {ptstate.swap_accept / ptstate.nswaps}")

    Notes
    -----
    - Temperature ladder is automatically constructed to target ~20% swap rates
    - Infinite temperature chains sample from the prior distribution
    - Adaptation helps optimize swap acceptance rates during burn-in
    """

    ndim: int
    ntemps: int
    swap_steps: int = 1
    min_temp: float = 1.0
    max_temp: float | None = None
    temp_step: float | None = None
    nswaps: int = 1  # start at 1 to avoid divide by zero errors
    ladder: np.ndarray | None = None
    inf_temp: bool = False
    # adaptive temperature ladder parameters:
    adapt_t0: float = 100
    adapt_nu: float = 10

    def __post_init__(self):
        if self.ladder is None:
            self.ladder = self.compute_temp_ladder()
        self.swap_accept = np.zeros(self.ntemps - 1)  # swap acceptance between chains

    def compute_accept_ratio(self):
        """
        Compute current swap acceptance rates between adjacent temperature chains.

        Returns
        -------
        np.ndarray
            Acceptance rates for each adjacent pair of temperatures, shape (ntemps-1,).
            Element i represents swap rate between temperatures i and i+1.

        Examples
        --------
        >>> rates = ptstate.compute_accept_ratio()
        >>> print(f"Average swap rate: {rates.mean():.3f}")
        >>> print(f"Rates by temperature pair: {rates}")

        Notes
        -----
        Target acceptance rates are typically 20-40% for efficient parallel tempering.
        """
        return self.swap_accept / self.nswaps

    def compute_temp_ladder(self):
        """
        Compute geometrically spaced temperature ladder for parallel tempering.

        Creates a temperature sequence designed to achieve ~25% swap acceptance
        rates on multivariate Gaussian distributions. Supports optional infinite
        temperature chain for enhanced mixing.

        Returns
        -------
        np.ndarray
            Temperature ladder with ntemps elements, geometrically spaced
            from min_temp to max_temp (or determined by temp_step).

        Examples
        --------
        >>> ptstate = PTState(ndim=3, ntemps=5, max_temp=16.0)
        >>> ladder = ptstate.compute_temp_ladder()
        >>> print(f"Temperatures: {ladder}")
        [1.0, 2.0, 4.0, 8.0, 16.0]

        >>> # With infinite temperature
        >>> ptstate_inf = PTState(ndim=3, ntemps=4, inf_temp=True)
        >>> ladder = ptstate_inf.compute_temp_ladder()
        >>> print(f"Last temperature: {ladder[-1]}")
        inf

        Notes
        -----
        - Default spacing: temp_step = 1 + √(2/ndim) for optimal swap rates
        - Infinite temperature chains sample from the prior distribution
        - Temperature spacing affects parallel tempering efficiency
        """
        if self.inf_temp:
            self.ntemps -= 1  # remove top value from ladder
        if self.temp_step is None and self.max_temp is None:
            self.temp_step = 1 + np.sqrt(2 / self.ndim)
        elif self.temp_step is None and self.max_temp is not None:
            if self.ntemps > 1:
                self.temp_step = np.exp(np.log(self.max_temp / self.min_temp) / (self.ntemps - 1))
            else:
                self.temp_step = 1.0
        temp_idxs = np.arange(self.ntemps)

        if self.temp_step is None:
            raise ValueError("temp_step is not initialized")

        if self.inf_temp:
            self.ntemps += 1  # add empty top value back to ladder
            ladder = self.min_temp * self.temp_step**temp_idxs  # compute ladder
            ladder = np.concatenate([ladder, [np.inf]])  # add inf value as top of ladder
        else:
            ladder = self.min_temp * self.temp_step**temp_idxs
        return ladder

    def adapt_ladder(self):
        """
        Adapt temperature ladder to optimize swap acceptance rates.

        Implements the adaptive temperature scheme from Vousden et al. (2016)
        arXiv:1501.05823 to automatically tune the temperature ladder during
        sampling for improved parallel tempering efficiency.

        The adaptation uses a hyperbolic decay schedule and targets uniform
        swap acceptance rates across all temperature pairs.

        Examples
        --------
        >>> # Typically called periodically during sampling
        >>> if iteration % adapt_nu == 0:
        ...     ptstate.adapt_ladder()
        >>> print(f"Updated ladder: {ptstate.ladder}")

        Notes
        -----
        - Adaptation strength decreases over time with hyperbolic decay
        - Should be called regularly but not too frequently (every ~10 steps)
        - Helps maintain efficient swapping as sampling progresses
        - Based on Vousden et al. (2016) adaptive parallel tempering algorithm
        """
        if self.ladder is None:
            raise ValueError("PTState ladder is not initialized")

        # Temperature adjustments with a hyperbolic decay.
        decay = self.adapt_t0 / (self.nswaps + self.adapt_t0)  # t0 / (t + t0)
        kappa = decay / self.adapt_nu  # 1 / nu
        # Construct temperature adjustments.
        accept_ratio = self.compute_accept_ratio()
        dscaled_accept = kappa * (
            accept_ratio[:-1] - accept_ratio[1:]
        )  # delta acceptance ratios for chains
        # Compute new ladder (hottest and coldest chains don't move).
        delta_temps = np.diff(self.ladder[:-1])
        delta_temps *= np.exp(dscaled_accept)
        self.ladder[1:-1] = np.cumsum(delta_temps) + self.ladder[0]
