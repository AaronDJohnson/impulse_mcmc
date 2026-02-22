from dataclasses import dataclass
import numpy as np
import pathlib
import os
from impulse.sampler_state import SamplerState
from impulse.utils import prepare_files

@dataclass
class ShortChain:
    """
    Circular buffer for efficient MCMC chain storage and periodic file output.

    Manages temporary storage of MCMC samples in memory before writing to disk
    at regular intervals. Uses a ring buffer design to minimize memory usage
    while ensuring all samples are preserved.

    Parameters
    ----------
    ndim : int
        Dimensionality of parameter space.
    ntemps : int
        Number of temperature chains.
    short_iters : int
        Size of circular buffer (number of iterations to store).
    iteration : int, default 0
        Current global iteration counter.
    outdir : str, default './chains/'
        Directory path for output files.
    resume : bool, default False
        Whether to append to existing files (True) or overwrite (False).
    thin : int, default 1
        Thinning factor - only every thin-th sample is saved to disk.

    Attributes
    ----------
    samples : np.ndarray, shape (ntemps, short_iters, ndim)
        Parameter positions buffer.
    lnprob : np.ndarray, shape (ntemps, short_iters)
        Log-posterior values buffer.
    lnlike : np.ndarray, shape (ntemps, short_iters)
        Log-likelihood values buffer.
    accept : np.ndarray, shape (ntemps, short_iters)  
        Acceptance indicators buffer.
    var_temp : np.ndarray, shape (ntemps, short_iters)
        Temperature values buffer.

    Examples
    --------
    >>> chain = ShortChain(ndim=3, ntemps=5, short_iters=1000)
    >>> for i in range(10000):
    ...     chain.add_state(sampler_state)
    ...     if i % 1000 == 0:
    ...         chain.save_chain()  # Periodic save to disk

    Notes
    -----
    - Automatically creates output directory structure
    - Files are named 'chain_{i}.txt' for temperature index i
    - Supports both new runs and resuming from checkpoints
    - Thinning is applied during save_chain(), not during storage
    """
    ndim: int
    ntemps: int
    short_iters: int
    iteration: int = 0
    outdir: str = './chains/'
    resume: bool = False
    thin: int = 1

    def __post_init__(self):
        if self.thin > self.short_iters:
            raise ValueError("There are not enough samples to thin. Increase save_freq.")
        self.samples = np.zeros((self.ntemps, self.short_iters, self.ndim))
        self.lnprob = np.zeros((self.ntemps, self.short_iters))
        self.lnlike = np.zeros((self.ntemps, self.short_iters))
        self.accept = np.zeros((self.ntemps, self.short_iters))
        self.var_temp = np.zeros((self.ntemps, self.short_iters))
        self._unsaved = 0
        self.filenames = [f'chain_{nchain}.txt' for nchain in range(self.ntemps)]
        self.filepaths = [os.path.join(self.outdir, filename) for filename in self.filenames]
        prepare_files(self.filepaths, resume=self.resume)

    def add_state(self,
                  new_state: SamplerState):
        """
        Add a new sampler state to the circular buffer.

        Parameters
        ----------
        new_state : SamplerState
            Complete state information from current MCMC iteration.

        Examples
        --------
        >>> state = SamplerState(positions, lnlikes, lnpriors, lnprobs, accepted, temps)
        >>> chain.add_state(state)
        >>> print(f"Stored iteration {chain.iteration}")
        """
        save_iter = self.iteration % self.short_iters
        self.samples[:, save_iter] = new_state.positions
        self.lnprob[:, save_iter] = new_state.lnprobs
        self.lnlike[:, save_iter] = new_state.lnlikes
        self.accept[:, save_iter] = new_state.accepted
        self.var_temp[:, save_iter] = new_state.temps
        self.iteration += 1
        self._unsaved += 1

    def exists(self, outdir, filename):
        """
        Check if a file exists in the specified directory.

        Parameters
        ----------
        outdir : str
            Directory path to check.
        filename : str
            Name of file to check for.

        Returns
        -------
        bool
            True if file exists, False otherwise.

        Examples
        --------
        >>> exists = chain.exists('./output', 'chain_0.txt')
        >>> if exists:
        ...     print("Chain file already exists")
        """
        return pathlib.Path(os.path.join(outdir, filename)).exists()

    def get_recent_samples(self, count: int) -> np.ndarray:
        """
        Get the most recent ``count`` samples from the buffer.

        Parameters
        ----------
        count : int
            Number of recent samples to retrieve.

        Returns
        -------
        np.ndarray, shape (ntemps, count, ndim)
            Most recent samples in chronological order.
        """
        count = min(count, self.short_iters, self.iteration)
        if count == 0:
            return self.samples[:, :0, :]
        end = self.iteration % self.short_iters
        start = (self.iteration - count) % self.short_iters
        if start < end:
            return self.samples[:, start:end, :]
        else:
            # Wraps around (or full buffer when start == end)
            return np.concatenate([self.samples[:, start:, :], self.samples[:, :end, :]], axis=1)

    def save_chain(self):
        """
        Write unsaved samples to disk files with optional thinning.

        Only saves samples added since the last call to save_chain,
        preventing stale data from being written at the end of runs.

        Examples
        --------
        >>> # Save every 1000 iterations during sampling
        >>> if iteration % 1000 == 0:
        ...     chain.save_chain()

        Notes
        -----
        - Files contain: [parameters, log_likelihood, log_posterior, accepted, temperature]
        - Thinning is applied: only every thin-th row is saved
        - Files are opened in append mode for resuming runs
        """
        if self._unsaved == 0:
            return
        count = min(self._unsaved, self.short_iters)
        end = self.iteration % self.short_iters
        start = (self.iteration - count) % self.short_iters

        if start < end:
            idx = slice(start, end)
        else:
            # Wraps around (or full buffer when start == end)
            idx = np.r_[start:self.short_iters, 0:end]

        for temp_idx, filepath in enumerate(self.filepaths):
            to_save = np.column_stack([self.samples[temp_idx, idx], self.lnlike[temp_idx, idx], self.lnprob[temp_idx, idx], self.accept[temp_idx, idx], self.var_temp[temp_idx, idx]])[::self.thin]
            with open(filepath, 'a') as fp:
                np.savetxt(fp, to_save, fmt='%.18e', delimiter=' ')
        self._unsaved = 0
