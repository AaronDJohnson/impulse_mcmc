"""Chain storage: in-memory buffering and per-temperature chain files.

:class:`ShortChain` is a ring buffer that accumulates :class:`SamplerState`
snapshots between flushes and appends them to one text file per temperature
chain (``chain_<i>.txt``: samples, log-likelihood, log-posterior, acceptance,
temperature) every ``save_freq`` iterations. It also implements thinning,
resume-time appending, and truncation of chain files back to the
checkpointed row count (``truncate_files_to_saved``), which is what makes
interrupted-then-resumed runs bit-identical to uninterrupted ones.
"""

import os
import pathlib
from dataclasses import dataclass

import numpy as np

from impulse import chain_io
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
    chain_format : {'binary', 'text'}, default 'binary'
        On-disk record encoding; see :mod:`impulse.chain_io`. ``'binary'``
        writes raw float64 to ``chain_<i>.bin``; ``'text'`` writes ``%.18e``
        columns to ``chain_<i>.txt``.

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
    outdir: str = "./chains/"
    resume: bool = False
    thin: int = 1
    chain_format: str = "binary"

    def __post_init__(self):
        if self.thin > self.short_iters:
            raise ValueError("There are not enough samples to thin. Increase save_freq.")
        self.samples = np.zeros((self.ntemps, self.short_iters, self.ndim))
        self.lnprob = np.zeros((self.ntemps, self.short_iters))
        self.lnlike = np.zeros((self.ntemps, self.short_iters))
        self.accept = np.zeros((self.ntemps, self.short_iters))
        self.var_temp = np.zeros((self.ntemps, self.short_iters))
        self._unsaved = 0
        self._rows_written = 0
        chain_io.validate_format(self.chain_format)
        suffix = chain_io.chain_suffix(self.chain_format)
        self.filenames = [f"chain_{nchain}{suffix}" for nchain in range(self.ntemps)]
        self.filepaths = [os.path.join(self.outdir, filename) for filename in self.filenames]
        prepare_files(self.filepaths, resume=self.resume)

    @property
    def ncols(self) -> int:
        """Values per stored row: the parameters plus four bookkeeping columns.

        The trailing four are lnlike, lnprob, accepted, temperature -- see
        :meth:`save_chain`. Binary row length depends on this, so it is derived
        from ``ndim`` rather than stored, and cannot drift out of sync.
        """
        return self.ndim + 4

    def add_state(self, new_state: SamplerState):
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

    def _ensure_rows_written(self):
        """Lazily initialize ``_rows_written`` for legacy unpickled instances.

        ShortChain instances unpickled from checkpoints written before row
        tracking existed lack ``_rows_written``.  The counter must NEVER
        restart at 0 in that case: the chain files already hold many rows,
        and an undercount pickled into the next checkpoint would make the
        following resume's :meth:`truncate_files_to_saved` rewrite the files
        as a tiny prefix, destroying history.  Instead the counter is
        re-seeded from the CURRENT on-disk line count of the chain files.

        The per-temperature files are flushed in lockstep, so their counts
        only differ after a torn (partially completed) flush; the minimum is
        used so a later truncation drops the torn tail rather than trusting
        it.  Called from every reader/writer of ``_rows_written`` so the
        value is correct regardless of whether :meth:`save_chain` or
        :meth:`truncate_files_to_saved` runs first after unpickling.
        """
        if hasattr(self, "_rows_written"):
            return
        counts = [
            chain_io.count_rows(filepath, self.ncols, self.chain_format)
            for filepath in self.filepaths
        ]
        self._rows_written = min(counts) if counts else 0

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
        # Seed the row counter from disk BEFORE appending, so instances
        # unpickled from pre-row-tracking checkpoints count on from the
        # true file length instead of restarting at 0.
        self._ensure_rows_written()
        count = min(self._unsaved, self.short_iters)
        end = self.iteration % self.short_iters
        start = (self.iteration - count) % self.short_iters

        if start < end:
            idx = slice(start, end)
        else:
            # Wraps around (or full buffer when start == end)
            idx = np.r_[start : self.short_iters, 0:end]

        # Thin on the GLOBAL iteration index, not per flush block.
        #
        # Slicing each block with [::thin] restarts the thinning phase at every
        # save_freq boundary, so the saved chain has non-uniform spacing that
        # depends on save_freq: with thin=3, save_freq=10 the kept iterations
        # were 0,3,6,9,10,13,16,19,20,... (gaps 3,3,3,1,3,3,3,1,...) instead of
        # 0,3,6,9,12,... That silently violates the documented "only every
        # thin-th sample is saved" contract and puts a periodic artifact into
        # any autocorrelation or ESS estimate computed from the file.
        #
        # first_global is the first iteration in this block that survives
        # thinning; offset positions it within the block. thin == 1 (the
        # default) always gives offset 0, so behavior -- and bit-exact
        # reproducibility -- is unchanged for every existing run.
        first_global = self.iteration - count
        offset = (-first_global) % self.thin

        nrows = 0
        for temp_idx, filepath in enumerate(self.filepaths):
            to_save = np.column_stack(
                [
                    self.samples[temp_idx, idx],
                    self.lnlike[temp_idx, idx],
                    self.lnprob[temp_idx, idx],
                    self.accept[temp_idx, idx],
                    self.var_temp[temp_idx, idx],
                ]
            )[offset :: self.thin]
            nrows = chain_io.append_rows(filepath, to_save, self.chain_format)
        self._unsaved = 0
        self._rows_written += nrows

    def get_checkpoint_state(self) -> tuple[dict, dict]:
        """Serialize the ring buffer and counters to ``(arrays, meta)``.

        The buffer arrays hold the unflushed samples (needed so the next
        flush and covariance update see identical data on resume); the
        counters (``iteration``, ``_unsaved``, ``_rows_written``) drive the
        resume-time truncation and bit-exact continuation.
        """
        arrays = {
            "samples": np.asarray(self.samples),
            "lnprob": np.asarray(self.lnprob),
            "lnlike": np.asarray(self.lnlike),
            "accept": np.asarray(self.accept),
            "var_temp": np.asarray(self.var_temp),
        }
        self._ensure_rows_written()
        meta = {
            "iteration": int(self.iteration),
            "unsaved": int(self._unsaved),
            "rows_written": int(self._rows_written),
            "short_iters": int(self.short_iters),
            "thin": int(self.thin),
        }
        return arrays, meta

    def set_checkpoint_state(self, arrays: dict, meta: dict) -> None:
        """Restore ring buffer and counters from :meth:`get_checkpoint_state`.

        ``short_iters``/``thin`` are restored alongside the buffers so the ring
        buffer stays self-consistent with the checkpointed data (mirroring the
        legacy pickle, which restored the whole object).
        """
        self.short_iters = int(meta["short_iters"])
        self.thin = int(meta["thin"])
        self.iteration = int(meta["iteration"])
        self._unsaved = int(meta["unsaved"])
        self._rows_written = int(meta["rows_written"])
        self.samples = np.array(arrays["samples"], dtype=float)
        self.lnprob = np.array(arrays["lnprob"], dtype=float)
        self.lnlike = np.array(arrays["lnlike"], dtype=float)
        self.accept = np.array(arrays["accept"], dtype=float)
        self.var_temp = np.array(arrays["var_temp"], dtype=float)

    def truncate_files_to_saved(self):
        """
        Truncate the on-disk chain files to the flushed-row count.

        Called when resuming from a checkpoint.  The checkpoint pickles this
        object with ``_rows_written`` — the number of (thinned) rows this
        buffer had flushed to each chain file when the checkpoint was taken.
        Any rows beyond that were written AFTER the checkpoint (e.g. by the
        final flush of a run that completed normally, or by a run killed
        between a flush and the next checkpoint).  The resumed run re-generates
        those iterations deterministically from the checkpointed RNG streams,
        so the stale rows must be dropped first or they would be duplicated.

        Instances restored from checkpoints that predate row tracking have no
        ``_rows_written``; for those the counter is re-seeded from the current
        on-disk line count (see :meth:`_ensure_rows_written`), so this call is
        a no-op — the historical append-only behavior — and subsequent flushes
        count on from the true file length.

        Examples
        --------
        >>> # after unpickling a checkpointed ShortChain on resume:
        >>> chain.truncate_files_to_saved()
        >>> # chain files now end exactly at the checkpointed row count
        """
        self._ensure_rows_written()
        for filepath in self.filepaths:
            chain_io.truncate_rows(filepath, self._rows_written, self.ncols, self.chain_format)
