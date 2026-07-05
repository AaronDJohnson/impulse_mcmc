"""NUTSSampler — No-U-Turn Sampler with Stan-style warmup.

Follows PTSampler patterns for familiarity: constructor sets up state,
sample() runs the loop, load_chain() reads results.
"""

import os
from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

from impulse.nuts.core import NUTSState, nuts_step
from impulse.nuts.mass_matrix import MassMatrix, MassMatrixType
from impulse.nuts.warmup import DualAveraging, WarmupSchedule, find_reasonable_step_size
from impulse.resume import checkpoint_sampler
from impulse.utils import prepare_files


class NUTSSampler:
    """No-U-Turn Sampler with Stan-style three-phase warmup.

    Parameters
    ----------
    ndim : int
        Dimensionality of parameter space.
    logp_and_grad : callable
        Function (x) -> (logp, grad) returning log-probability and gradient.
    num_warmup : int
        Number of warmup iterations.
    mass_matrix_type : str or MassMatrixType
        Type of mass matrix: 'unit', 'diagonal', or 'dense'.
    target_accept : float
        Target acceptance probability for step size adaptation.
    max_tree_depth : int
        Maximum trajectory tree depth.
    initial_step_size : float, optional
        Initial leapfrog step size. If None, found automatically.
    seed : int, optional
        Random seed for reproducibility.
    outdir : str
        Output directory for chain files and checkpoints.
    save_freq : int
        Frequency of disk writes (iterations).
    resume : bool
        Whether to resume from checkpoint.
    save_warmup : bool
        Whether to include warmup samples in saved chain.

    Examples
    --------
    >>> import numpy as np
    >>> from impulse import NUTSSampler, compose_logp_and_grad
    >>>
    >>> def lnlike(x):
    ...     return -0.5 * np.sum(x**2)
    >>> def lnprior(x):
    ...     return 0.0 if np.all(np.abs(x) < 10) else -np.inf
    >>>
    >>> logp_and_grad = compose_logp_and_grad(lnlike, lnprior)
    >>> sampler = NUTSSampler(ndim=2, logp_and_grad=logp_and_grad, seed=42)
    >>> sampler.sample(np.zeros(2), num_iterations=2000)
    >>> chain = sampler.load_chain()
    """

    def __init__(
        self,
        ndim: int,
        logp_and_grad: Callable,
        num_warmup: int = 1000,
        mass_matrix_type: Union[str, MassMatrixType] = "diagonal",
        target_accept: float = 0.8,
        max_tree_depth: int = 10,
        initial_step_size: Optional[float] = None,
        seed: Optional[int] = None,
        outdir: str = "./chains",
        save_freq: int = 1000,
        resume: bool = False,
        save_warmup: bool = False,
    ) -> None:
        self.ndim = ndim
        self.logp_and_grad = logp_and_grad
        self.num_warmup = num_warmup
        self.target_accept = target_accept
        self.max_tree_depth = max_tree_depth
        self.initial_step_size = initial_step_size
        self.outdir = outdir
        self.save_freq = save_freq
        self.resume = resume
        self.save_warmup = save_warmup

        # Parse mass matrix type
        if isinstance(mass_matrix_type, str):
            self.mass_matrix_type = MassMatrixType(mass_matrix_type)
        else:
            self.mass_matrix_type = mass_matrix_type

        # RNG
        seq = np.random.SeedSequence(seed)
        self.rng = np.random.default_rng(seq)

        # State (initialized in sample())
        self.state: Optional[NUTSState] = None
        self._chain_data: Optional[np.ndarray] = None

    def sample(self, initial_position: ArrayLike, num_iterations: int) -> None:
        """Run NUTS sampling with warmup.

        Parameters
        ----------
        initial_position : array_like
            Starting position, shape (ndim,).
        num_iterations : int
            Number of post-warmup sampling iterations.
        """
        position = np.asarray(initial_position, dtype=np.float64)
        if position.ndim != 1 or len(position) != self.ndim:
            raise ValueError(f"initial_position must be 1-D with length {self.ndim}")

        # Evaluate at initial position
        logp, grad = self.logp_and_grad(position)
        if not np.isfinite(logp):
            raise ValueError("Initial position has non-finite log-probability")

        # Initial mass matrix
        mass_matrix = MassMatrix(self.ndim, MassMatrixType.UNIT)

        # Find reasonable step size
        if self.initial_step_size is None:
            step_size = find_reasonable_step_size(
                position, logp, grad, self.logp_and_grad, mass_matrix, self.rng
            )
        else:
            step_size = self.initial_step_size

        self.state = NUTSState(
            position=position,
            logp=logp,
            grad=grad,
            step_size=step_size,
            mass_matrix=mass_matrix,
        )

        # --- Warmup phase ---
        warmup_schedule = WarmupSchedule(
            num_warmup=self.num_warmup,
            ndim=self.ndim,
            mass_matrix_type=self.mass_matrix_type,
            target_accept=self.target_accept,
            initial_step_size=step_size,
        )

        # only populated (and read) when save_warmup is set
        warmup_samples: list[np.ndarray] = []

        for i in tqdm(range(self.num_warmup), desc="Warmup"):
            self.state = nuts_step(
                self.state,
                self.logp_and_grad,
                self.rng,
                max_tree_depth=self.max_tree_depth,
            )

            new_step_size, new_mass_matrix = warmup_schedule.update(
                i, self.state, self.state.mean_accept_prob
            )

            self.state.step_size = new_step_size
            if new_mass_matrix is not None:
                self.state.mass_matrix = new_mass_matrix

            if self.save_warmup:
                warmup_samples.append(self._state_to_row())

        # Finalize step size
        self.state.step_size = warmup_schedule.finalize()

        # --- Sampling phase ---
        # Set up chain storage
        total_rows = num_iterations
        if self.save_warmup:
            total_rows += self.num_warmup

        # Columns: params(ndim) + logp + accepted + tree_depth + divergent + energy_error + step_size + mean_accept_prob
        ncols = self.ndim + 7
        self._chain_data = np.empty((total_rows, ncols))
        write_idx = 0

        if self.save_warmup:
            for row in warmup_samples:
                self._chain_data[write_idx] = row
                write_idx += 1

        # Set up output file
        filepath = os.path.join(self.outdir, "chain_nuts.txt")
        prepare_files([filepath], resume=self.resume)

        # Write warmup samples if saved
        if self.save_warmup and len(warmup_samples) > 0:
            with open(filepath, "a") as fp:
                np.savetxt(fp, self._chain_data[:write_idx], fmt="%.18e")

        checkpoint_path = os.path.join(self.outdir, "sampler_checkpoint.pkl")
        unsaved = 0

        for i in tqdm(range(num_iterations), desc="Sampling"):
            self.state = nuts_step(
                self.state,
                self.logp_and_grad,
                self.rng,
                max_tree_depth=self.max_tree_depth,
            )

            self._chain_data[write_idx] = self._state_to_row()
            write_idx += 1
            unsaved += 1

            if unsaved >= self.save_freq:
                self._flush_to_disk(filepath, write_idx - unsaved, write_idx)
                checkpoint_sampler(self, path=checkpoint_path, omit=("logp_and_grad",))
                unsaved = 0

        # Final flush
        if unsaved > 0:
            self._flush_to_disk(filepath, write_idx - unsaved, write_idx)

        self._total_iterations = num_iterations
        self._warmup_saved = self.save_warmup

    def _state_to_row(self) -> NDArray[np.float64]:
        """Convert current state to a chain row."""
        s = self.state
        assert s is not None  # only called from sample() after state is set
        return np.concatenate(
            [
                s.position,
                [
                    s.logp,
                    float(s.accepted),
                    s.tree_depth,
                    float(s.divergent),
                    s.energy_error,
                    s.step_size,
                    s.mean_accept_prob,
                ],
            ]
        )

    def _flush_to_disk(self, filepath: str, start: int, end: int) -> None:
        """Write rows [start, end) to disk."""
        assert self._chain_data is not None  # allocated in sample()
        with open(filepath, "a") as fp:
            np.savetxt(fp, self._chain_data[start:end], fmt="%.18e")

    def load_chain(self) -> dict:
        """Load saved chain from disk.

        Returns
        -------
        dict
            Dictionary with keys: samples, logp, accepted, tree_depth,
            divergent, energy_error, step_size, mean_accept_prob.
        """
        filepath = os.path.join(self.outdir, "chain_nuts.txt")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Chain file not found: {filepath}")

        data = np.loadtxt(filepath)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        return {
            "samples": data[:, : self.ndim],
            "logp": data[:, self.ndim],
            "accepted": data[:, self.ndim + 1].astype(bool),
            "tree_depth": data[:, self.ndim + 2].astype(int),
            "divergent": data[:, self.ndim + 3].astype(bool),
            "energy_error": data[:, self.ndim + 4],
            "step_size": data[:, self.ndim + 5],
            "mean_accept_prob": data[:, self.ndim + 6],
        }

    def get_diagnostics(self) -> dict:
        """Summary diagnostics from the sampling run.

        Returns
        -------
        dict
            Diagnostic summary including divergence count, max-depth hits,
            mean tree depth, mean acceptance probability, final step size,
            and mass matrix type.
        """
        if self._chain_data is None:
            raise RuntimeError("No sampling data available. Run sample() first.")
        assert self.state is not None  # set alongside _chain_data in sample()

        # Only look at post-warmup samples
        if self._warmup_saved:
            data = self._chain_data[self.num_warmup :]
        else:
            data = self._chain_data

        # Filter out unwritten rows (zeros)
        mask = data[:, self.ndim] != 0  # logp column
        data = data[mask]

        if len(data) == 0:
            return {}

        divergent = data[:, self.ndim + 3].astype(bool)
        tree_depth = data[:, self.ndim + 2].astype(int)
        accept_prob = data[:, self.ndim + 6]

        return {
            "num_divergent": int(np.sum(divergent)),
            "num_max_depth": int(np.sum(tree_depth >= self.max_tree_depth)),
            "mean_tree_depth": float(np.mean(tree_depth)),
            "mean_accept_prob": float(np.mean(accept_prob)),
            "final_step_size": float(self.state.step_size),
            "mass_matrix_type": self.mass_matrix_type.value,
        }
