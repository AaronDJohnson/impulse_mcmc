"""NUTSSampler — No-U-Turn Sampler with Stan-style warmup.

Follows PTSampler patterns for familiarity: constructor sets up state,
sample() runs the loop, load_chain() reads results.
"""

import os
from typing import Any, Callable, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

from impulse.nuts.core import NUTSState, nuts_step
from impulse.nuts.mass_matrix import MassMatrix, MassMatrixType
from impulse.nuts.warmup import DualAveraging, WarmupSchedule, find_reasonable_step_size
from impulse.resume import (
    _jsonable_rng_state,
    _rng_state_from_json,
    check_for_checkpoint,
    checkpoint_sampler,
    restore_state_checkpoint,
)
from impulse.utils import prepare_files

_CHAIN_NAME = "chain_nuts.txt"


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
        Continue a previous run in ``outdir`` from its checkpoint. When a
        checkpoint is present, warmup is **skipped** (the adapted step size and
        mass matrix are restored) and ``num_iterations`` is treated as a
        **global** target: a run checkpointed at iteration 400 resumed with
        ``num_iterations=1000`` performs 600 more. With no checkpoint present
        the run simply starts fresh, so the same script works for a first
        submission and every requeue. See :meth:`sample`.
    save_warmup : bool
        Whether to include warmup samples in saved chain.
    verbose : bool, default True
        Show tqdm progress bars for the warmup and sampling phases. Set
        ``False`` to silence them (batch jobs, nested SBC loops, notebooks).
        Presentation only -- it does not affect the chain, and is neither
        checkpointed nor verified on resume.

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
        verbose: bool = True,
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
        # Presentation only: not checkpointed, not verified on resume.
        self.verbose = verbose

        # Parse mass matrix type
        if isinstance(mass_matrix_type, str):
            self.mass_matrix_type = MassMatrixType(mass_matrix_type)
        else:
            self.mass_matrix_type = mass_matrix_type

        # RNG
        seq = np.random.SeedSequence(seed)
        self.rng = np.random.default_rng(seq)

        # State (initialized in sample(), or restored from a checkpoint)
        self.state: Optional[NUTSState] = None
        self._chain_data: Optional[np.ndarray] = None
        # Post-warmup sampling iterations completed, and rows flushed to the
        # chain file. Both are checkpointed: the first makes num_iterations a
        # global target across a resume, the second lets the resumed run
        # truncate rows written after the last checkpoint before regenerating
        # them from the restored RNG stream.
        self._iteration = 0
        self._rows_written = 0
        self._warmup_saved = False
        self._total_iterations = 0
        self._warmup_rows: list[np.ndarray] = []
        self._resume_path: Optional[str] = None
        if resume:
            self._resume_path = check_for_checkpoint(outdir)

    def sample(self, initial_position: ArrayLike, num_iterations: int) -> None:
        """Run NUTS sampling with warmup, or continue a checkpointed run.

        Parameters
        ----------
        initial_position : array_like
            Starting position, shape (ndim,). Ignored when resuming from a
            checkpoint (the restored position continues the chain).
        num_iterations : int
            Post-warmup sampling iterations. This is a **global target**, not
            an increment: resuming a run checkpointed at iteration 400 with
            ``num_iterations=1000`` performs 600 more. Passing a target already
            reached is a no-op.

        Notes
        -----
        Resuming is **bit-exact**: an interrupted run resumed to ``N`` total
        iterations produces a chain file identical to a single uninterrupted
        ``N``-iteration run (``tests/test_reproducibility.py``). Warmup runs
        once, in the original run; the resumed run restores the adapted step
        size and mass matrix rather than re-adapting them.
        """
        filepath = os.path.join(self.outdir, _CHAIN_NAME)
        resumed = self._maybe_restore(filepath)

        if not resumed:
            self._initialize_and_warm_up(initial_position)
        assert self.state is not None  # set by whichever branch ran

        warmup_rows = self._warmup_rows if not resumed else []

        remaining = num_iterations - self._iteration
        if remaining <= 0:
            # Target already met by the checkpointed run: nothing to add, and
            # the chain file already holds those rows.
            self._total_iterations = self._iteration
            return

        # Columns: params(ndim) + logp + accepted + tree_depth + divergent
        #          + energy_error + step_size + mean_accept_prob
        ncols = self.ndim + 7
        self._chain_data = np.empty((remaining + len(warmup_rows), ncols))
        write_idx = 0
        for row in warmup_rows:
            self._chain_data[write_idx] = row
            write_idx += 1

        # A fresh run overwrites the chain file; a resumed one keeps it and is
        # truncated to the checkpointed row count by _maybe_restore above.
        prepare_files([filepath], resume=resumed)

        if warmup_rows:
            self._flush_to_disk(filepath, 0, write_idx)

        unsaved = 0
        for _ in tqdm(range(remaining), desc="Sampling", disable=not self.verbose):
            self.state = nuts_step(
                self.state,
                self.logp_and_grad,
                self.rng,
                max_tree_depth=self.max_tree_depth,
            )

            self._chain_data[write_idx] = self._state_to_row()
            write_idx += 1
            unsaved += 1
            self._iteration += 1

            if unsaved >= self.save_freq:
                self._flush_to_disk(filepath, write_idx - unsaved, write_idx)
                # Written at the END of the iteration, after the row is on
                # disk, so the restored RNG stream resumes exactly here.
                checkpoint_sampler(self)
                unsaved = 0

        # Final flush
        if unsaved > 0:
            self._flush_to_disk(filepath, write_idx - unsaved, write_idx)

        self._total_iterations = self._iteration
        self._warmup_saved = bool(warmup_rows)

    def _initialize_and_warm_up(self, initial_position: ArrayLike) -> None:
        """Evaluate the start point, find a step size, and run the warmup phase."""
        position = np.asarray(initial_position, dtype=np.float64)
        if position.ndim != 1 or len(position) != self.ndim:
            raise ValueError(f"initial_position must be 1-D with length {self.ndim}")

        logp, grad = self.logp_and_grad(position)
        if not np.isfinite(logp):
            raise ValueError("Initial position has non-finite log-probability")

        mass_matrix = MassMatrix(self.ndim, MassMatrixType.UNIT)

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

        warmup_schedule = WarmupSchedule(
            num_warmup=self.num_warmup,
            ndim=self.ndim,
            mass_matrix_type=self.mass_matrix_type,
            target_accept=self.target_accept,
            initial_step_size=step_size,
        )

        # only populated (and read) when save_warmup is set
        self._warmup_rows = []

        for i in tqdm(range(self.num_warmup), desc="Warmup", disable=not self.verbose):
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
                self._warmup_rows.append(self._state_to_row())

        # Freeze the smoothed dual-averaging step size, not the noisy iterate
        self.state.step_size = warmup_schedule.finalize()

    def _maybe_restore(self, filepath: str) -> bool:
        """Restore from a discovered checkpoint; return True if one was applied.

        Raises when ``resume=True`` finds chain rows but no checkpoint to
        continue from: appending to that file would splice a fresh,
        re-warmed-up run onto the old one, silently mixing two chains.
        """
        if self._resume_path is None:
            if self.resume and os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                raise RuntimeError(
                    f"resume=True but no usable checkpoint was found in {self.outdir!r}, "
                    f"while {_CHAIN_NAME} already holds samples. Continuing would append a "
                    "fresh, re-warmed-up run to the existing chain, mixing two runs in one "
                    "file. Either restore the checkpoint, or start a fresh run "
                    "(resume=False, or a new outdir)."
                )
            return False

        if self._resume_path.endswith(".pkl"):
            raise RuntimeError(
                f"found a legacy pickle checkpoint ({self._resume_path}) in {self.outdir!r}. "
                "NUTSSampler pickle checkpoints predate resume support -- they were written "
                "but never read back, so they do not record the iteration count or row "
                "count needed to continue a run. Start a fresh run (resume=False, or a new "
                "outdir); load_nuts_checkpoint can still restore the object for inspection."
            )

        restore_state_checkpoint(self, self._resume_path)
        self._truncate_chain_to_saved(filepath)
        return True

    def _truncate_chain_to_saved(self, filepath: str) -> None:
        """Drop chain rows written after the checkpoint was taken.

        Rows beyond ``_rows_written`` were flushed after the last checkpoint
        (by the final flush of a run that finished, or by one killed between a
        flush and the next checkpoint). The resumed run regenerates exactly
        those iterations from the restored RNG stream, so leaving them would
        duplicate rows.
        """
        if not os.path.exists(filepath):
            return
        with open(filepath, "r") as fp:
            lines = fp.readlines()
        if len(lines) > self._rows_written:
            with open(filepath, "w") as fp:
                fp.writelines(lines[: self._rows_written])

    def _capture_checkpoint_state(self) -> tuple[dict, dict]:
        """Serialize state for the no-code-execution (``.npz`` + JSON) format.

        Presence of this hook is what makes :func:`checkpoint_sampler` choose
        the safe format over a pickle for this sampler.
        """
        s = self.state
        assert s is not None  # only called from sample() after state is set
        mass_arrays, mass_meta = s.mass_matrix.get_checkpoint_state()
        arrays = {
            "position": np.asarray(s.position, dtype=float),
            "grad": np.asarray(s.grad, dtype=float),
            **{f"mass_{key}": value for key, value in mass_arrays.items()},
        }
        meta = {
            "sampler_class": type(self).__name__,
            # Run-shaping scalars, verified on restore (see
            # impulse.resume._verify_checkpoint_metadata).
            "ndim": int(self.ndim),
            "save_freq": int(self.save_freq),
            "num_warmup": int(self.num_warmup),
            "max_tree_depth": int(self.max_tree_depth),
            "save_warmup": int(bool(self.save_warmup)),
            "target_accept": float(self.target_accept),
            "mass_matrix_type": self.mass_matrix_type.value,
            # Progress counters.
            "iteration": int(self._iteration),
            "rows_written": int(self._rows_written),
            # Exact RNG stream position, so the resumed run continues it.
            "rng": _jsonable_rng_state(self.rng.bit_generator.state),
            "mass_matrix": mass_meta,
            "state": {
                "logp": float(s.logp),
                "step_size": float(s.step_size),
                "iteration": int(s.iteration),
                "accepted": bool(s.accepted),
                "divergent": bool(s.divergent),
                "tree_depth": int(s.tree_depth),
                "energy_error": float(s.energy_error),
                "mean_accept_prob": float(s.mean_accept_prob),
            },
        }
        return arrays, meta

    def _restore_checkpoint_state(self, arrays: dict, meta: dict[str, Any]) -> None:
        """Restore :meth:`_capture_checkpoint_state` output into this sampler."""
        mass_arrays = {
            key[len("mass_") :]: value for key, value in arrays.items() if key.startswith("mass_")
        }
        st = meta["state"]
        self.state = NUTSState(
            position=np.array(arrays["position"], dtype=float),
            logp=float(st["logp"]),
            grad=np.array(arrays["grad"], dtype=float),
            step_size=float(st["step_size"]),
            mass_matrix=MassMatrix.from_checkpoint_state(mass_arrays, meta["mass_matrix"]),
            iteration=int(st["iteration"]),
            accepted=bool(st["accepted"]),
            divergent=bool(st["divergent"]),
            tree_depth=int(st["tree_depth"]),
            energy_error=float(st["energy_error"]),
            mean_accept_prob=float(st["mean_accept_prob"]),
        )
        self.rng.bit_generator.state = _rng_state_from_json(meta["rng"])
        self._iteration = int(meta["iteration"])
        self._rows_written = int(meta["rows_written"])

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
        """Append rows [start, end) to the chain file and count them.

        ``_rows_written`` is what a resumed run truncates the file back to, so
        it must advance in lockstep with the appends -- never be recomputed
        from the file length, which would also count rows a later resume is
        supposed to discard.
        """
        assert self._chain_data is not None  # allocated in sample()
        with open(filepath, "a") as fp:
            np.savetxt(fp, self._chain_data[start:end], fmt="%.18e")
        self._rows_written += end - start

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
