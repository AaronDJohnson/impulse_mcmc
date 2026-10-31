"""Chain I/O, file preparation, and in-memory ring-buffer storage."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

FORMATS = ("binary", "text")
SUFFIXES = {"binary": ".bin", "text": ".txt"}
ITEMSIZE = 8
_TEXT_FMT = "%.18e"
_BUFFER_FIELDS = ("samples", "lnprob", "lnlike", "accept", "var_temp")


def validate_format(chain_format: str) -> str:
    """Return ``chain_format`` if supported ('binary' or 'text'), else raise ValueError."""
    if chain_format not in FORMATS:
        raise ValueError(
            f"chain_format must be one of {FORMATS}, got {chain_format!r}. "
            "'binary' writes raw float64 records (fast, compact); 'text' "
            "writes %.18e columns (human-readable, historical default)."
        )
    return chain_format


def chain_suffix(chain_format: str) -> str:
    """Filename suffix for an encoding (``.bin`` or ``.txt``)."""
    return SUFFIXES[validate_format(chain_format)]


def detect_format(basepath: str) -> Optional[str]:
    """Identify the encoding of an existing chain file given its stem."""
    for fmt in FORMATS:
        if os.path.exists(basepath + SUFFIXES[fmt]):
            return fmt
    return None


def append_rows(filepath: str, rows: np.ndarray, chain_format: str) -> int:
    """Append ``rows`` (shape ``(n, ncols)``) to ``filepath``; return ``n``."""
    validate_format(chain_format)
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.shape[0] == 0:
        return 0
    if chain_format == "binary":
        with open(filepath, "ab") as fp:
            fp.write(np.ascontiguousarray(rows).tobytes())
    else:
        with open(filepath, "a", encoding="ascii") as fp:
            np.savetxt(fp, rows, fmt=_TEXT_FMT, delimiter=" ")
    return int(rows.shape[0])


def read_rows(filepath: str, ncols: int, chain_format: Optional[str] = None) -> np.ndarray:
    """Read a whole chain file as a ``(nrows, ncols)`` float64 array."""
    if chain_format is None:
        chain_format = "binary" if filepath.endswith(SUFFIXES["binary"]) else "text"
    validate_format(chain_format)
    if chain_format == "binary":
        flat = np.fromfile(filepath, dtype=np.float64)
        nrows = flat.size // ncols
        return flat[: nrows * ncols].reshape(nrows, ncols)
    data = np.loadtxt(filepath)
    if data.size == 0:
        return np.empty((0, ncols))
    return data.reshape(1, -1) if data.ndim == 1 else data


def count_rows(filepath: str, ncols: int, chain_format: str) -> int:
    """Number of complete rows currently in ``filepath`` (0 if absent)."""
    validate_format(chain_format)
    if not os.path.exists(filepath):
        return 0
    if chain_format == "binary":
        return os.path.getsize(filepath) // (ncols * ITEMSIZE)
    with open(filepath, "r", encoding="ascii") as fp:
        return sum(1 for _ in fp)


def truncate_rows(filepath: str, nrows: int, ncols: int, chain_format: str) -> None:
    """Shorten ``filepath`` to its first ``nrows`` rows; no-op if already shorter."""
    validate_format(chain_format)
    if not os.path.exists(filepath):
        return
    if chain_format == "binary":
        target = nrows * ncols * ITEMSIZE
        if os.path.getsize(filepath) > target:
            os.truncate(filepath, target)
        return
    with open(filepath, "r", encoding="ascii") as fp:
        lines = fp.readlines()
    if len(lines) > nrows:
        with open(filepath, "w", encoding="ascii") as fp:
            fp.writelines(lines[:nrows])


def prepare_files(filepaths, resume: bool = False) -> None:
    """Ensure each file in ``filepaths`` exists, overwriting unless ``resume=True``."""
    for filepath in filepaths:
        path = Path(filepath)
        if path.exists():
            if not resume:
                path.unlink()
                path.touch()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()


def shift_array(arr: np.ndarray, num: int, fill_value: float = 0) -> np.ndarray:
    """Shift an array along axis 0 by ``num`` positions, filling vacated entries."""
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


@dataclass
class ShortChain:
    """Circular buffer for in-memory chain storage and periodic disk flushes."""

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
        suffix = chain_suffix(self.chain_format)
        self.filenames = [f"chain_{i}{suffix}" for i in range(self.ntemps)]
        self.filepaths = [os.path.join(self.outdir, name) for name in self.filenames]
        prepare_files(self.filepaths, resume=self.resume)

    @property
    def ncols(self) -> int:
        return self.ndim + 4

    def add_state(self, new_state) -> None:
        """Record a ``SamplerState`` at the current ring position."""
        idx = self.iteration % self.short_iters
        self.samples[:, idx] = new_state.positions
        self.lnprob[:, idx] = new_state.lnprobs
        self.lnlike[:, idx] = new_state.lnlikes
        self.accept[:, idx] = new_state.accepted
        self.var_temp[:, idx] = new_state.temps
        self.iteration += 1
        self._unsaved += 1

    def exists(self, outdir: str, filename: str) -> bool:
        return Path(os.path.join(outdir, filename)).exists()

    def get_recent_samples(self, count: int) -> np.ndarray:
        """Return the ``count`` most recent samples as ``(ntemps, count, ndim)``."""
        count = min(count, self.short_iters, self.iteration)
        if count == 0:
            return self.samples[:, :0, :]
        end = self.iteration % self.short_iters
        start = (self.iteration - count) % self.short_iters
        if start < end:
            return self.samples[:, start:end, :]
        return np.concatenate([self.samples[:, start:, :], self.samples[:, :end, :]], axis=1)

    def save_chain(self) -> None:
        """Flush unsaved samples to disk with global-phase thinning."""
        if self._unsaved == 0:
            return
        count = min(self._unsaved, self.short_iters)
        end = self.iteration % self.short_iters
        start = (self.iteration - count) % self.short_iters
        idx = slice(start, end) if start < end else np.r_[start : self.short_iters, 0:end]

        offset = (-(self.iteration - count)) % self.thin
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
            nrows = append_rows(filepath, to_save, self.chain_format)
        self._unsaved = 0
        self._rows_written += nrows

    def get_checkpoint_state(self) -> tuple[dict, dict]:
        arrays = {k: np.asarray(getattr(self, k)) for k in _BUFFER_FIELDS}
        meta = {
            "iteration": int(self.iteration),
            "unsaved": int(self._unsaved),
            "rows_written": int(self._rows_written),
            "short_iters": int(self.short_iters),
            "thin": int(self.thin),
        }
        return arrays, meta

    def set_checkpoint_state(self, arrays: dict, meta: dict) -> None:
        self.short_iters = int(meta["short_iters"])
        self.thin = int(meta["thin"])
        self.iteration = int(meta["iteration"])
        self._unsaved = int(meta["unsaved"])
        self._rows_written = int(meta["rows_written"])
        for k in _BUFFER_FIELDS:
            setattr(self, k, np.array(arrays[k], dtype=float))

    def truncate_files_to_saved(self) -> None:
        """Truncate chain files on disk to the checkpointed row count."""
        for filepath in self.filepaths:
            truncate_rows(filepath, self._rows_written, self.ncols, self.chain_format)
