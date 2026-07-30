"""On-disk chain record format: raw binary (default) or whitespace text.

Chain files are append-only tables of fixed-width ``float64`` rows. Two
encodings are supported, selected per sampler by the ``chain_format`` argument:

``"binary"`` (default)
    Raw little-endian-native ``float64``, no header, no delimiters --
    ``nrows * ncols * 8`` bytes exactly. Appending is a ``bytes`` write,
    reading is one ``numpy.fromfile``, and truncating to a row count is an
    ``os.truncate`` at a byte offset. This is the default because formatting
    floats as text dominated chain I/O: ``numpy.savetxt`` measured ~18% of
    total wall time on a 21-temperature run, essentially all of it in decimal
    conversion.

``"text"``
    The historical encoding: one row per line, ``%.18e`` fields separated by
    single spaces, readable with ``numpy.loadtxt`` or any text tool. Slower to
    write and ~2.7x larger on disk, but greppable and diffable, which is worth
    having when debugging a run on a cluster.

Both encodings are lossless for finite values: ``%.18e`` round-trips a
``float64`` exactly. They differ for non-finite values -- binary preserves the
exact NaN/inf bit patterns, text writes ``nan``/``inf`` spellings that
``loadtxt`` maps back to canonical values.

Row counts are derived from the file itself (byte length for binary, line count
for text) rather than tracked separately, so a torn write is visible as a short
or partial final row instead of being silently trusted.
"""

import os
from typing import Optional

import numpy as np

#: Encodings understood by every function in this module.
FORMATS = ("binary", "text")

#: Filename suffix per encoding. The suffix IS the format marker on disk, so a
#: reader never has to be told which encoding a directory holds.
SUFFIXES = {"binary": ".bin", "text": ".txt"}

#: Bytes per stored value. The binary encoding is float64 and nothing else --
#: a narrower dtype would silently lose precision the text encoding preserves.
ITEMSIZE = 8

_TEXT_FMT = "%.18e"


def validate_format(chain_format: str) -> str:
    """Return ``chain_format`` if supported, else raise ``ValueError``."""
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
    """Identify the encoding of an existing chain file given its stem.

    Parameters
    ----------
    basepath : str
        Path WITHOUT a suffix, e.g. ``<outdir>/chain_0``.

    Returns
    -------
    str or None
        ``"binary"``, ``"text"``, or ``None`` when neither file exists.
        Binary wins if somehow both are present, matching the write default.
    """
    for fmt in FORMATS:
        if os.path.exists(basepath + SUFFIXES[fmt]):
            return fmt
    return None


def append_rows(filepath: str, rows: np.ndarray, chain_format: str) -> int:
    """Append ``rows`` (shape ``(n, ncols)``) to ``filepath``; return ``n``.

    Writing zero rows is a no-op that still returns 0, so callers can append
    unconditionally.
    """
    validate_format(chain_format)
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.shape[0] == 0:
        return 0
    if chain_format == "binary":
        with open(filepath, "ab") as fp:
            # ascontiguousarray: a sliced/strided view would otherwise be
            # written in memory order rather than row-major order.
            fp.write(np.ascontiguousarray(rows).tobytes())
    else:
        with open(filepath, "a", encoding="ascii") as fp:
            np.savetxt(fp, rows, fmt=_TEXT_FMT, delimiter=" ")
    return int(rows.shape[0])


def read_rows(filepath: str, ncols: int, chain_format: Optional[str] = None) -> np.ndarray:
    """Read a whole chain file as a ``(nrows, ncols)`` float64 array.

    ``chain_format`` defaults to the encoding implied by the file's suffix.
    A trailing partial row (a torn write) is dropped with the rest intact.
    """
    if chain_format is None:
        chain_format = "binary" if filepath.endswith(SUFFIXES["binary"]) else "text"
    validate_format(chain_format)
    if chain_format == "binary":
        flat = np.fromfile(filepath, dtype=np.float64)
        nrows = flat.size // ncols
        if flat.size != nrows * ncols:
            # Torn final row: keep the complete ones.
            flat = flat[: nrows * ncols]
        return flat.reshape(nrows, ncols)
    data = np.loadtxt(filepath)
    if data.size == 0:
        return np.empty((0, ncols))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def count_rows(filepath: str, ncols: int, chain_format: str) -> int:
    """Number of COMPLETE rows currently in ``filepath`` (0 if absent)."""
    validate_format(chain_format)
    if not os.path.exists(filepath):
        return 0
    if chain_format == "binary":
        return os.path.getsize(filepath) // (ncols * ITEMSIZE)
    with open(filepath, "r", encoding="ascii") as fp:
        return sum(1 for _ in fp)


def truncate_rows(filepath: str, nrows: int, ncols: int, chain_format: str) -> None:
    """Shorten ``filepath`` to its first ``nrows`` rows; no-op if already shorter.

    Binary truncates at a byte offset without reading the file. Text must
    rewrite, since line lengths are not fixed.
    """
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
