"""Small shared helpers.

:func:`prepare_files` creates (or, on resume, preserves) the chain output
files and their parent directories before sampling starts, and
:func:`shift_array` rolls entries of the circular sample-history buffers
used by :mod:`impulse.chain_stats`.
"""

from pathlib import Path

import numpy as np


def prepare_files(filepaths, resume=False):
    """
    Ensure each file in filepaths exists.

    If resume is False, overwrite existing files. If resume is True,
    keep existing files intact. Creates parent directories as needed.

    Parameters
    ----------
    filepaths : list of str
        List of file paths to prepare.
    resume : bool, default False
        If True, keep existing files. If False, overwrite existing files.

    Examples
    --------
    >>> prepare_files(['./output/chain_0.txt', './output/chain_1.txt'])
    >>> # Creates files, overwriting if they exist
    >>>
    >>> prepare_files(['./output/chain_0.txt'], resume=True)
    >>> # Creates file only if it doesn't exist, preserves existing content
    """
    for filepath in filepaths:
        path = Path(filepath)

        if path.exists():
            if not resume:
                path.unlink()  # remove existing file
                path.touch()  # create a fresh empty file
            # else: do nothing, keep the existing file
        else:
            path.parent.mkdir(parents=True, exist_ok=True)  # ensure directories exist
            path.touch()  # create new file


def shift_array(arr: np.ndarray, num: int, fill_value: float = 0) -> np.ndarray:
    """
    Shift an array along axis 0 by specified number of positions.

    Positive values shift to the right (elements move towards higher indices),
    negative values shift to the left (elements move towards lower indices).
    Empty positions are filled with fill_value.

    Parameters
    ----------
    arr : np.ndarray
        Input array to shift.
    num : int
        Number of positions to shift. Positive values shift right (towards
        higher indices), negative values shift left (towards lower indices).
    fill_value : float, default 0
        Value to fill empty positions created by the shift.

    Returns
    -------
    np.ndarray
        Shifted array with same shape as input.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> shift_array(arr, 2)  # shift right by 2
    array([0, 0, 1, 2, 3])
    >>> shift_array(arr, -2)  # shift left by 2
    array([3, 4, 5, 0, 0])
    >>> shift_array(arr, 1, fill_value=-1)  # shift with custom fill
    array([-1,  1,  2,  3,  4])
    """
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
