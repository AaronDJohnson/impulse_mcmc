import numpy as np
from pathlib import Path

def prepare_files(filepaths, resume=False):
    """
    Ensure each file in filepaths exists.
    If resume is False, overwrite existing files.
    """
    for filepath in filepaths:
        path = Path(filepath)

        if path.exists():
            if not resume:
                path.unlink()  # remove existing file
                path.touch()   # create a fresh empty file
            # else: do nothing, keep the existing file
        else:
            path.parent.mkdir(parents=True, exist_ok=True)  # ensure directories exist
            path.touch()   # create new file

def shift_array(arr: np.ndarray,
                num: int,
                fill_value: float = 0
                ) -> np.ndarray:
    """
    Shift an array (arr) by to the left (negative num) or the right (positive num)
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
