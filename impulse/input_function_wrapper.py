import numpy as np
from functools import partial
from typing import Any, Optional, Callable

class _function_wrapper(object):
    """
    Wrapper for user-defined functions to handle extra args/kwargs and vectorization.

    Behavior:
      - 2-D input (n, ...) -> treated as a batch:
          * if self.vectorized == True: call f(batch) and validate leading dim == n
          * else: call f(row) per-row and return array of results
    """
    def __init__(self,
                 f: Callable,
                 args: Optional[tuple] = None,
                 kwargs: Optional[dict] = None,
                 *,
                 vectorized: bool = False):
        self.f = f
        self.args = tuple(args) if args else ()
        self.kwargs = dict(kwargs) if kwargs else {}
        self.vectorized = bool(vectorized)

    def __call__(self, x: Any):
        x_arr = np.asarray(x)

        # check sample shape
        if x_arr.ndim != 2:
            raise ValueError("Input to _function_wrapper must be 2-D")

        # batch (expected shape (n, ...))
        n = x_arr.shape[0]
        if n == 0:
            raise ValueError("Input to _function_wrapper must have nonzero leading dimension")

        if self.vectorized:
            out = self.f(x_arr, *self.args, **self.kwargs)
            out_arr = np.asarray(out)
            # check for correct leading dimension
            if out_arr.shape[0] != n:
                raise ValueError("Vectorized function returned array with incorrect leading dimension")
            return out_arr

        # non-vectorized: per-row calls
        func = partial(self.f, *self.args, **self.kwargs)
        results = [func(x_arr[i]) for i in range(n)]
        return np.asarray(results)

    def __repr__(self):
        name = getattr(self.f, "__name__", repr(self.f))
        return f"<_function_wrapper {name} vectorized={self.vectorized}>"
