"""Standardized wrapper around user likelihood and prior callables.

:class:`_function_wrapper` gives every user function a single batched
interface: it binds extra ``args``/``kwargs``, evaluates non-vectorized
functions row by row over ``(n, ndim)`` inputs (allocating float64 output so
``-inf`` rows cannot overflow integer returns), passes batches straight
through when ``vectorized=True``, and validates output shapes. The
``jax=True`` mode keeps the batch shape constant on every call (invalid rows
are masked after the fact) so JIT-compiled likelihoods never recompile, and
an optional thread pool parallelizes row-by-row evaluation.
"""

from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor


class _function_wrapper(object):
    """
    Wrapper for user-defined functions to handle extra args/kwargs and vectorization.

    This class provides a standardized interface for likelihood and prior functions
    in MCMC sampling, supporting both vectorized and non-vectorized implementations.
    It ensures proper handling of batch evaluations and validates output dimensions.

    Parameters
    ----------
    f : callable
        The function to wrap. Should accept parameter arrays and return log-likelihood
        or log-prior values.
    args : tuple, optional
        Additional positional arguments to pass to the function.
    kwargs : dict, optional
        Additional keyword arguments to pass to the function.
    vectorized : bool, default False
        If True, the function is expected to handle batch inputs directly.
        If False, the function will be called row-by-row for batch inputs.
    jax : bool, default False
        If True, callers may assume the function is JAX-traced/JIT-compiled and
        therefore sensitive to input-shape changes. The MH step uses this to
        avoid masking out rows before the likelihood call (which would otherwise
        trigger costly JIT recompilation each time a different number of
        proposals lands outside the prior).

        Note that this does *not* skip work for out-of-prior rows: the function
        is evaluated on the full batch every iteration and the invalid rows are
        masked to ``-inf`` afterwards. This trades a small amount of wasted
        per-row likelihood compute for keeping the JIT cache warm. If your
        prior-rejection rate is high and each row is expensive, a non-JIT'd
        vectorized NumPy implementation (``jax=False``, ``vectorized=True``)
        will actually skip work on invalid rows and may be faster overall.
    zero_copy : bool, default True
        If True, avoids unnecessary array copies for better performance in
        multicore environments. Uses array views when possible.

    Attributes
    ----------
    f : callable
        The wrapped function.
    args : tuple
        Additional positional arguments.
    kwargs : dict
        Additional keyword arguments.
    vectorized : bool
        Whether the function supports vectorized evaluation.
    zero_copy : bool
        Whether to use zero-copy array operations when possible.

    Examples
    --------
    >>> import numpy as np
    >>> def log_prior(x):
    ...     return 0.0 if np.all(x >= 0) and np.all(x <= 1) else -np.inf
    >>> wrapper = _function_wrapper(log_prior)
    >>> x = np.array([[0.5, 0.3], [0.8, 0.9]])
    >>> result = wrapper(x)
    >>> print(result.shape)
    (2,)

    >>> # Vectorized function example
    >>> def vectorized_log_likelihood(x):
    ...     return -0.5 * np.sum(x**2, axis=1)
    >>> wrapper = _function_wrapper(vectorized_log_likelihood, vectorized=True)
    >>> result = wrapper(x)
    >>> print(result.shape)
    (2,)

    Notes
    -----
    - Input must be 2-D with shape (n_samples, n_parameters)
    - For vectorized=False, function is called n_samples times
    - For vectorized=True, function is called once with full batch
    - Output validation ensures correct leading dimension
    """

    def __init__(
        self,
        f: Callable,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        *,
        vectorized: bool = False,
        jax: bool = False,
        zero_copy: bool = True,
        threads: int = 1,
    ):
        self.f = f
        self.args = tuple(args) if args else ()
        self.kwargs = dict(kwargs) if kwargs else {}
        self.vectorized = bool(vectorized)
        self.jax = bool(jax)
        self.zero_copy = bool(zero_copy)
        self.threads = int(threads)
        self._executor: Optional["ThreadPoolExecutor"] = None  # lazily created

    def __call__(self, x: Any):
        # Use asanyarray to preserve views when possible, or asarray for copies
        if self.zero_copy:
            x_arr = np.asanyarray(x)
        else:
            x_arr = np.asarray(x)

        # check sample shape
        if x_arr.ndim != 2:
            raise ValueError("Input to _function_wrapper must be 2-D")

        # batch (expected shape (n, ...))
        n = x_arr.shape[0]
        if n == 0:
            raise ValueError("Input to _function_wrapper must have nonzero leading dimension")

        # Ensure contiguous array for multicore efficiency
        if self.zero_copy and not x_arr.flags.c_contiguous:
            # Only copy if not contiguous and zero_copy is enabled
            x_arr = np.ascontiguousarray(x_arr)

        if self.vectorized:
            out = self.f(x_arr, *self.args, **self.kwargs)
            # Use asanyarray to preserve views when possible
            if self.zero_copy:
                out_arr = np.asanyarray(out)
            else:
                out_arr = np.asarray(out)
            # check for correct leading dimension
            if out_arr.shape[0] != n:
                raise ValueError(
                    "Vectorized function returned array with incorrect leading dimension"
                )
            return out_arr

        # non-vectorized: threaded path when threads > 1 and multiple rows
        if self.threads > 1 and n > 1:
            return self._threaded_eval(x_arr, n)

        # non-vectorized: per-row calls
        if self.zero_copy:
            # Pre-allocate result array for zero-copy efficiency
            # First call to determine output shape and dtype
            first_result = self.f(x_arr[0], *self.args, **self.kwargs)
            first_arr = np.asanyarray(first_result)

            # Pre-allocate output array. Promote to at least float64: log-densities
            # are floats by contract, and an int/bool dtype from the first row
            # (e.g. a prior returning 0 in-bounds) cannot hold -inf from later rows.
            out_dtype = np.result_type(first_arr.dtype, np.float64)
            if first_arr.ndim == 0:
                # Scalar output
                results = np.empty(n, dtype=out_dtype)
                results[0] = first_arr
                for i in range(1, n):
                    results[i] = self.f(x_arr[i], *self.args, **self.kwargs)
            else:
                # Array output
                results = np.empty((n,) + first_arr.shape, dtype=out_dtype)
                results[0] = first_arr
                for i in range(1, n):
                    results[i] = self.f(x_arr[i], *self.args, **self.kwargs)
            return results
        else:
            # Original behavior: collect results in list then convert
            raw_results = [self.f(x_arr[i], *self.args, **self.kwargs) for i in range(n)]
            return np.asarray(raw_results)

    def _get_executor(self):
        if self._executor is None:
            from concurrent.futures import ThreadPoolExecutor

            self._executor = ThreadPoolExecutor(max_workers=self.threads)
        return self._executor

    def _threaded_eval(self, x_arr, n):
        executor = self._get_executor()
        f, args, kwargs = self.f, self.args, self.kwargs
        futures = [executor.submit(f, x_arr[i], *args, **kwargs) for i in range(n)]
        results = np.empty(n, dtype=np.float64)
        for i, fut in enumerate(futures):
            results[i] = fut.result()
        return results

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_executor"] = None
        return state

    def __del__(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False)

    def __repr__(self):
        name = getattr(self.f, "__name__", repr(self.f))
        return f"<_function_wrapper {name} vectorized={self.vectorized} jax={self.jax} zero_copy={self.zero_copy}>"
