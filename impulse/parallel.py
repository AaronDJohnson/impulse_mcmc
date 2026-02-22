"""
Parallel likelihood evaluation with shared memory for zero-copy performance.

This module provides efficient parallel evaluation of likelihood functions
using shared memory arrays to minimize inter-process communication overhead.
Particularly beneficial for fast likelihood functions where IPC dominates
computation time.
"""

import os
import numpy as np
from multiprocessing import shared_memory, Pool, cpu_count
from typing import Callable, Optional, Union
import warnings
import atexit


class ParallelLikelihood:
    """
    Zero-copy parallel likelihood evaluation for fast likelihood functions.

    Uses shared memory arrays to avoid pickle serialization overhead when
    distributing likelihood evaluations across multiple processes. Most
    beneficial when likelihood evaluation time is comparable to or less
    than inter-process communication time.

    Parameters
    ----------
    likelihood_fn : callable
        Likelihood function that accepts parameter arrays and returns
        log-likelihood values. Must be pickleable for multiprocessing.
    n_workers : int, optional
        Number of worker processes. Defaults to CPU count.
    batch_size : int, default 1000
        Minimum batch size for parallel processing. Smaller batches
        are evaluated directly without parallelization overhead.
    max_batch_size : int, optional
        Maximum batch size for shared memory allocation. Defaults to
        10 times batch_size to handle larger evaluations efficiently.

    Attributes
    ----------
    likelihood_fn : callable
        The wrapped likelihood function.
    n_workers : int
        Number of worker processes.
    batch_size : int
        Minimum batch size for parallel processing.
    max_batch_size : int
        Maximum batch size for shared memory.

    Examples
    --------
    >>> import numpy as np
    >>> def simple_likelihood(params):
    ...     return -0.5 * np.sum(params**2, axis=1)
    >>>
    >>> parallel_like = ParallelLikelihood(simple_likelihood, n_workers=4)
    >>> params = np.random.randn(5000, 10)  # 5000 samples, 10 parameters
    >>> results = parallel_like(params)
    >>> print(results.shape)
    (5000,)

    Notes
    -----
    - Shared memory is allocated once and reused across evaluations
    - Cleanup is handled automatically via atexit handlers
    - For small batches (< batch_size), falls back to direct evaluation
    - Memory usage scales with max_batch_size, not total sample count
    """

    def __init__(self,
                 likelihood_fn: Callable,
                 n_workers: Optional[int] = None,
                 batch_size: int = 1000,
                 max_batch_size: Optional[int] = None):

        self.likelihood_fn = likelihood_fn
        self.n_workers = n_workers or cpu_count()
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size or (10 * batch_size)

        # Initialize shared memory components
        self._param_shm = None
        self._param_array = None
        self._pool = None
        self._param_dim = None

        # Register cleanup
        atexit.register(self.cleanup)

    def _setup_shared_memory(self, param_dim: int):
        """Initialize shared memory arrays for given parameter dimension."""
        if self._param_dim == param_dim and self._param_shm is not None:
            return  # Already set up for this dimension

        # Clean up existing shared memory
        self._cleanup_shared_memory()

        self._param_dim = param_dim

        # Create shared memory block for parameters
        param_size = self.max_batch_size * param_dim * 8  # float64

        try:
            self._param_shm = shared_memory.SharedMemory(create=True, size=param_size)

            # Create NumPy array view (zero-copy)
            self._param_array = np.ndarray(
                (self.max_batch_size, param_dim),
                dtype=np.float64,
                buffer=self._param_shm.buf
            )

        except Exception as e:
            self._cleanup_shared_memory()
            raise RuntimeError(f"Failed to create shared memory: {e}")

    def _setup_worker_pool(self):
        """Initialize worker pool with shared memory names."""
        if self._pool is not None:
            return

        if self._param_shm is None:
            raise RuntimeError("Shared memory must be set up before worker pool")

        try:
            # Pass shared memory names to workers
            init_args = (
                self._param_shm.name,
                self.max_batch_size,
                self._param_dim,
                self.likelihood_fn
            )

            self._pool = Pool(
                processes=self.n_workers,
                initializer=_worker_init,
                initargs=init_args
            )
        except Exception as e:
            warnings.warn(f"Failed to create worker pool: {e}. Using direct evaluation.")
            self._pool = None

    def __call__(self, params: np.ndarray) -> np.ndarray:
        """
        Evaluate likelihood function in parallel.

        Parameters
        ----------
        params : np.ndarray
            Parameter array of shape (n_samples, n_params).

        Returns
        -------
        np.ndarray
            Log-likelihood values of shape (n_samples,).
        """
        params = np.asarray(params)

        if params.ndim != 2:
            raise ValueError("params must be 2-D array")

        n_samples, param_dim = params.shape

        if n_samples == 0:
            return np.array([])

        # For small batches, evaluate directly
        if n_samples < self.batch_size:
            return self.likelihood_fn(params)

        # Set up shared memory for this parameter dimension
        try:
            self._setup_shared_memory(param_dim)
            self._setup_worker_pool()
        except Exception as e:
            warnings.warn(f"Parallel setup failed, using direct evaluation: {e}")
            return self.likelihood_fn(params)

        # If pool setup failed, use direct evaluation
        if self._pool is None:
            return self.likelihood_fn(params)

        # Process in chunks
        results = np.empty(n_samples, dtype=np.float64)

        for start_idx in range(0, n_samples, self.max_batch_size):
            end_idx = min(start_idx + self.max_batch_size, n_samples)
            chunk_size = end_idx - start_idx
            chunk = params[start_idx:end_idx]

            try:
                # Copy chunk to shared memory (one-time cost per chunk)
                self._param_array[:chunk_size] = chunk

                # Determine work distribution
                work_per_proc = max(1, chunk_size // self.n_workers)
                work_items = []

                for i in range(0, chunk_size, work_per_proc):
                    work_end = min(i + work_per_proc, chunk_size)
                    work_items.append((i, work_end))

                # Parallel evaluation (zero IPC cost after setup)
                chunk_results = self._pool.map(_worker_process, work_items)

                # Collect results
                for (work_start, work_end), worker_results in zip(work_items, chunk_results):
                    results[start_idx + work_start:start_idx + work_end] = worker_results

            except Exception as e:
                # Fallback to direct evaluation
                warnings.warn(f"Parallel evaluation failed, using direct evaluation: {e}")
                results[start_idx:end_idx] = self.likelihood_fn(chunk)

        return results

    def _cleanup_shared_memory(self):
        """Clean up shared memory resources."""
        if self._param_shm is not None:
            try:
                self._param_shm.close()
                self._param_shm.unlink()
            except Exception:
                pass  # May already be cleaned up
            finally:
                self._param_shm = None

        self._param_array = None
        self._param_dim = None

    def cleanup(self):
        """Clean up all resources."""
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            except Exception:
                pass
            finally:
                self._pool = None

        self._cleanup_shared_memory()

    def __del__(self):
        """Cleanup on object deletion."""
        self.cleanup()

    def __repr__(self):
        return (f"<ParallelLikelihood n_workers={self.n_workers} "
                f"batch_size={self.batch_size} max_batch_size={self.max_batch_size}>")


# Global worker state
_worker_param_array = None
_worker_likelihood_fn = None


def _worker_init(param_shm_name: str,
                max_batch_size: int,
                param_dim: int,
                likelihood_fn: Callable):
    """Initialize worker process with shared memory access."""
    global _worker_param_array, _worker_likelihood_fn

    try:
        # Connect to existing shared memory
        param_shm = shared_memory.SharedMemory(name=param_shm_name)

        # Create array view
        _worker_param_array = np.ndarray(
            (max_batch_size, param_dim),
            dtype=np.float64,
            buffer=param_shm.buf
        )

        _worker_likelihood_fn = likelihood_fn

    except Exception as e:
        raise RuntimeError(f"Worker initialization failed: {e}")


def _worker_process(work_range: tuple) -> np.ndarray:
    """Process a range of parameters in worker process."""
    global _worker_param_array, _worker_likelihood_fn

    start_idx, end_idx = work_range

    # Extract work chunk from shared memory (zero-copy view)
    work_params = _worker_param_array[start_idx:end_idx]

    # Evaluate likelihood
    results = _worker_likelihood_fn(work_params)

    return results.copy()