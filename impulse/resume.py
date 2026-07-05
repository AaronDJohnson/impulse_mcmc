import os
import pathlib
import pickle
import tempfile
from typing import Any, Callable, Iterable, Optional


def checkpoint_sampler(
    sampler: Any, path: Optional[str] = None, omit: Iterable[str] = ("lnlike", "lnprior")
) -> str:
    """
    Create atomic checkpoint of sampler state for resuming interrupted runs.

    Safely serializes the sampler object to disk while temporarily removing
    non-serializable function objects. Uses atomic file operations to prevent
    corruption from interrupted writes.

    Parameters
    ----------
    sampler : Any
        PTSampler instance to checkpoint.
    path : str, optional
        Output file path. If None, uses sampler.outdir/sampler_checkpoint.pkl.
    omit : iterable of str, default ('lnlike', 'lnprior')
        Attribute names to temporarily set to None during pickling.

    Returns
    -------
    str
        Path to created checkpoint file.

    Examples
    --------
    >>> checkpoint_path = checkpoint_sampler(sampler)
    >>> print(f"Checkpoint saved to {checkpoint_path}")
    >>> # Later, resume with: sampler = load_checkpoint(checkpoint_path, lnlike, lnprior)

    Notes
    -----
    - Uses atomic rename to prevent corruption during writes
    - Function objects are temporarily removed as they can't be pickled reliably
    - Original sampler object in memory is restored after checkpointing
    - Creates parent directories if they don't exist
    """
    if path is None:
        path = os.path.join(getattr(sampler, "outdir", "."), "sampler_checkpoint.pkl")
    pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

    # stash values and set to None
    stashed = {}
    for name in omit:
        if hasattr(sampler, name):
            stashed[name] = getattr(sampler, name)
            setattr(sampler, name, None)

    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".ckpt.", suffix=".tmp")
    os.close(tmp_fd)
    try:
        with open(tmp_path, "wb") as fp:
            pickle.dump(sampler, fp, protocol=pickle.HIGHEST_PROTOCOL)
        # atomic rename
        os.replace(tmp_path, path)
    finally:
        # always restore the in-memory sampler attributes
        for name, val in stashed.items():
            setattr(sampler, name, val)
        # remove tmp if still present
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return path


def load_checkpoint(path: str, lnlike: Callable, lnprior: Callable):
    """
    Load sampler from checkpoint and restore function objects.

    Deserializes a pickled sampler and rebinds the likelihood and prior
    functions that were omitted during checkpointing.

    Parameters
    ----------
    path : str
        Path to checkpoint file created by checkpoint_sampler.
    lnlike : callable
        Log-likelihood function to rebind to the sampler.
    lnprior : callable
        Log-prior function to rebind to the sampler.

    Returns
    -------
    PTSampler
        Restored sampler object ready to continue sampling.

    Examples
    --------
    >>> def log_likelihood(x):
    ...     return -0.5 * np.sum(x**2)
    >>> def log_prior(x):
    ...     return 0.0 if np.all(np.abs(x) < 5) else -np.inf
    >>>
    >>> sampler = load_checkpoint('checkpoint.pkl', log_likelihood, log_prior)
    >>> # Continue sampling from where we left off
    >>> sampler.sample(sampler.state.positions[0], num_iterations=5000)

    Notes
    -----
    - Functions must be identical to those used in original run
    - Sampler state, statistics, and random generators are fully restored
    - Can resume sampling immediately after loading
    """
    with open(path, "rb") as fp:
        sampler = pickle.load(fp)
    sampler.lnlike = lnlike
    sampler.lnprior = lnprior

    return sampler


def load_nuts_checkpoint(path: str, logp_and_grad: Callable):
    """Load NUTSSampler from checkpoint and restore gradient function.

    Parameters
    ----------
    path : str
        Path to checkpoint file created by checkpoint_sampler.
    logp_and_grad : callable
        Function (x) -> (logp, grad) to rebind to the sampler.

    Returns
    -------
    NUTSSampler
        Restored sampler ready to continue sampling.

    Examples
    --------
    >>> sampler = load_nuts_checkpoint('checkpoint.pkl', logp_and_grad)
    """
    with open(path, "rb") as fp:
        sampler = pickle.load(fp)
    sampler.logp_and_grad = logp_and_grad
    return sampler


def load_rjpt_checkpoint(
    path: str,
    lnlike: Callable,
    lnprior: Callable,
    raw_lnlike: Optional[Callable] = None,
    raw_lnprior: Optional[Callable] = None,
    lnlike_grad: Optional[Callable] = None,
):
    """Load RJPTSampler from checkpoint and restore callable attributes.

    Parameters
    ----------
    path : str
        Path to checkpoint file.
    lnlike, lnprior : callable
        Wrapped likelihood/prior to rebind.
    raw_lnlike, raw_lnprior : callable, optional
        Unwrapped functions for NUTS gradient building.
    lnlike_grad : callable, optional
        Gradient function for NUTS.

    Returns
    -------
    RJPTSampler
        Restored sampler ready to continue sampling.
    """
    with open(path, "rb") as fp:
        sampler = pickle.load(fp)
    sampler.lnlike = lnlike
    sampler.lnprior = lnprior
    if raw_lnlike is not None:
        sampler._raw_lnlike = raw_lnlike
    if raw_lnprior is not None:
        sampler._raw_lnprior = raw_lnprior
    if lnlike_grad is not None:
        sampler.lnlike_grad = lnlike_grad
    return sampler


def check_for_checkpoint(outdir: str) -> Optional[str]:
    """
    Search for existing checkpoint file in output directory.

    Parameters
    ----------
    outdir : str
        Directory to search for checkpoint files.

    Returns
    -------
    str or None
        Path to checkpoint file if found, None otherwise.

    Examples
    --------
    >>> checkpoint_path = check_for_checkpoint('./chains')
    >>> if checkpoint_path:
    ...     print(f"Found checkpoint: {checkpoint_path}")
    ...     sampler = load_checkpoint(checkpoint_path, lnlike, lnprior)
    >>> else:
    ...     print("No checkpoint found, starting fresh")
    """
    path = os.path.join(outdir, "sampler_checkpoint.pkl")
    if os.path.exists(path):
        return path
    return None
