import os
import pickle
import pathlib
import tempfile
from typing import Iterable, Optional, Callable, Any


def checkpoint_sampler(sampler: Any,
                        path: Optional[str] = None,
                        omit: Iterable[str] = ("lnlike", "lnprior")) -> str:
    """
    Atomically pickle `sampler` while temporarily setting attributes named in `omit` to None.
    Restores the attributes on the in-memory sampler after writing.

    Returns the path written.
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

def load_checkpoint(path: str,
                    lnlike: Callable,
                    lnprior: Callable):
    """
    Load a sampler pickled with checkpoint_sampler_strip and rebind lnlike/lnprior.

    - lnlike_factory / lnprior_factory: callables that return the rebuilt functions (or objects)
      to attach to sampler.lnlike / sampler.lnprior. If None, attribute left as None.
    """
    with open(path, "rb") as fp:
        sampler = pickle.load(fp)
    sampler.lnlike = lnlike
    sampler.lnprior = lnprior

    return sampler

def check_for_checkpoint(outdir: str) -> Optional[str]:
    """
    Check for existence of a checkpoint file in `outdir`.
    Returns the path if found, else None.
    """
    path = os.path.join(outdir, "sampler_checkpoint.pkl")
    if os.path.exists(path):
        return path
    return None
