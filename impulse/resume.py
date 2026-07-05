"""Checkpointing and resume for all samplers.

Two on-disk formats exist:

**New (default) — no code execution on load.** :class:`impulse.PTSampler`
and :class:`impulse.HybridPTSampler` checkpoint to ``sampler_checkpoint.npz``
(array state, via :func:`numpy.savez_compressed`) plus
``sampler_checkpoint.json`` (a schema-versioned metadata sidecar: RNG
bit-generator states, the ordered proposal names/weights, and every
component's scalar state). Loading uses ``numpy.load(..., allow_pickle=
False)`` and ``json.load``, so a tampered checkpoint can NOT execute code —
loading one is as safe as reading a data file. The resume contract is
*reconstruct then restore*: rebuild the sampler exactly as the original run
did (same constructor / ``from_product_space`` / ``add_custom_jump`` calls), then
``resume=True`` (or :func:`restore_state_checkpoint`) verifies the
reconstruction matches the checkpoint metadata and restores STATE into it.
Callables and the product space are never serialized — the reconstruction
supplies them.

**Legacy — pickle.** Older checkpoints are ``sampler_checkpoint.pkl``: a
Python pickle of the whole sampler. **Unpickling can execute arbitrary
code**, so :func:`load_checkpoint`, :func:`load_hybrid_checkpoint`, and
:func:`load_nuts_checkpoint` emit a loud security/deprecation warning; only
resume from pickle checkpoints you trust (see SECURITY.md). This path is
retained unchanged for backward compatibility and is deprecated for removal
in a future 2.x release. :class:`~impulse.nuts.sampler.NUTSSampler` still
checkpoints via pickle (its checkpointing is separate from the PT engine).

:func:`checkpoint_sampler` writes the new format for PT/hybrid samplers and
pickle only for a legacy ``.pkl`` target; :func:`check_for_checkpoint`
locates a checkpoint in an output directory, preferring the new format and
treating a lone ``.npz`` (json sidecar absent — a torn write) as no
checkpoint.
"""

import json
import os
import pathlib
import pickle
import tempfile
import uuid
import warnings
from typing import Any, Callable, Iterable, Optional

import numpy as np

# Bump when the on-disk metadata layout changes incompatibly. The loader
# accepts any version <= this and refuses newer ones with a clear error.
CHECKPOINT_SCHEMA_VERSION = 1

# Basenames (in ``outdir``) for each checkpoint artifact.
_NEW_BASENAME = "sampler_checkpoint"
_NPZ_NAME = _NEW_BASENAME + ".npz"
_JSON_NAME = _NEW_BASENAME + ".json"
_PKL_NAME = _NEW_BASENAME + ".pkl"

_PICKLE_SECURITY_WARNING = (
    "Loading a legacy pickle checkpoint (sampler_checkpoint.pkl). Unpickling "
    "can execute arbitrary code, so only resume from pickle checkpoints you "
    "trust (see SECURITY.md). Pickle checkpoints are deprecated; a run "
    "resumed from a .pkl keeps writing .pkl. To move to the safe .npz/.json "
    "format, start a fresh run (resume=False, or a new outdir)."
)


class CheckpointMismatchError(ValueError):
    """Raised when a reconstructed sampler does not match checkpoint metadata."""


class TornCheckpointError(ValueError):
    """Raised when a checkpoint's .json and .npz do not belong to the same write."""


# ---------------------------------------------------------------------------
# New format: array (.npz) + JSON metadata
# ---------------------------------------------------------------------------


def _checkpoint_base(path: Optional[str], sampler: Any) -> str:
    """Return the extension-less base path for the new-format artifact pair.

    ``None`` -> ``<outdir>/sampler_checkpoint``; a ``.json``/``.npz``/``.pkl``
    path -> its stem; any other path is used verbatim as the base.
    """
    if path is None:
        outdir = getattr(sampler, "outdir", ".")
        return os.path.join(outdir, _NEW_BASENAME)
    for ext in (".json", ".npz", ".pkl"):
        if path.endswith(ext):
            return path[: -len(ext)]
    return path


def save_state_checkpoint(sampler: Any, path: Optional[str] = None) -> str:
    """Atomically write the no-code-execution checkpoint (``.npz`` + ``.json``).

    The array state goes to ``<base>.npz`` (compressed) and the metadata to
    ``<base>.json``. Both are written to temp files and ``os.replace``\\ d into
    place with the JSON sidecar committed LAST, so an interrupted write leaves
    a ``.npz`` with no ``.json`` — which :func:`check_for_checkpoint` treats as
    torn and ignores.

    Parameters
    ----------
    sampler : PTSampler or HybridPTSampler
        Must implement ``_capture_checkpoint_state`` (the PT engine does).
    path : str, optional
        Target path or base; defaults to ``<sampler.outdir>/sampler_checkpoint``.

    Returns
    -------
    str
        Path to the committed JSON sidecar (the commit marker).
    """
    base = _checkpoint_base(path, sampler)
    npz_path = base + ".npz"
    json_path = base + ".json"
    dirn = os.path.dirname(npz_path) or "."
    pathlib.Path(dirn).mkdir(parents=True, exist_ok=True)

    arrays, meta = sampler._capture_checkpoint_state()
    # Per-write generation token stamped into BOTH files. Every save after
    # the first OVERWRITES an existing json+npz pair; a crash between the
    # two os.replace calls below would leave a fresh npz beside a stale
    # json (both present, so the file-existence check alone would treat
    # the mismatched pair as valid). load_state_checkpoint compares the
    # tokens and rejects a mismatch as torn.
    write_token = uuid.uuid4().hex
    meta = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "write_token": write_token,
        **meta,
    }
    arrays = {**arrays, "_write_token": np.array(write_token)}

    # Write the npz to a temp file (open handle so numpy does not append a
    # second ".npz" to the name).
    npz_fd, tmp_npz = tempfile.mkstemp(dir=dirn, prefix=".ckpt.", suffix=".npz.tmp")
    os.close(npz_fd)
    json_fd, tmp_json = tempfile.mkstemp(dir=dirn, prefix=".ckpt.", suffix=".json.tmp")
    os.close(json_fd)
    try:
        with open(tmp_npz, "wb") as fp:
            np.savez_compressed(fp, **arrays)
        with open(tmp_json, "w") as fp:
            json.dump(meta, fp)
        # Commit: npz first, JSON last (the JSON sidecar is the marker).
        os.replace(tmp_npz, npz_path)
        os.replace(tmp_json, json_path)
    finally:
        for tmp in (tmp_npz, tmp_json):
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    return json_path


def load_state_checkpoint(path: str) -> tuple[dict, dict]:
    """Read the new-format checkpoint WITHOUT executing any code.

    Uses ``numpy.load(..., allow_pickle=False)`` and ``json.load``; a
    checkpoint containing a Python object array (the only way an ``.npz`` can
    smuggle code) raises instead of unpickling.

    Parameters
    ----------
    path : str
        Path to the ``.json`` sidecar, the ``.npz`` array file, or the base.

    Returns
    -------
    (arrays, meta) : tuple[dict, dict]
        ``arrays`` maps npz keys to ``np.ndarray``; ``meta`` is the parsed JSON.
    """
    base = path
    for ext in (".json", ".npz", ".pkl"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    with open(base + ".json", "r") as fp:
        meta = json.load(fp)
    with np.load(base + ".npz", allow_pickle=False) as npz:
        arrays = {key: npz[key] for key in npz.files}
    json_token = meta.get("write_token")
    npz_token = arrays.get("_write_token")
    npz_token = None if npz_token is None else str(npz_token)
    if json_token != npz_token:
        raise TornCheckpointError(
            "checkpoint .json and .npz carry different write tokens "
            f"({json_token!r} vs {npz_token!r}); the pair is torn (a crash "
            "between the two file commits left a fresh array file beside a "
            "stale metadata file). This checkpoint is ignored."
        )
    arrays.pop("_write_token", None)
    return arrays, meta


def _check_schema(meta: dict) -> None:
    """Validate ``schema_version``; refuse checkpoints newer than we support."""
    version = meta.get("schema_version")
    if version is None:
        raise CheckpointMismatchError(
            "checkpoint metadata is missing 'schema_version'; it was not "
            "written by this checkpoint format."
        )
    if version > CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointMismatchError(
            f"checkpoint schema_version {version} is newer than this "
            f"impulse supports (max {CHECKPOINT_SCHEMA_VERSION}); upgrade "
            "impulse to resume this checkpoint."
        )


# Groups of sampler-class names that are the same class under different
# (current / deprecated) names, so a checkpoint written under one resumes
# into the other. ``HybridPTSampler`` was called ``RJPTSampler`` before the
# birth-death/product-space rename.
_SAMPLER_CLASS_ALIASES = (frozenset({"HybridPTSampler", "RJPTSampler"}),)


def _sampler_class_matches(ck_name: Any, cls_name: str) -> bool:
    """Return True if a checkpoint's sampler-class name matches the class name.

    An exact match, or a match through a known deprecated-alias group (e.g. a
    checkpoint written by the pre-rename ``RJPTSampler`` resuming into
    ``HybridPTSampler``).
    """
    if ck_name == cls_name:
        return True
    return any(ck_name in group and cls_name in group for group in _SAMPLER_CLASS_ALIASES)


def _verify_checkpoint_metadata(sampler: Any, meta: dict) -> None:
    """Verify the reconstructed sampler matches the checkpoint; raise on mismatch.

    Checks (first mismatch reported): sampler class, ``ndim``, ``ntemps``,
    and — per chain — the ordered proposal names and their weights. This is
    the guardrail behind the reconstruct-then-restore contract: callables are
    not serialized, so the caller must rebuild the sampler exactly as the
    original run did.
    """
    cls_name = type(sampler).__name__
    if not _sampler_class_matches(meta.get("sampler_class"), cls_name):
        raise CheckpointMismatchError(
            f"checkpoint was written by {meta.get('sampler_class')!r} but is "
            f"being resumed into a {cls_name!r}; reconstruct the same sampler class."
        )
    if int(meta["ndim"]) != int(sampler.ndim):
        raise CheckpointMismatchError(
            f"ndim mismatch: checkpoint has ndim={meta['ndim']} but the "
            f"reconstructed sampler has ndim={sampler.ndim}."
        )
    if int(meta["ntemps"]) != int(sampler.ntemps):
        raise CheckpointMismatchError(
            f"ntemps mismatch: checkpoint has ntemps={meta['ntemps']} but the "
            f"reconstructed sampler has ntemps={sampler.ntemps}."
        )
    ck_bundle = meta["proposal_bundle"]
    jump_proposals = sampler.proposal_bundle.jump_proposals
    for i, jp in enumerate(jump_proposals):
        fresh_names = [getattr(p, "__name__", type(p).__name__) for p in jp.proposal_list]
        fresh_weights = [float(w) for w in jp.proposal_weights]
        ck_names = list(ck_bundle[i]["names"])
        ck_weights = [float(w) for w in ck_bundle[i]["weights"]]
        if fresh_names != ck_names:
            missing = [n for n in ck_names if n not in fresh_names]
            extra = [n for n in fresh_names if n not in ck_names]
            detail = ""
            if missing:
                detail += f" missing proposal(s) {missing};"
            if extra:
                detail += f" unexpected proposal(s) {extra};"
            raise CheckpointMismatchError(
                f"proposal mismatch on chain {i}: checkpoint registered "
                f"{ck_names} but the reconstructed sampler has {fresh_names}."
                f"{detail} register the same proposals in the same order "
                "before resuming."
            )
        if fresh_weights != ck_weights:
            for j, (fw, cw) in enumerate(zip(fresh_weights, ck_weights)):
                if fw != cw:
                    raise CheckpointMismatchError(
                        f"proposal weight mismatch on chain {i} at position "
                        f"{j} ({ck_names[j]!r}): checkpoint weight {cw} but "
                        f"the reconstructed sampler has weight {fw}."
                    )


def restore_state_checkpoint(sampler: Any, path: str) -> dict:
    """Load, verify, and restore a new-format checkpoint into ``sampler`` in place.

    Reads the checkpoint (no code execution), checks the schema version,
    verifies the reconstructed sampler matches the metadata, then restores
    every state family (RNG streams, positions, adaptive statistics, DE
    buffers, PT ladder, chain-file ring buffer, NUTS adapter, ...) into the
    already-wired ``sampler`` objects.

    Parameters
    ----------
    sampler : PTSampler or HybridPTSampler
        Freshly reconstructed sampler to restore into.
    path : str
        Path to the ``.json`` sidecar (or base/``.npz``).

    Returns
    -------
    dict
        The parsed metadata (callers read ``num_adapt`` for resume semantics).
    """
    arrays, meta = load_state_checkpoint(path)
    _check_schema(meta)
    _verify_checkpoint_metadata(sampler, meta)
    sampler._restore_checkpoint_state(arrays, meta)
    return meta


# ---------------------------------------------------------------------------
# Public write entry point (dispatches new format vs. legacy pickle)
# ---------------------------------------------------------------------------


def checkpoint_sampler(
    sampler: Any,
    path: Optional[str] = None,
    omit: Iterable[str] = ("lnlike", "lnprior"),
    format: Optional[str] = None,
) -> str:
    """
    Create an atomic checkpoint of sampler state for resuming interrupted runs.

    By default this writes the no-code-execution format (``.npz`` + ``.json``)
    for PT/hybrid samplers. It falls back to a legacy pickle when the target
    ``path`` ends in ``.pkl`` (e.g. :class:`~impulse.nuts.sampler.NUTSSampler`,
    whose checkpointing is separate) or when the sampler does not implement the
    new-format capture hook.

    Parameters
    ----------
    sampler : Any
        Sampler instance to checkpoint.
    path : str, optional
        Output path. If ``None``, uses ``<sampler.outdir>/sampler_checkpoint``
        with the extension chosen by the format.
    omit : iterable of str, default ('lnlike', 'lnprior')
        Attribute names temporarily set to ``None`` during pickling. Used only
        on the legacy pickle path (the new format never serializes callables).
    format : {'npz', 'pickle'}, optional
        Force a format. ``None`` (default) auto-selects: new format for
        PT/hybrid samplers, pickle for a ``.pkl`` target.

    Returns
    -------
    str
        Path to the created checkpoint (the ``.json`` sidecar for the new
        format, or the ``.pkl`` for the legacy format).

    Notes
    -----
    - New format: array state via ``np.savez_compressed``, metadata as JSON;
      loading executes no code. See :func:`save_state_checkpoint`.
    - Legacy format: a Python pickle of the whole object; loading it can
      execute arbitrary code (see SECURITY.md).
    """
    use_pickle = format == "pickle" or (
        format is None
        and (
            (path is not None and path.endswith(".pkl"))
            or not hasattr(sampler, "_capture_checkpoint_state")
        )
    )
    if not use_pickle:
        return save_state_checkpoint(sampler, path=path)
    return _pickle_checkpoint(sampler, path=path, omit=omit)


def _pickle_checkpoint(
    sampler: Any, path: Optional[str] = None, omit: Iterable[str] = ("lnlike", "lnprior")
) -> str:
    """Legacy pickle writer (whole-object pickle with callables stripped).

    Retained for :class:`~impulse.nuts.sampler.NUTSSampler` and for continuing
    a run that was resumed from a ``.pkl``. Uses atomic rename and restores the
    stripped attributes on the in-memory object afterwards.
    """
    if path is None:
        path = os.path.join(getattr(sampler, "outdir", "."), _PKL_NAME)
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


# ---------------------------------------------------------------------------
# Legacy pickle loaders (unchanged behavior + loud security warning)
# ---------------------------------------------------------------------------


def load_checkpoint(path: str, lnlike: Callable, lnprior: Callable):
    """
    Load a sampler from a LEGACY pickle checkpoint and restore the callables.

    .. warning::
        This unpickles a whole sampler object, which can execute arbitrary
        code. Only load pickle checkpoints you trust (see SECURITY.md). The
        pickle format is deprecated in favor of the ``.npz``/``.json`` format;
        prefer ``resume=True``, which reads the new format when present.

    Parameters
    ----------
    path : str
        Path to a ``.pkl`` checkpoint created by the legacy format.
    lnlike, lnprior : callable
        Functions to rebind to the sampler (stripped before pickling).

    Returns
    -------
    PTSampler
        Restored sampler object ready to continue sampling.
    """
    warnings.warn(_PICKLE_SECURITY_WARNING, UserWarning, stacklevel=2)
    with open(path, "rb") as fp:
        sampler = pickle.load(fp)  # noqa: S301 - documented legacy trust boundary
    sampler.lnlike = lnlike
    sampler.lnprior = lnprior

    return sampler


def load_nuts_checkpoint(path: str, logp_and_grad: Callable):
    """Load a NUTSSampler from a LEGACY pickle checkpoint and restore its gradient.

    .. warning::
        Unpickling can execute arbitrary code; only load checkpoints you trust
        (see SECURITY.md). ``NUTSSampler`` checkpointing is separate from the
        PT engine and remains on the pickle format.

    Parameters
    ----------
    path : str
        Path to a ``.pkl`` checkpoint.
    logp_and_grad : callable
        Function ``(x) -> (logp, grad)`` to rebind to the sampler.

    Returns
    -------
    NUTSSampler
        Restored sampler ready to continue sampling.
    """
    warnings.warn(_PICKLE_SECURITY_WARNING, UserWarning, stacklevel=2)
    with open(path, "rb") as fp:
        sampler = pickle.load(fp)  # noqa: S301 - documented legacy trust boundary
    sampler.logp_and_grad = logp_and_grad
    return sampler


def load_hybrid_checkpoint(
    path: str,
    lnlike: Callable,
    lnprior: Callable,
    raw_lnlike: Optional[Callable] = None,
    raw_lnprior: Optional[Callable] = None,
    lnlike_grad: Optional[Callable] = None,
):
    """Load a HybridPTSampler from a LEGACY pickle checkpoint and restore callables.

    .. warning::
        Unpickling can execute arbitrary code; only load checkpoints you trust
        (see SECURITY.md). Prefer ``resume=True``, which reads the safe
        ``.npz``/``.json`` format when present.

    Parameters
    ----------
    path : str
        Path to a ``.pkl`` checkpoint.
    lnlike, lnprior : callable
        Wrapped likelihood/prior to rebind.
    raw_lnlike, raw_lnprior : callable, optional
        Unwrapped functions for NUTS gradient building.
    lnlike_grad : callable, optional
        Gradient function for NUTS.

    Returns
    -------
    HybridPTSampler
        Restored sampler ready to continue sampling.
    """
    warnings.warn(_PICKLE_SECURITY_WARNING, UserWarning, stacklevel=2)
    with open(path, "rb") as fp:
        sampler = pickle.load(fp)  # noqa: S301 - documented legacy trust boundary
    sampler.lnlike = lnlike
    sampler.lnprior = lnprior
    if raw_lnlike is not None:
        sampler._raw_lnlike = raw_lnlike
    if raw_lnprior is not None:
        sampler._raw_lnprior = raw_lnprior
    if lnlike_grad is not None:
        sampler.lnlike_grad = lnlike_grad
    return sampler


# Deprecated alias (pre-rename name; kept so existing code keeps importing).
load_rjpt_checkpoint = load_hybrid_checkpoint


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def check_for_checkpoint(outdir: str) -> Optional[str]:
    """
    Locate an existing checkpoint in ``outdir``, preferring the new format.

    Resolution order:

    1. ``sampler_checkpoint.json`` **and** ``sampler_checkpoint.npz`` present
       -> return the ``.json`` path (new format).
    2. A lone ``.npz`` (json sidecar absent) is a torn write and is ignored.
    3. ``sampler_checkpoint.pkl`` present -> return it (legacy fallback).
    4. Otherwise ``None``.

    Parameters
    ----------
    outdir : str
        Directory to search.

    Returns
    -------
    str or None
        Path to the checkpoint to resume from, or ``None`` if none is usable.
    """
    json_path = os.path.join(outdir, _JSON_NAME)
    npz_path = os.path.join(outdir, _NPZ_NAME)
    if (
        os.path.exists(json_path)
        and os.path.exists(npz_path)
        and _tokens_match(json_path, npz_path)
    ):
        return json_path
    pkl_path = os.path.join(outdir, _PKL_NAME)
    if os.path.exists(pkl_path):
        return pkl_path
    return None


def _tokens_match(json_path: str, npz_path: str) -> bool:
    """True if the .json and .npz share a write token (i.e. the write was not torn).

    Reads only the token from each file (the npz member is decompressed
    lazily), so an overwrite crash that left a fresh npz beside a stale
    json is detected here and the pair is skipped in favour of the legacy
    fallback. Any read error is treated as a non-match (torn/unreadable).
    """
    try:
        with open(json_path, "r") as fp:
            json_token = json.load(fp).get("write_token")
        with np.load(npz_path, allow_pickle=False) as npz:
            if "_write_token" not in npz.files:
                return json_token is None
            npz_token = str(npz["_write_token"])
        return json_token == npz_token
    except (OSError, ValueError, json.JSONDecodeError):
        return False
