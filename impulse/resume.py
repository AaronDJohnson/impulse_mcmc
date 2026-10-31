"""No-code-execution checkpointing and resume (.npz + .json)."""

import json
import os
import pathlib
import tempfile
import uuid
import zipfile
from typing import Any, Optional

import numpy as np

CHECKPOINT_SCHEMA_VERSION = 2

_NEW_BASENAME = "sampler_checkpoint"
# ``.pkl`` is stripped too: a legacy pickle path names the checkpoint, it does
# not select a format (pickles are no longer written or read).
_STRIP_EXTS = (".json", ".npz", ".pkl")
_NPZ_NAME = _NEW_BASENAME + ".npz"
_JSON_NAME = _NEW_BASENAME + ".json"


class CheckpointMismatchError(ValueError):
    """Raised when a reconstructed sampler does not match checkpoint metadata."""


class TornCheckpointError(ValueError):
    """Raised when a checkpoint's .json and .npz do not belong to the same write."""


def _jsonable_rng_state(state: Any) -> Any:
    """Deep-convert a ``bit_generator.state`` mapping to JSON-native types."""
    if isinstance(state, dict):
        return {k: _jsonable_rng_state(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return [_jsonable_rng_state(v) for v in state]
    if isinstance(state, bool):
        return state
    if isinstance(state, (int, np.integer)):
        return int(state)
    if isinstance(state, (float, np.floating)):
        return float(state)
    return state


def _rng_state_from_json(state: Any) -> Any:
    return state


def _checkpoint_base(path: Optional[str], sampler: Any) -> str:
    if path is None:
        outdir = getattr(sampler, "outdir", ".")
        return os.path.join(outdir, _NEW_BASENAME)
    for ext in _STRIP_EXTS:
        if path.endswith(ext):
            return path[: -len(ext)]
    return path


def save_state_checkpoint(sampler: Any, path: Optional[str] = None) -> str:
    """Atomically write a checkpoint as ``<base>.npz`` (arrays) + ``<base>.json`` (metadata)."""
    base = _checkpoint_base(path, sampler)
    npz_path = base + ".npz"
    json_path = base + ".json"
    dirn = os.path.dirname(npz_path) or "."
    pathlib.Path(dirn).mkdir(parents=True, exist_ok=True)

    arrays, meta = sampler._capture_checkpoint_state()
    write_token = uuid.uuid4().hex
    meta = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "write_token": write_token,
        **meta,
    }
    arrays = {**arrays, "_write_token": np.array(write_token)}

    npz_fd, tmp_npz = tempfile.mkstemp(dir=dirn, prefix=".ckpt.", suffix=".npz.tmp")
    os.close(npz_fd)
    json_fd, tmp_json = tempfile.mkstemp(dir=dirn, prefix=".ckpt.", suffix=".json.tmp")
    os.close(json_fd)
    try:
        with open(tmp_npz, "wb") as fp:
            np.savez(fp, **arrays)
        with open(tmp_json, "w", encoding="utf-8") as fp:
            json.dump(meta, fp)
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
    """Read a ``.npz`` + ``.json`` checkpoint without executing code."""
    base = path
    for ext in _STRIP_EXTS:
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    with open(base + ".json", "r", encoding="utf-8") as fp:
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


def _verify_checkpoint_metadata(sampler: Any, meta: dict) -> None:
    cls_name = type(sampler).__name__
    if meta.get("sampler_class") != cls_name:
        raise CheckpointMismatchError(
            f"checkpoint was written by {meta.get('sampler_class')!r} but is being resumed "
            f"into a {cls_name!r}; reconstruct the same sampler class."
        )

    # Subclasses verify their own run-shaping settings first, so a changed
    # setting is reported by name rather than as a derived ndim mismatch.
    verify_extra = getattr(sampler, "_verify_extra_metadata", None)
    if callable(verify_extra):
        verify_extra(meta)

    scalar_checks = (
        ("ndim", lambda s: int(s.ndim)),
        ("ntemps", lambda s: int(s.ntemps)),
        ("swap_steps", lambda s: int(s.swap_steps)),
        ("cov_update", lambda s: int(s.cov_update)),
        ("save_freq", lambda s: int(s.save_freq)),
        ("buffer_size", lambda s: int(s.multi_chain_stats.chain_stats[0].buffer_size)),
        ("buffer_thin", lambda s: int(s.multi_chain_stats.chain_stats[0].buffer_thin)),
    )
    for key, getter in scalar_checks:
        if key not in meta or meta[key] is None:
            continue
        expected = int(meta[key])
        actual = getter(sampler)
        if expected != actual:
            raise CheckpointMismatchError(
                f"{key} mismatch: checkpoint has {key}={expected} but the reconstructed "
                f"sampler has {key}={actual}; rebuild the sampler with {key}={expected} to "
                "resume this run, or start a fresh run (resume=False, or a new outdir) to "
                "change it."
            )

    if meta.get("chain_format") is not None:
        expected_cf = str(meta["chain_format"])
        actual_cf = str(getattr(sampler, "chain_format", expected_cf))
        if expected_cf != actual_cf:
            raise CheckpointMismatchError(
                f"chain_format mismatch: checkpoint has {expected_cf!r} but the reconstructed "
                f"sampler has {actual_cf!r}. The encoding lives in the chain files on disk, "
                "not in the checkpoint, so resuming into the other one writes a SECOND set "
                "of files in the new encoding and leaves load_chain() returning only the "
                f"original half. Rebuild with chain_format={expected_cf!r} to resume this "
                "run, or start a fresh run (resume=False, or a new outdir) to change it."
            )

    if "proposal_bundle" not in meta:
        return

    ck_bundle = meta["proposal_bundle"]
    for i, jp in enumerate(sampler.proposal_bundle.jump_proposals):
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
                f"proposal mismatch on chain {i}: checkpoint registered {ck_names} but the "
                f"reconstructed sampler has {fresh_names}.{detail} register the same "
                "proposals in the same order before resuming."
            )
        if fresh_weights != ck_weights:
            for j, (fw, cw) in enumerate(zip(fresh_weights, ck_weights)):
                if fw != cw:
                    raise CheckpointMismatchError(
                        f"proposal weight mismatch on chain {i} at position {j} "
                        f"({ck_names[j]!r}): checkpoint weight {cw} but the reconstructed "
                        f"sampler has weight {fw}."
                    )


def restore_state_checkpoint(sampler: Any, path: str) -> dict:
    """Load, verify, and restore a checkpoint into ``sampler`` in place."""
    arrays, meta = load_state_checkpoint(path)
    _check_schema(meta)
    _verify_checkpoint_metadata(sampler, meta)
    sampler._restore_checkpoint_state(arrays, meta)
    return meta


def checkpoint_sampler(sampler: Any, path: Optional[str] = None) -> str:
    """Create an atomic ``.npz`` + ``.json`` checkpoint of ``sampler``."""
    return save_state_checkpoint(sampler, path=path)


def check_for_checkpoint(outdir: str) -> Optional[str]:
    """Locate an existing checkpoint in ``outdir``, returning the ``.json`` path or ``None``."""
    json_path = os.path.join(outdir, _JSON_NAME)
    npz_path = os.path.join(outdir, _NPZ_NAME)
    if (
        os.path.exists(json_path)
        and os.path.exists(npz_path)
        and _tokens_match(json_path, npz_path)
    ):
        return json_path
    return None


def _tokens_match(json_path: str, npz_path: str) -> bool:
    try:
        with open(json_path, "r", encoding="utf-8") as fp:
            json_token = json.load(fp).get("write_token")
        with np.load(npz_path, allow_pickle=False) as npz:
            if "_write_token" not in npz.files:
                return json_token is None
            npz_token = str(npz["_write_token"])
        return json_token == npz_token
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        return False
