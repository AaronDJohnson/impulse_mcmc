"""Tests for the no-code-execution checkpoint format (.npz arrays + .json).

The default checkpoint format is a compressed ``.npz`` of the array state
plus a schema-versioned ``.json`` metadata sidecar.  Loading it uses
``numpy.load(..., allow_pickle=False)`` and ``json.load`` — no pickle, so a
tampered checkpoint cannot execute code.  Resume is *reconstruct then
restore*: rebuild the sampler exactly as the original run did, then restore
STATE into it (verifying the reconstruction matches the metadata first).

These tests cover: every state family round-trips (chain stats, DE buffer,
NUTS adapter incl. injected mass matrices and mid-window dual averaging,
flow frozen flag); the allow_pickle=False guarantee; metadata-mismatch
errors; torn-write recovery; the legacy-pickle fallback and its warning; and
schema-version handling.
"""

import importlib
import json
import os
import pickle

import numpy as np
import pytest

from impulse.birth_death import BirthDeathProductSpace
from impulse.hybrid_sampler import HybridPTSampler
from impulse.nuts.mass_matrix import MassMatrix, MassMatrixType
from impulse.resume import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointMismatchError,
    check_for_checkpoint,
    checkpoint_sampler,
    load_checkpoint,
    load_state_checkpoint,
    restore_state_checkpoint,
    save_state_checkpoint,
)
from impulse.samplers import PTSampler

SEED = 4321


# ---------------------------------------------------------------------------
# Picklable model functions (module-level so the legacy pickle path works)
# ---------------------------------------------------------------------------


def _gauss_lnlike(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return -0.5 * np.sum(x**2)
    return -0.5 * np.sum(x**2, axis=1)


def _flat_lnprior(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return 0.0 if np.all(np.abs(x) <= 10) else -np.inf
    result = np.zeros(x.shape[0])
    result[np.any(np.abs(x) > 10, axis=1)] = -np.inf
    return result


def _shift_proposal(chain_stats):
    """Picklable symmetric custom proposal (module-level so it is importable)."""
    x = chain_stats.current_sample.copy()
    x += 0.05 * chain_stats.rng.standard_normal(chain_stats.ndim)
    return x, 0.0


_shift_proposal.__name__ = "shift_proposal"


NUM_PARAMS = 2
MAX_SOURCES = 2
LO = np.array([0.0, 0.0])
HI = np.array([5.0, 3.0])
_N_PTS = 40
_T_GRID = np.linspace(0, 2 * np.pi, _N_PTS)
_DATA = 2.0 * np.sin(2 * np.pi * 1.0 * _T_GRID)


def _rj_source_draw(rng):
    return rng.uniform(LO, HI)


def _rj_logprior(params):
    for i in range(len(params) // NUM_PARAMS):
        p = params[i * NUM_PARAMS : (i + 1) * NUM_PARAMS]
        if np.any(p < LO) or np.any(p > HI):
            return -np.inf
    return 0.0


def _rj_model(active):
    model = np.zeros(_N_PTS)
    for i in range(len(active) // NUM_PARAMS):
        a = active[i * NUM_PARAMS]
        f = active[i * NUM_PARAMS + 1]
        model += a * np.sin(2 * np.pi * f * _T_GRID)
    return model


def _rj_loglike(active):
    return -0.5 * np.sum((_DATA - _rj_model(np.asarray(active))) ** 2)


def _rj_lnlike_grad(active):
    active = np.asarray(active, dtype=np.float64)
    model = _rj_model(active)
    residual = _DATA - model
    ll = -0.5 * np.sum(residual**2)
    grad = np.zeros_like(active)
    for i in range(len(active) // NUM_PARAMS):
        a = active[i * NUM_PARAMS]
        f = active[i * NUM_PARAMS + 1]
        sin_term = np.sin(2 * np.pi * f * _T_GRID)
        cos_term = np.cos(2 * np.pi * f * _T_GRID)
        grad[i * NUM_PARAMS] = np.sum(residual * sin_term)
        grad[i * NUM_PARAMS + 1] = np.sum(residual * a * 2 * np.pi * _T_GRID * cos_term)
    return ll, grad


def _make_rj_space():
    return BirthDeathProductSpace(
        loglikelihood=_rj_loglike,
        logprior=_rj_logprior,
        num_sources=MAX_SOURCES,
        num_params=NUM_PARAMS,
        source_prior_draw=_rj_source_draw,
    )


def _pt(outdir, resume=False, **kw):
    return PTSampler(
        ndim=2,
        lnlike=_gauss_lnlike,
        lnprior=_flat_lnprior,
        ntemps=3,
        seed=SEED,
        outdir=outdir,
        resume=resume,
        save_freq=20,
        cov_update=10,
        buffer_size=80,
        **kw,
    )


def _hybrid_nuts(outdir, resume=False):
    return HybridPTSampler.from_product_space(
        _make_rj_space(),
        lnlike_grad=_rj_lnlike_grad,
        ntemps=3,
        seed=SEED,
        outdir=outdir,
        resume=resume,
        save_freq=20,
        cov_update=10,
        buffer_size=120,
        max_tree_depth=4,
        hot_chain_max_depth=3,
        mass_matrix_adapt_interval=25,
        mass_matrix_min_samples=10,
    )


def _rj_x0():
    return _make_rj_space().draw_initial_position(np.random.default_rng(SEED), nmodel=0)


# ---------------------------------------------------------------------------
# Format / file layout
# ---------------------------------------------------------------------------


class TestFormatLayout:
    def test_writes_npz_and_json_not_pkl(self, tmp_path):
        s = _pt(str(tmp_path))
        s.sample(np.array([0.1, 0.2]), num_iterations=40)
        assert os.path.exists(tmp_path / "sampler_checkpoint.npz")
        assert os.path.exists(tmp_path / "sampler_checkpoint.json")
        assert not os.path.exists(tmp_path / "sampler_checkpoint.pkl")

    def test_check_for_checkpoint_prefers_new_format(self, tmp_path):
        s = _pt(str(tmp_path))
        s.sample(np.array([0.1, 0.2]), num_iterations=40)
        found = check_for_checkpoint(str(tmp_path))
        assert found == str(tmp_path / "sampler_checkpoint.json")

    def test_json_has_schema_version(self, tmp_path):
        s = _pt(str(tmp_path))
        s.sample(np.array([0.1, 0.2]), num_iterations=40)
        with open(tmp_path / "sampler_checkpoint.json") as fp:
            meta = json.load(fp)
        assert meta["schema_version"] == CHECKPOINT_SCHEMA_VERSION
        assert meta["sampler_class"] == "PTSampler"
        assert meta["ndim"] == 2
        assert meta["ntemps"] == 3
        assert "impulse_version" in meta

    def test_compressed_smaller_than_pickle(self, tmp_path):
        s = _hybrid_nuts(str(tmp_path))
        s.sample(_rj_x0(), num_iterations=120)
        npz = os.path.getsize(tmp_path / "sampler_checkpoint.npz")
        pkl = os.path.join(str(tmp_path), "sampler_checkpoint.pkl")
        checkpoint_sampler(
            s,
            path=pkl,
            format="pickle",
            omit=("lnlike", "lnprior", "_raw_lnlike", "_raw_lnprior", "lnlike_grad"),
        )
        assert npz < os.path.getsize(pkl)


# ---------------------------------------------------------------------------
# No pickle on load
# ---------------------------------------------------------------------------


class TestNoPickleOnLoad:
    def test_load_never_calls_pickle(self, tmp_path, monkeypatch):
        """Restoring a new-format checkpoint must not touch pickle at all."""
        s = _pt(str(tmp_path))
        s.sample(np.array([0.1, 0.2]), num_iterations=40)

        import impulse.resume as resume_mod

        def _boom(*a, **k):
            raise AssertionError("pickle.load must not be called by the new format")

        monkeypatch.setattr(resume_mod.pickle, "load", _boom)

        fresh = _pt(str(tmp_path), resume=True)
        # A full resume goes through restore_state_checkpoint; if it touched
        # pickle.load the monkeypatched bomb would fire.
        fresh.sample(np.array([0.1, 0.2]), num_iterations=45)
        assert fresh.short_chain.iteration == 45

    def test_np_load_uses_allow_pickle_false(self, tmp_path, monkeypatch):
        s = _pt(str(tmp_path))
        s.sample(np.array([0.1, 0.2]), num_iterations=40)

        seen = {}
        real_load = np.load

        def _spy(*args, **kwargs):
            seen["allow_pickle"] = kwargs.get("allow_pickle", "MISSING")
            return real_load(*args, **kwargs)

        monkeypatch.setattr(np, "load", _spy)
        load_state_checkpoint(str(tmp_path / "sampler_checkpoint.json"))
        assert seen["allow_pickle"] is False

    def test_object_array_npz_is_refused(self, tmp_path):
        """A checkpoint whose .npz smuggles a Python object array must raise
        on load (allow_pickle=False rejects object arrays)."""
        base = str(tmp_path / "sampler_checkpoint")
        # Object array can only be written with allow_pickle (pickled inside
        # the npz); it can only be read back with allow_pickle=True.
        np.savez(base + ".npz", evil=np.array([{"code": "exec"}], dtype=object))
        with open(base + ".json", "w") as fp:
            json.dump({"schema_version": CHECKPOINT_SCHEMA_VERSION}, fp)
        with pytest.raises(ValueError):
            load_state_checkpoint(base + ".json")


# ---------------------------------------------------------------------------
# Round-trip of every state family
# ---------------------------------------------------------------------------


class TestStateRoundTrip:
    def test_pt_chain_stats_and_de_buffer(self, tmp_path):
        original = _pt(str(tmp_path))
        original.sample(np.array([0.3, -0.2]), num_iterations=60)
        # Capture the exact end-of-run state (later than the last auto boundary).
        save_state_checkpoint(original)

        restored = _pt(str(tmp_path))  # fresh, not yet resumed
        restore_state_checkpoint(restored, str(tmp_path / "sampler_checkpoint.json"))

        for i in range(original.ntemps):
            oc = original.multi_chain_stats.chain_stats[i]
            rc = restored.multi_chain_stats.chain_stats[i]
            assert rc.sample_total == oc.sample_total
            assert rc.buffer_full == oc.buffer_full
            np.testing.assert_array_equal(rc.sample_cov, oc.sample_cov)
            np.testing.assert_array_equal(rc.sample_mean, oc.sample_mean)
            # DE history buffer content must round-trip exactly.
            np.testing.assert_array_equal(rc._buffer, oc._buffer)
            for gi in range(len(oc.groups)):
                np.testing.assert_array_equal(rc.proposal_L[gi], oc.proposal_L[gi])

    def test_hybrid_per_model_de_buffers(self, tmp_path):
        original = _hybrid_nuts(str(tmp_path))
        original.sample(_rj_x0(), num_iterations=120)
        save_state_checkpoint(original)

        restored = _hybrid_nuts(str(tmp_path))
        restore_state_checkpoint(restored, str(tmp_path / "sampler_checkpoint.json"))

        for i in range(original.ntemps):
            oc = original.multi_chain_stats.chain_stats[i]
            rc = restored.multi_chain_stats.chain_stats[i]
            assert set(rc._per_model.keys()) == set(oc._per_model.keys())
            for k in oc._per_model:
                opm, rpm = oc._per_model[k], rc._per_model[k]
                assert rpm.sample_total == opm.sample_total
                assert rpm.buffer_full == opm.buffer_full
                np.testing.assert_array_equal(rpm.buffer, opm.buffer)
                np.testing.assert_array_equal(rpm.sample_cov, opm.sample_cov)

    def test_rng_states_round_trip(self, tmp_path):
        original = _pt(str(tmp_path))
        original.sample(np.array([0.3, -0.2]), num_iterations=60)
        save_state_checkpoint(original)
        restored = _pt(str(tmp_path))
        restore_state_checkpoint(restored, str(tmp_path / "sampler_checkpoint.json"))
        for orig_rng, res_rng in zip(original.rngs, restored.rngs):
            assert res_rng.bit_generator.state == orig_rng.bit_generator.state

    def test_pt_ladder_and_swap_counters(self, tmp_path):
        original = _pt(str(tmp_path))
        original.sample(np.array([0.3, -0.2]), num_iterations=60)
        save_state_checkpoint(original)
        restored = _pt(str(tmp_path))
        restore_state_checkpoint(restored, str(tmp_path / "sampler_checkpoint.json"))
        np.testing.assert_array_equal(restored.ptstate.ladder, original.ptstate.ladder)
        np.testing.assert_array_equal(restored.ptstate.swap_accept, original.ptstate.swap_accept)
        assert restored.ptstate.nswaps == original.ptstate.nswaps
        # temps must alias the (restored) ladder, exactly as the loop leaves it.
        assert restored.state.temps is restored.ptstate.ladder

    def test_nuts_adapter_round_trip(self, tmp_path):
        original = _hybrid_nuts(str(tmp_path))
        original.sample(_rj_x0(), num_iterations=120)
        save_state_checkpoint(original)
        ad = original._nuts_adapter
        # Sanity: something actually got adapted.
        assert ad.step_sizes
        assert any(mm.matrix_type != MassMatrixType.UNIT for mm in ad.mass_matrices.values())
        assert any(da.count > 0 for da in ad.dual_averagers.values())

        restored = _hybrid_nuts(str(tmp_path))
        restore_state_checkpoint(restored, str(tmp_path / "sampler_checkpoint.json"))
        rad = restored._nuts_adapter

        assert rad.step_sizes == ad.step_sizes
        assert rad.steps_since_mm_update == ad.steps_since_mm_update
        assert set(rad.mass_matrices) == set(ad.mass_matrices)
        for na, mm in ad.mass_matrices.items():
            rmm = rad.mass_matrices[na]
            assert rmm.matrix_type == mm.matrix_type
            # kinetic energy / momentum scaling reproduce exactly
            p = np.arange(1.0, na + 1.0)
            assert rmm.kinetic_energy(p) == mm.kinetic_energy(p)
        # Dual-averaging mid-window state (count>0) restores exactly.
        for key, da in ad.dual_averagers.items():
            rda = rad.dual_averagers[key]
            assert rda.count == da.count
            assert rda.finalize() == da.finalize()
            assert rda.h_bar == da.h_bar

    def test_injected_mass_matrix_round_trip(self, tmp_path):
        original = _hybrid_nuts(str(tmp_path))
        original.sample(_rj_x0(), num_iterations=40)
        # Inject a Fisher/precision mass matrix for n_active=2 and checkpoint.
        precision = np.array([[4.0, 0.5], [0.5, 2.0]])
        original.set_mass_matrix(2, MassMatrix.from_precision(precision, MassMatrixType.DENSE))
        assert 2 in original._nuts_adapter.mass_matrix_injected
        save_state_checkpoint(original)

        restored = _hybrid_nuts(str(tmp_path))
        restore_state_checkpoint(restored, str(tmp_path / "sampler_checkpoint.json"))
        assert 2 in restored._nuts_adapter.mass_matrix_injected
        omm = original._nuts_adapter.mass_matrices[2]
        rmm = restored._nuts_adapter.mass_matrices[2]
        assert rmm.matrix_type == MassMatrixType.DENSE == omm.matrix_type
        rng = np.random.default_rng(0)
        # Same Cholesky factor -> identical momentum draw
        np.testing.assert_array_equal(
            rmm.sample_momentum(np.random.default_rng(0)),
            omm.sample_momentum(np.random.default_rng(0)),
        )
        _ = rng

    def test_short_chain_ring_buffer_round_trip(self, tmp_path):
        original = _pt(str(tmp_path))
        original.sample(np.array([0.3, -0.2]), num_iterations=57)
        save_state_checkpoint(original)
        restored = _pt(str(tmp_path))
        restore_state_checkpoint(restored, str(tmp_path / "sampler_checkpoint.json"))
        osc, rsc = original.short_chain, restored.short_chain
        assert rsc.iteration == osc.iteration
        assert rsc._rows_written == osc._rows_written
        assert rsc._unsaved == osc._unsaved
        np.testing.assert_array_equal(rsc.samples, osc.samples)
        np.testing.assert_array_equal(rsc.lnprob, osc.lnprob)


# ---------------------------------------------------------------------------
# Metadata-mismatch verification
# ---------------------------------------------------------------------------


class TestMetadataMismatch:
    def test_wrong_ndim_raises(self, tmp_path):
        original = _pt(str(tmp_path))
        original.sample(np.array([0.3, -0.2]), num_iterations=40)
        wrong = PTSampler(
            ndim=3,
            lnlike=_gauss_lnlike,
            lnprior=_flat_lnprior,
            ntemps=3,
            seed=SEED,
            outdir=str(tmp_path),
        )
        with pytest.raises(CheckpointMismatchError, match="ndim"):
            restore_state_checkpoint(wrong, str(tmp_path / "sampler_checkpoint.json"))

    def test_wrong_ntemps_raises(self, tmp_path):
        original = _pt(str(tmp_path))
        original.sample(np.array([0.3, -0.2]), num_iterations=40)
        wrong = PTSampler(
            ndim=2,
            lnlike=_gauss_lnlike,
            lnprior=_flat_lnprior,
            ntemps=5,
            seed=SEED,
            outdir=str(tmp_path),
        )
        with pytest.raises(CheckpointMismatchError, match="ntemps"):
            restore_state_checkpoint(wrong, str(tmp_path / "sampler_checkpoint.json"))

    def test_wrong_class_raises(self, tmp_path):
        original = _pt(str(tmp_path))
        original.sample(np.array([0.3, -0.2]), num_iterations=40)
        wrong = HybridPTSampler(
            ndim=2,
            lnlike=_gauss_lnlike,
            lnprior=_flat_lnprior,
            ntemps=3,
            seed=SEED,
            outdir=str(tmp_path),
        )
        with pytest.raises(CheckpointMismatchError, match="written by"):
            restore_state_checkpoint(wrong, str(tmp_path / "sampler_checkpoint.json"))

    def test_missing_custom_jump_raises(self, tmp_path):
        original = _pt(str(tmp_path))
        original.add_custom_jump(_shift_proposal, weight=10)
        original.sample(np.array([0.3, -0.2]), num_iterations=40)
        # Reconstruct WITHOUT re-registering the custom jump.
        wrong = _pt(str(tmp_path))
        with pytest.raises(CheckpointMismatchError, match="missing"):
            restore_state_checkpoint(wrong, str(tmp_path / "sampler_checkpoint.json"))

    def test_reordered_weights_raise(self, tmp_path):
        original = _pt(str(tmp_path))
        original.add_custom_jump(_shift_proposal, weight=10)
        original.sample(np.array([0.3, -0.2]), num_iterations=40)
        # Re-register the same jump with a DIFFERENT weight.
        wrong = _pt(str(tmp_path))
        wrong.add_custom_jump(_shift_proposal, weight=99)
        with pytest.raises(CheckpointMismatchError, match="weight"):
            restore_state_checkpoint(wrong, str(tmp_path / "sampler_checkpoint.json"))

    @pytest.mark.parametrize(
        "field,value",
        [
            ("save_freq", 55),
            ("cov_update", 33),
            ("buffer_size", 500),
            ("swap_steps", 7),
        ],
    )
    def test_run_shaping_scalar_mismatch_raises(self, tmp_path, field, value):
        """Every scalar the checkpoint records must be verified, not just ndim/ntemps.

        Regression test: these four were written into the checkpoint metadata but
        never compared on resume. Silently accepting them corrupts the run --
        a changed save_freq discards chain rows and a changed buffer_size breaks
        the len(_buffer) == buffer_size invariant.
        """
        original = _pt(str(tmp_path))
        original.sample(np.array([0.3, -0.2]), num_iterations=40)
        # _pt pins save_freq/cov_update/buffer_size, so build explicitly to
        # override exactly one of them.
        params = dict(
            ndim=2,
            lnlike=_gauss_lnlike,
            lnprior=_flat_lnprior,
            ntemps=3,
            seed=SEED,
            outdir=str(tmp_path),
            save_freq=20,
            cov_update=10,
            buffer_size=80,
        )
        params[field] = value
        wrong = PTSampler(**params)
        with pytest.raises(CheckpointMismatchError, match=field):
            restore_state_checkpoint(wrong, str(tmp_path / "sampler_checkpoint.json"))

    def test_resume_preserves_every_chain_row(self, tmp_path):
        """Resuming to N total iterations must leave exactly N rows on disk.

        Regression test for silent data loss: with save_freq unverified, resuming
        a 100-iteration run with a different save_freq truncated the chain file to
        the last checkpointed row and discarded the remainder without warning.
        """
        _pt(str(tmp_path)).sample(np.array([0.3, -0.2]), num_iterations=100)
        _pt(str(tmp_path), resume=True).sample(np.array([0.3, -0.2]), num_iterations=260)

        with open(tmp_path / "chain_0.txt") as fh:
            assert sum(1 for _ in fh) == 260

    def test_restore_keeps_buffer_size_invariant(self, tmp_path):
        """set_checkpoint_state must restore buffer_size alongside the buffer.

        Regression test: restoring _buffer without buffer_size left the DE
        proposal drawing indices from a range that did not match the array it
        indexes -- IndexError at proposals.py when the buffer grew.
        """
        original = _pt(str(tmp_path))
        original.sample(np.array([0.3, -0.2]), num_iterations=60)
        good = _pt(str(tmp_path))
        restore_state_checkpoint(good, str(tmp_path / "sampler_checkpoint.json"))

        for cs in good.multi_chain_stats.chain_stats:
            assert len(cs._buffer) == cs.buffer_size

    def test_matching_custom_jump_restores(self, tmp_path):
        original = _pt(str(tmp_path))
        original.add_custom_jump(_shift_proposal, weight=10)
        original.sample(np.array([0.3, -0.2]), num_iterations=40)
        save_state_checkpoint(original)
        good = _pt(str(tmp_path))
        good.add_custom_jump(_shift_proposal, weight=10)
        # No raise; state restored.
        meta = restore_state_checkpoint(good, str(tmp_path / "sampler_checkpoint.json"))
        assert good.short_chain.iteration == original.short_chain.iteration
        assert meta["ndim"] == 2


# ---------------------------------------------------------------------------
# Schema version handling
# ---------------------------------------------------------------------------


class TestSchemaVersion:
    def test_future_schema_version_refused(self, tmp_path):
        s = _pt(str(tmp_path))
        s.sample(np.array([0.1, 0.2]), num_iterations=40)
        json_path = tmp_path / "sampler_checkpoint.json"
        with open(json_path) as fp:
            meta = json.load(fp)
        meta["schema_version"] = CHECKPOINT_SCHEMA_VERSION + 5
        with open(json_path, "w") as fp:
            json.dump(meta, fp)
        fresh = _pt(str(tmp_path))
        with pytest.raises(CheckpointMismatchError, match="newer than"):
            restore_state_checkpoint(fresh, str(json_path))

    def test_missing_schema_version_refused(self, tmp_path):
        s = _pt(str(tmp_path))
        s.sample(np.array([0.1, 0.2]), num_iterations=40)
        json_path = tmp_path / "sampler_checkpoint.json"
        with open(json_path) as fp:
            meta = json.load(fp)
        del meta["schema_version"]
        with open(json_path, "w") as fp:
            json.dump(meta, fp)
        fresh = _pt(str(tmp_path))
        with pytest.raises(CheckpointMismatchError, match="schema_version"):
            restore_state_checkpoint(fresh, str(json_path))


# ---------------------------------------------------------------------------
# Torn write recovery + legacy fallback
# ---------------------------------------------------------------------------


class TestTornWriteAndFallback:
    def test_npz_without_json_is_ignored(self, tmp_path):
        s = _pt(str(tmp_path))
        s.sample(np.array([0.1, 0.2]), num_iterations=40)
        # Simulate a torn write: JSON commit marker never landed.
        os.remove(tmp_path / "sampler_checkpoint.json")
        assert os.path.exists(tmp_path / "sampler_checkpoint.npz")
        assert check_for_checkpoint(str(tmp_path)) is None

    def test_torn_write_falls_back_to_legacy_pkl(self, tmp_path):
        s = _pt(str(tmp_path))
        s.sample(np.array([0.1, 0.2]), num_iterations=40)
        # Legacy pickle present alongside a torn new-format write.
        checkpoint_sampler(s, path=str(tmp_path / "sampler_checkpoint.pkl"), format="pickle")
        os.remove(tmp_path / "sampler_checkpoint.json")
        found = check_for_checkpoint(str(tmp_path))
        assert found == str(tmp_path / "sampler_checkpoint.pkl")

    def test_mismatched_write_tokens_treated_as_torn(self, tmp_path):
        """An OVERWRITE crash can leave a fresh .npz beside a STALE .json
        (both present). The per-write token detects the mismatch: the pair
        is ignored (falls back to legacy .pkl) and a direct load raises."""
        from impulse.resume import TornCheckpointError, load_state_checkpoint

        s = _pt(str(tmp_path))
        s.sample(np.array([0.1, 0.2]), num_iterations=40)
        json_path = tmp_path / "sampler_checkpoint.json"
        # Simulate the stale-json half of a torn overwrite: rewrite the JSON
        # with a different write_token than the .npz carries.
        meta = json.loads(json_path.read_text())
        meta["write_token"] = "0" * 32
        json_path.write_text(json.dumps(meta))
        # Both files exist, but check_for_checkpoint rejects the torn pair.
        checkpoint_sampler(s, path=str(tmp_path / "sampler_checkpoint.pkl"), format="pickle")
        assert check_for_checkpoint(str(tmp_path)) == str(tmp_path / "sampler_checkpoint.pkl")
        # A direct load of the torn pair raises rather than restoring
        # mismatched state.
        with pytest.raises(TornCheckpointError):
            load_state_checkpoint(str(json_path))

    def test_legacy_pkl_loads_with_warning(self, tmp_path):
        s = _pt(str(tmp_path))
        s.sample(np.array([0.1, 0.2]), num_iterations=40)
        pkl = str(tmp_path / "sampler_checkpoint.pkl")
        checkpoint_sampler(s, path=pkl, format="pickle")
        with pytest.warns(UserWarning, match="pickle"):
            loaded = load_checkpoint(pkl, _gauss_lnlike, _flat_lnprior)
        assert loaded.ndim == 2

    def test_resume_prefers_new_format_over_stale_pkl(self, tmp_path):
        """With both a fresh .npz/.json and a stale .pkl present, resume must
        pick the new format (safe) and never unpickle the .pkl."""
        s = _pt(str(tmp_path))
        s.sample(np.array([0.1, 0.2]), num_iterations=40)
        checkpoint_sampler(s, path=str(tmp_path / "sampler_checkpoint.pkl"), format="pickle")
        found = check_for_checkpoint(str(tmp_path))
        assert found.endswith(".json")


# ---------------------------------------------------------------------------
# Flow proposal frozen flag (optional dependency)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("coppuccino") is None,
    reason="coppuccino is an optional dependency",
)
class TestFlowProposalState:
    def test_frozen_flag_round_trips(self):
        from impulse.flow_proposals import NormalizingFlowProposal

        prop = NormalizingFlowProposal(min_samples=50, refit_interval=5)
        prop._call_count = 123
        prop._fit_count = 4
        prop._last_fit_at = 100
        prop.freeze_adaptation()
        assert prop.frozen is True

        state = prop.get_checkpoint_state()
        # State is JSON-scalar-only (no arrays, no flow object).
        assert json.loads(json.dumps(state)) == state
        assert "flow" not in state

        fresh = NormalizingFlowProposal(min_samples=50, refit_interval=5)
        fresh.set_checkpoint_state(state)
        assert fresh.frozen is True
        assert fresh._call_count == 123
        assert fresh._fit_count == 4
        assert fresh._last_fit_at == 100
        assert fresh.flow is None
