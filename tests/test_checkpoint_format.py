"""Tests for the no-code-execution checkpoint format (.npz arrays + .json)."""

import importlib
import json
import os

import numpy as np
import pytest
from conftest import count_chain_rows

from impulse.chain_stats import ChainStats
from impulse.resume import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointMismatchError,
    TornCheckpointError,
    check_for_checkpoint,
    load_state_checkpoint,
    restore_state_checkpoint,
    save_state_checkpoint,
)
from impulse.samplers import PTSampler, PTState

SEED = 4321


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
    x = chain_stats.current_sample.copy()
    x += 0.05 * chain_stats.rng.standard_normal(chain_stats.ndim)
    return x, 0.0


_shift_proposal.__name__ = "shift_proposal"


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


class TestFormatLayout:
    def test_writes_npz_and_json_not_pkl(self, tmp_path):
        s = _pt(str(tmp_path))
        s.sample(np.array([0.1, 0.2]), num_iterations=40)
        assert os.path.exists(tmp_path / "sampler_checkpoint.npz")
        assert os.path.exists(tmp_path / "sampler_checkpoint.json")
        assert not os.path.exists(tmp_path / "sampler_checkpoint.pkl")

    def test_check_for_checkpoint_finds_json(self, tmp_path):
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

    def test_npz_is_uncompressed_and_not_bloated(self, tmp_path):
        s = _pt(str(tmp_path))
        s.sample(np.array([0.1, 0.2]), num_iterations=120)

        npz_path = tmp_path / "sampler_checkpoint.npz"
        on_disk = os.path.getsize(npz_path)
        with np.load(npz_path, allow_pickle=False) as z:
            raw = sum(z[k].nbytes for k in z.files)

        assert on_disk >= 0.95 * raw, "smaller than its own arrays -- compression is back"
        assert on_disk <= raw + 64 * 1024, "far larger than its arrays -- unexpected payload"


class TestNoPickleOnLoad:
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
        base = str(tmp_path / "sampler_checkpoint")
        np.savez(base + ".npz", evil=np.array([{"code": "exec"}], dtype=object))
        with open(base + ".json", "w") as fp:
            json.dump({"schema_version": CHECKPOINT_SCHEMA_VERSION}, fp)
        with pytest.raises(ValueError):
            load_state_checkpoint(base + ".json")


class TestStateRoundTrip:
    def test_pt_chain_stats_and_de_buffer(self, tmp_path):
        original = _pt(str(tmp_path))
        original.sample(np.array([0.3, -0.2]), num_iterations=60)
        save_state_checkpoint(original)

        restored = _pt(str(tmp_path))
        restore_state_checkpoint(restored, str(tmp_path / "sampler_checkpoint.json"))

        for i in range(original.ntemps):
            oc = original.multi_chain_stats.chain_stats[i]
            rc = restored.multi_chain_stats.chain_stats[i]
            assert rc.sample_total == oc.sample_total
            assert rc.buffer_full == oc.buffer_full
            np.testing.assert_array_equal(rc.sample_cov, oc.sample_cov)
            np.testing.assert_array_equal(rc.sample_mean, oc.sample_mean)
            np.testing.assert_array_equal(rc._buffer, oc._buffer)
            for gi in range(len(oc.groups)):
                np.testing.assert_array_equal(rc.proposal_L[gi], oc.proposal_L[gi])

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
        assert restored.state.temps is restored.ptstate.ladder

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
        with pytest.raises(CheckpointMismatchError, match="ndim") as err:
            restore_state_checkpoint(wrong, str(tmp_path / "sampler_checkpoint.json"))
        assert "rebuild the sampler with ndim=2" in str(err.value)

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

        class OtherSampler(PTSampler):
            pass

        wrong = OtherSampler(
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
        wrong = _pt(str(tmp_path))
        with pytest.raises(CheckpointMismatchError, match="missing") as err:
            restore_state_checkpoint(wrong, str(tmp_path / "sampler_checkpoint.json"))
        assert "missing proposal(s) ['shift_proposal']" in str(err.value)
        assert "register the same proposals in the same order" in str(err.value)

    def test_reordered_weights_raise(self, tmp_path):
        original = _pt(str(tmp_path))
        original.add_custom_jump(_shift_proposal, weight=10)
        original.sample(np.array([0.3, -0.2]), num_iterations=40)
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
        original = _pt(str(tmp_path))
        original.sample(np.array([0.3, -0.2]), num_iterations=40)
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
        _pt(str(tmp_path)).sample(np.array([0.3, -0.2]), num_iterations=100)
        _pt(str(tmp_path), resume=True).sample(np.array([0.3, -0.2]), num_iterations=260)
        assert count_chain_rows(str(tmp_path / "chain_0.bin"), 2 + 4) == 260

    def test_set_checkpoint_state_restores_buffer_size(self):
        ptstate = PTState(ndim=2, ntemps=1, min_temp=1.0, max_temp=1.0)
        rng = np.random.default_rng(0)

        source = ChainStats(ndim=2, pt_state=ptstate, chain_index=0, rng=rng, buffer_size=200)
        source.recursive_update(0, rng.standard_normal((10, 2)))
        arrays, meta = source.get_checkpoint_state()

        target = ChainStats(ndim=2, pt_state=ptstate, chain_index=0, rng=rng, buffer_size=25)
        target.set_checkpoint_state(arrays, meta)

        assert len(target._buffer) == 200
        assert target.buffer_size == len(target._buffer)

    def test_matching_custom_jump_restores(self, tmp_path):
        original = _pt(str(tmp_path))
        original.add_custom_jump(_shift_proposal, weight=10)
        original.sample(np.array([0.3, -0.2]), num_iterations=40)
        save_state_checkpoint(original)
        good = _pt(str(tmp_path))
        good.add_custom_jump(_shift_proposal, weight=10)
        meta = restore_state_checkpoint(good, str(tmp_path / "sampler_checkpoint.json"))
        assert good.short_chain.iteration == original.short_chain.iteration
        assert meta["ndim"] == 2


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


class TestTornWriteDetection:
    def test_npz_without_json_is_ignored(self, tmp_path):
        s = _pt(str(tmp_path))
        s.sample(np.array([0.1, 0.2]), num_iterations=40)
        os.remove(tmp_path / "sampler_checkpoint.json")
        assert os.path.exists(tmp_path / "sampler_checkpoint.npz")
        assert check_for_checkpoint(str(tmp_path)) is None

    def test_mismatched_write_tokens_treated_as_torn(self, tmp_path):
        s = _pt(str(tmp_path))
        s.sample(np.array([0.1, 0.2]), num_iterations=40)
        json_path = tmp_path / "sampler_checkpoint.json"
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        meta["write_token"] = "0" * 32
        json_path.write_text(json.dumps(meta))
        assert check_for_checkpoint(str(tmp_path)) is None
        with pytest.raises(TornCheckpointError):
            load_state_checkpoint(str(json_path))


@pytest.mark.skipif(
    importlib.util.find_spec("coppuccino") is None,
    reason="coppuccino is an optional dependency",
)
class TestFlowProposalState:
    def test_frozen_flag_round_trips(self):
        from impulse.proposals import NormalizingFlowProposal

        prop = NormalizingFlowProposal(min_samples=50, refit_interval=5)
        prop._call_count = 123
        prop._fit_count = 4
        prop._last_fit_at = 100
        prop.freeze_adaptation()
        assert prop.frozen is True

        state = prop.get_checkpoint_state()
        assert json.loads(json.dumps(state)) == state
        assert "flow" not in state

        fresh = NormalizingFlowProposal(min_samples=50, refit_interval=5)
        fresh.set_checkpoint_state(state)
        assert fresh.frozen is True
        assert fresh._call_count == 123
        assert fresh._fit_count == 4
        assert fresh._last_fit_at == 100
        assert fresh.flow is None
