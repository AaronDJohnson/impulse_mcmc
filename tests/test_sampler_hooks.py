"""PTSampler extension seams used by RJPTSampler."""

import numpy as np
import pytest

from impulse import CheckpointMismatchError, PTSampler


def _lnlike(x):
    return -0.5 * float(np.sum(x**2))


def _lnprior(x):
    return 0.0 if np.all(np.abs(x) < 10) else -np.inf


class _HookedSampler(PTSampler):
    def __init__(self, *args, **kwargs):
        self.calls = []
        super().__init__(*args, **kwargs)

    def _build_chain_stats(self, groups, sample_cov, sample_mean, buffer_size, buffer_thin):
        self.calls.append("chain_stats")
        return super()._build_chain_stats(groups, sample_cov, sample_mean, buffer_size, buffer_thin)

    def _build_proposal_bundle(self, am_weight, scam_weight, de_weight, de_min_fill):
        self.calls.append(("bundle", am_weight, scam_weight, de_weight, de_min_fill))
        return super()._build_proposal_bundle(am_weight, scam_weight, de_weight, de_min_fill)


def test_build_hooks_are_called_in_order(tmp_path):
    sampler = _HookedSampler(
        ndim=2,
        lnlike=_lnlike,
        lnprior=_lnprior,
        ntemps=2,
        outdir=str(tmp_path),
        am_weight=1,
        scam_weight=2,
        de_weight=3,
        de_min_fill=50,
        verbose=False,
    )
    assert sampler.calls == ["chain_stats", ("bundle", 1, 2, 3, 50)]
    assert sampler.multi_chain_stats.ntemps == 2
    names = [p.__name__ for p in sampler.proposal_bundle.jump_proposals[0].proposal_list]
    assert names == ["am", "scam", "de"]


class _ExtraCheckSampler(PTSampler):
    token = "a"

    def _capture_checkpoint_state(self):
        arrays, meta = super()._capture_checkpoint_state()
        meta["extra_token"] = self.token
        return arrays, meta

    def _verify_extra_metadata(self, meta):
        if meta.get("extra_token") != self.token:
            raise CheckpointMismatchError("extra_token mismatch")


def _extra_kwargs(tmp_path):
    return dict(
        ndim=2,
        lnlike=_lnlike,
        lnprior=_lnprior,
        ntemps=2,
        outdir=str(tmp_path),
        save_freq=50,
        seed=3,
        verbose=False,
    )


def test_verify_extra_metadata_rejects_a_mismatch(tmp_path):
    _ExtraCheckSampler(**_extra_kwargs(tmp_path)).sample(np.zeros(2), 120)
    resumed = _ExtraCheckSampler(resume=True, **_extra_kwargs(tmp_path))
    resumed.token = "b"
    with pytest.raises(CheckpointMismatchError, match="extra_token"):
        resumed.sample(np.zeros(2), 200)


def test_verify_extra_metadata_accepts_a_match(tmp_path):
    _ExtraCheckSampler(**_extra_kwargs(tmp_path)).sample(np.zeros(2), 120)
    resumed = _ExtraCheckSampler(resume=True, **_extra_kwargs(tmp_path))
    resumed.sample(np.zeros(2), 200)
    assert resumed.load_chain()["samples"].shape == (2, 200, 2)
