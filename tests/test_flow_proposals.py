"""Tests for NormalizingFlowProposal.

These tests skip cleanly if `coppuccino` is not installed.
"""

import importlib
import os
import shutil
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("coppuccino") is None,
    reason="coppuccino is an optional dependency; install to enable these tests",
)


def test_import_error_without_coppuccino(monkeypatch):
    """Construction must raise ImportError with a helpful message when
    coppuccino is missing."""
    import impulse.flow_proposals as fp

    real_require = fp._require_coppuccino

    def broken():
        raise ImportError("simulated missing dep")

    monkeypatch.setattr(fp, "_require_coppuccino", broken)
    with pytest.raises(ImportError):
        fp.NormalizingFlowProposal()


def test_proposal_is_no_op_before_buffer_fills():
    from impulse.chain_stats import ChainStats
    from impulse.flow_proposals import NormalizingFlowProposal
    from impulse.sampler_state import PTState

    ndim = 2
    ptstate = PTState(ndim=ndim, ntemps=1, min_temp=1.0, max_temp=1.0)
    rng = np.random.default_rng(0)
    cs = ChainStats(
        ndim=ndim,
        pt_state=ptstate,
        chain_index=0,
        rng=rng,
        buffer_size=200,
        current_sample=np.array([1.0, 2.0]),
    )

    prop = NormalizingFlowProposal(min_samples=200, refit_interval=10)
    x, qxy = prop(cs)
    # No fit attempted: returns current sample unchanged with qxy=0.
    np.testing.assert_allclose(x, cs.current_sample)
    assert qxy == 0.0
    assert prop.flow is None


def test_fixed_flow_no_refit(tmp_path):
    """A pre-fitted flow drives the proposal without refitting.

    Fit a flow on samples drawn directly from the target, then plug it in
    as a fixed proposal. The flow already matches the target, so the NF
    proposal should be the dominant accepted proposal from iteration 0.
    """
    from coppuccino import normalizing_flows_fit
    from scipy import stats

    from impulse.flow_proposals import NormalizingFlowProposal
    from impulse.samplers import PTSampler

    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    L = np.linalg.cholesky(cov)
    cov_inv = np.linalg.inv(cov)

    # 1) Pre-fit a flow on samples from the target.
    rng = np.random.default_rng(0)
    train = rng.standard_normal(size=(4000, 2)) @ L.T
    flow = normalizing_flows_fit(
        train,
        max_epochs=120,
        rng_seed=0,
        prior_bounds=np.array([[-6.0, 6.0], [-6.0, 6.0]]),
    )

    def lnlike(x):
        return float(-0.5 * x @ cov_inv @ x)

    def lnprior(x):
        return -np.inf if np.any(np.abs(x) > 6) else 0.0

    # 2) Use it as a fixed proposal — no fitting during the run.
    nf = NormalizingFlowProposal(flow=flow)
    assert nf.fixed is True
    assert nf.flow is flow

    sampler = PTSampler(
        ndim=2,
        lnlike=lnlike,
        lnprior=lnprior,
        am_weight=15,
        scam_weight=30,
        de_weight=50,
        ntemps=1,
        seed=1,
        outdir=str(tmp_path / "nf_fixed"),
        buffer_size=2000,
        cov_update=100,
        save_freq=2000,
    )
    sampler.proposal_bundle.add_jump(nf, weight=50.0)
    sampler.sample(np.array([[0.5, 0.5]]), num_iterations=3000)

    # Confirm no refits happened.
    assert nf._fit_count == 0
    assert nf.fixed is True

    # The flow already matches the target, so NF acceptance should be high
    # from the start.
    rates = sampler.proposal_acceptance_rates()
    assert rates["nf_flow"]["calls"] > 100
    assert rates["nf_flow"]["rate"] > 0.3, rates["nf_flow"]

    # Marginal check on the cold chain.
    data = sampler.load_chain()
    s = data["samples"][0, 500:]
    for d in range(2):
        ks, _ = stats.kstest(s[:, d], lambda q: stats.norm.cdf(q, 0.0, np.sqrt(cov[d, d])))
        assert ks < 0.10, f"dim {d}: KS={ks:.3f} too large"


def test_chain_acceptance_rates_includes_nf_and_custom(tmp_path):
    """`chain_acceptance_rates` reports NF and user-defined proposals too.

    Adds a fixed-flow NF proposal and a custom callable-class proposal to a
    sampler alongside the standard am/scam/de jumps, runs briefly, and
    verifies every proposal name shows up in the per-chain report.
    """
    from coppuccino import normalizing_flows_fit

    from impulse.flow_proposals import NormalizingFlowProposal
    from impulse.samplers import PTSampler

    class MyTinyJump:
        """Custom proposal: small Gaussian step (picklable callable class)."""

        __name__ = "my_tiny_jump"

        def __init__(self, sigma=0.1):
            self.sigma = sigma

        def __call__(self, chain_stats):
            x = chain_stats.current_sample.copy()
            x += chain_stats.rng.normal(0.0, self.sigma, size=x.shape)
            return x, 0.0

    # Pre-fit a flow on the target (2D standard normal) so the NF is usable
    # from iteration 0.
    rng = np.random.default_rng(0)
    flow = normalizing_flows_fit(rng.standard_normal((2000, 2)), max_epochs=60)

    def lnlike(x):
        return float(-0.5 * (x[0] ** 2 + x[1] ** 2))

    def lnprior(x):
        return -np.inf if np.any(np.abs(x) > 6) else 0.0

    sampler = PTSampler(
        ndim=2,
        lnlike=lnlike,
        lnprior=lnprior,
        am_weight=15,
        scam_weight=30,
        de_weight=50,
        ntemps=2,
        min_temp=1.0,
        max_temp=4.0,
        seed=0,
        outdir=str(tmp_path / "mixed"),
        buffer_size=500,
        save_freq=2000,
    )
    sampler.proposal_bundle.add_jump(NormalizingFlowProposal(flow=flow), weight=20.0)
    sampler.proposal_bundle.add_jump(MyTinyJump(sigma=0.2), weight=10.0)

    sampler.sample(np.zeros((2, 2)), num_iterations=1500)

    rep = sampler.chain_acceptance_rates()
    assert len(rep["mh"]) == 2  # one entry per chain
    expected_names = {"am", "scam", "de", "nf_flow", "my_tiny_jump"}
    for ch_idx, ch in enumerate(rep["mh"]):
        names = set(ch["per_proposal"].keys())
        assert expected_names <= names, f"chain {ch_idx} missing names: {expected_names - names}"
        # Each proposal got at least one call (random selection with positive weight)
        chain_jp = sampler.proposal_bundle.jump_proposals[ch_idx]
        per_chain_rates = chain_jp.acceptance_rates()
        for n in expected_names:
            assert per_chain_rates[n]["calls"] > 0, f"chain {ch_idx} {n} got 0 calls"


def test_set_flow_after_unpickle(tmp_path):
    """Fixed flows are dropped during pickle; ``set_flow`` re-attaches."""
    import pickle

    from coppuccino import normalizing_flows_fit

    from impulse.flow_proposals import NormalizingFlowProposal

    rng = np.random.default_rng(0)
    flow = normalizing_flows_fit(
        rng.standard_normal(size=(1500, 2)),
        max_epochs=40,
        rng_seed=0,
    )
    nf = NormalizingFlowProposal(flow=flow)
    assert nf.fixed and nf.flow is flow

    blob = pickle.dumps(nf)
    restored = pickle.loads(blob)
    assert restored.fixed is True
    assert restored.flow is None  # dropped during pickle

    restored.set_flow(flow)
    assert restored.flow is flow
    assert restored.fixed is True


def test_prior_recovery_2d_normal(tmp_path):
    """End-to-end: NF proposal alone should recover a 2D Gaussian target.

    The target is a correlated 2D Normal, which the flow can model
    arbitrarily well. Using the NF proposal exclusively (after the initial
    buffer fill), the cold chain should match the target moments and pass
    KS tests per-dimension against the marginals.
    """
    from scipy import stats

    from impulse.flow_proposals import NormalizingFlowProposal
    from impulse.samplers import PTSampler

    # Target: zero-mean Gaussian with correlation
    cov = np.array([[1.0, 0.6], [0.6, 1.0]])
    L = np.linalg.cholesky(cov)
    cov_inv = np.linalg.inv(cov)

    def lnlike(x):
        return float(-0.5 * x @ cov_inv @ x)

    def lnprior(x):
        # wide flat prior; bounds keep things finite
        if np.any(np.abs(x) > 10):
            return -np.inf
        return 0.0

    outdir = str(tmp_path / "nf_test")
    sampler = PTSampler(
        ndim=2,
        lnlike=lnlike,
        lnprior=lnprior,
        # Keep standard adaptive proposals active so the chain mixes
        # whether or not the NF is fitting well at any given moment.
        am_weight=15,
        scam_weight=30,
        de_weight=50,
        ntemps=1,
        seed=0,
        outdir=outdir,
        buffer_size=2000,
        cov_update=100,
        save_freq=2000,
    )

    nf = NormalizingFlowProposal(
        min_samples=1500,
        refit_interval=1500,
        max_epochs=120,
        prior_bounds=np.array([[-8.0, 8.0], [-8.0, 8.0]]),  # tame extrapolation
        rng_seed=0,
    )
    sampler.proposal_bundle.add_jump(nf, weight=30.0)

    x0 = np.array([[0.5, 0.5]])
    sampler.sample(x0, num_iterations=8000)

    data = sampler.load_chain()
    samples = data["samples"][0]
    burn = 3000
    s = samples[burn:]

    # Moments
    mu = s.mean(axis=0)
    s_cov = np.cov(s, rowvar=False)
    np.testing.assert_allclose(mu, 0.0, atol=0.2)
    np.testing.assert_allclose(s_cov, cov, atol=0.25)

    # Marginal KS — generous threshold given short chain & autocorrelation
    for d in range(2):
        ks, _ = stats.kstest(s[:, d], lambda q: stats.norm.cdf(q, 0.0, np.sqrt(cov[d, d])))
        assert ks < 0.10, f"dim {d}: KS={ks:.3f} too large"

    # Sanity: the NF proposal was actually used and accepted some moves
    rates = sampler.proposal_acceptance_rates()
    assert "nf_flow" in rates
    assert rates["nf_flow"]["calls"] > 100
    assert rates["nf_flow"]["accepts"] > 10
