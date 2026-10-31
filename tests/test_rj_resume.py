"""RJPTSampler reproducibility and checkpoint/resume."""

import math

import numpy as np
import pytest
from rj_helpers import AffineBirthDeath, assert_rj_chains_identical

from impulse import CheckpointMismatchError, IndependentComponentProposal, PTSampler, RJPTSampler
from impulse.resume import check_for_checkpoint

P = 2
K = 3
N_SHARED = 1
D = N_SHARED + K * P
N_FIRST, TOTAL = 300, 500


def _draw(rng):
    return rng.random(P)


def _logpdf(u):
    u = np.asarray(u)
    return 0.0 if np.all((u >= 0) & (u <= 1)) else -np.inf


UNIFORM = IndependentComponentProposal(_draw, _logpdf)


class OtherUniform(IndependentComponentProposal):
    pass


class CountingUniform(IndependentComponentProposal):
    """Stateful component proposal: counts draws and freezes."""

    def __init__(self):
        super().__init__(_draw, _logpdf)
        self.draws = 0
        self.freezes = 0
        self.draws_at_freeze = None

    def draw(self, rng, x, k):
        self.draws += 1
        return super().draw(rng, x, k)

    def get_checkpoint_state(self):
        return {"draws": self.draws}

    def set_checkpoint_state(self, state):
        self.draws = int(state["draws"])

    def freeze_adaptation(self):
        self.freezes += 1
        self.draws_at_freeze = self.draws


def _lnlike(x, k):  # works for any layout: uses only the active (non-NaN) entries
    active = x[~np.isnan(x)]
    # the 1.5 * k bonus keeps p(k) roughly flat over 0..3, so redraws and wraps really run
    return -0.5 * float(np.sum(((active - 0.5) / 0.2) ** 2)) + 1.5 * k


def _lnprior(x, k):
    active = x[~np.isnan(x)]
    return 0.0 if np.all((active >= 0) & (active <= 1)) else -np.inf


def _make(outdir, resume=False, log_jacobian=2 * math.log(0.2), **kwargs):
    options = dict(
        n_shared=N_SHARED,
        n_component=P,
        max_components=K,
        lnlike=_lnlike,
        lnprior=_lnprior,
        component_proposal=UNIFORM,
        component_periodic={1: 1.0},
        component_redraw_weight=5,
        ntemps=3,
        seed=11,
        outdir=str(outdir),
        resume=resume,
        save_freq=100,
        cov_update=25,
        buffer_size=150,
        buffer_thin=2,
        de_min_fill=20,
        verbose=False,
    )
    options.update(kwargs)
    sampler = RJPTSampler(**options)
    sampler.add_model_jump(
        AffineBirthDeath(mu=[0.5, 0.5], sigma=[0.2, 0.2], support=(0.0, 1.0)),
        weight=10,
        log_jacobian=log_jacobian,
    )
    return sampler


X0 = np.full(D, 0.5)


def test_same_seed_gives_the_same_chain(tmp_path):
    chains = []
    for sub in ("a", "b"):
        sampler = _make(tmp_path / sub)
        sampler.sample(X0, 400, initial_k=1)
        chains.append(sampler.load_chain())
    assert_rj_chains_identical(*chains)
    assert len(np.unique(chains[0]["k"])) > 1


@pytest.mark.parametrize("chain_format", ["binary", "text"])
def test_resume_is_bit_identical(tmp_path, chain_format):
    full = _make(tmp_path / "full", chain_format=chain_format)
    full.sample(X0, TOTAL, initial_k=1)
    first = _make(tmp_path / "split", chain_format=chain_format)
    first.sample(X0, N_FIRST, initial_k=1)
    assert check_for_checkpoint(str(tmp_path / "split")) is not None
    resumed = _make(tmp_path / "split", resume=True, chain_format=chain_format)
    resumed.sample(X0, TOTAL, initial_k=1)
    assert_rj_chains_identical(full.load_chain(), resumed.load_chain())
    assert resumed.proposal_acceptance_rates() == full.proposal_acceptance_rates()


@pytest.mark.parametrize(
    "change, match",
    [
        (dict(n_shared=2), "n_shared mismatch"),
        (dict(n_component=3), "n_component mismatch"),
        (dict(max_components=4), "max_components mismatch"),
        (dict(min_components=1), "min_components mismatch"),
        (dict(component_proposal=OtherUniform(_draw, _logpdf)), "component_proposal mismatch"),
        (dict(log_jacobian=0.0), "model jump mismatch"),
    ],
    ids=[
        "n_shared",
        "n_component",
        "max_components",
        "min_components",
        "component_proposal",
        "log_jacobian",
    ],
)
def test_changed_settings_refuse_to_resume(tmp_path, change, match):
    _make(tmp_path).sample(X0, 250, initial_k=1)
    resumed = _make(tmp_path, resume=True, **change)
    with pytest.raises(CheckpointMismatchError, match=match):
        resumed.sample(np.full(resumed.layout.ndim, 0.5), 400, initial_k=1)


def test_ptsampler_checkpoint_does_not_resume_into_rjptsampler(tmp_path):
    def lnlike(x):
        return _lnlike(x[:-1], int(x[-1]))

    def lnprior(x):
        return 0.0 if np.all((x >= 0) & (x <= 1)) else -np.inf

    PTSampler(
        ndim=D + 1,
        lnlike=lnlike,
        lnprior=lnprior,
        ntemps=3,
        outdir=str(tmp_path),
        save_freq=100,
        verbose=False,
    ).sample(np.r_[X0, 1.0], 250)
    with pytest.raises(CheckpointMismatchError, match="'PTSampler'"):
        _make(tmp_path, resume=True).sample(X0, 400, initial_k=1)


def test_component_proposal_state_and_freeze(tmp_path):
    full = CountingUniform()
    _make(tmp_path / "full", component_proposal=full, num_adapt=150).sample(X0, TOTAL, initial_k=1)
    assert full.freezes == 1 and full.draws > 0
    # the freeze happens at num_adapt, not when sample() starts: a run of exactly num_adapt
    # iterations has made the same draws by then
    upto = CountingUniform()
    _make(tmp_path / "upto", component_proposal=upto, num_adapt=150).sample(X0, 150, initial_k=1)
    assert full.draws_at_freeze == upto.draws > 0
    _make(tmp_path / "split", component_proposal=CountingUniform(), num_adapt=150).sample(
        X0, N_FIRST, initial_k=1
    )
    again = CountingUniform()
    _make(tmp_path / "split", component_proposal=again, num_adapt=150, resume=True).sample(
        X0, TOTAL, initial_k=1
    )
    assert again.draws == full.draws
    assert again.freezes == 1


def test_resume_restores_fallback_covariance_over_constructor_argument(tmp_path):
    """The checkpointed fallback covariance wins over a different ``sample_cov`` on resume."""
    full = _make(tmp_path / "full")
    full.sample(X0, TOTAL, initial_k=1)
    _make(tmp_path / "split").sample(X0, N_FIRST, initial_k=1)
    resumed = _make(
        tmp_path / "split", resume=True, sample_cov=4.0 * np.eye(D), sample_mean=np.zeros(D)
    )
    resumed.sample(X0, TOTAL, initial_k=1)
    assert_rj_chains_identical(full.load_chain(), resumed.load_chain())
    assert resumed.proposal_acceptance_rates() == full.proposal_acceptance_rates()
