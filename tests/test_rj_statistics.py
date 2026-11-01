"""Statistical correctness of RJPTSampler: stationary distributions against exact answers.

Each check compares a chain average with its exact value, using a batch-means
standard error. The slow cases also run a deliberately broken kernel and
require the same check to fail, so the tolerance is known to have power.
The tests check the default adaptive configuration: ``num_adapt`` is unset,
so AM/SCAM/DE and the parallel-tempering ladder keep adapting throughout.
"""

import math

import numpy as np
import pytest
from rj_helpers import AffineBirthDeath, batch_mean_z, normal_logpdf
from scipy import stats

from impulse import IndependentComponentProposal, RJLayout, RJPTSampler
from impulse.reversible_jump import BirthDeath

SMOKE, FULL = 20_000, 200_000
SMOKE_BOUND, FULL_BOUND = 6.0, 5.0
RUNS = [
    pytest.param(SMOKE, SMOKE_BOUND, id="smoke"),
    pytest.param(FULL, FULL_BOUND, id="full", marks=pytest.mark.slow),
]
POWER = 8.0


def _k_z_scores(ks, probs):
    """Batch-means z of each k indicator, for every k with probability >= 0.005."""
    return np.array(
        [batch_mean_z(ks == k, p) for k, p in enumerate(probs) if p >= 0.005], dtype=float
    )


def _k_report(ks, probs):
    """Observed vs target p(k), for failure messages and the -rA output."""
    observed = np.bincount(ks, minlength=len(probs)) / len(ks)
    return f"p(k) observed {np.round(observed, 4)} target {np.round(probs, 4)}"


class KillLastBirthDeath(BirthDeath):
    """Broken on purpose: always removes the last component."""

    __name__ = "kill_last"

    def _choose_death_slot(self, rng, k):
        return k - 1


class NoSelectionRatioBirthDeath(BirthDeath):
    """Broken on purpose: drops log(d_{k+1} / b_k), which is not 0 at the k limits."""

    __name__ = "no_selection_ratio"

    def _selection_log_ratio(self, k):
        return 0.0


# --- Prior only: lnlike = 0, truncated Poisson(2) on k, U(0, 1)^2 components -------------

PRIOR_K, PRIOR_P = 4, 2
POISSON = stats.poisson.pmf(np.arange(PRIOR_K + 1), 2.0)
POISSON = POISSON / POISSON.sum()
LOG_POISSON = np.log(POISSON)
_LOG_BETA_8_24 = math.lgamma(8) + math.lgamma(24) - math.lgamma(32)


def _zero_lnlike(x, k):
    return 0.0


def _poisson_lnprior(x, k):
    comps = x.reshape(PRIOR_K, PRIOR_P)[:k]
    if np.any((comps < 0) | (comps > 1)):
        return -np.inf
    return float(LOG_POISSON[k])


def _uniform_draw(rng):
    return rng.random(PRIOR_P)


def _uniform_logpdf(u):
    u = np.asarray(u)
    return 0.0 if np.all((u >= 0) & (u <= 1)) else -np.inf


def _mixed_draw(rng):
    return rng.random(PRIOR_P) if rng.random() < 0.5 else rng.beta(8.0, 24.0, PRIOR_P)


def _mixed_logpdf(u):
    u = np.asarray(u, dtype=float)
    if np.any((u < 0) | (u > 1)):
        return -np.inf
    with np.errstate(divide="ignore"):
        log_beta = float(np.sum(7.0 * np.log(u) + 23.0 * np.log1p(-u) - _LOG_BETA_8_24))
    return math.log(0.5 + 0.5 * math.exp(log_beta))


PRIOR_BIRTHS = IndependentComponentProposal(_uniform_draw, _uniform_logpdf)
MIXED_BIRTHS = IndependentComponentProposal(_mixed_draw, _mixed_logpdf)


class CentredBeta:
    """Half uniform, half a Beta centred on the existing components' mean (state-dependent)."""

    def __init__(self, layout):
        self.layout = layout

    def _shape(self, x, k):
        comps = self.layout.components(x)[:k]
        centre = np.clip(comps.mean(axis=0), 0.1, 0.9) if k else np.full(PRIOR_P, 0.5)
        return 20.0 * centre, 20.0 * (1.0 - centre)

    def draw(self, rng, x, k):
        if rng.random() < 0.5:
            return rng.random(PRIOR_P)
        a, b = self._shape(x, k)
        return rng.beta(a, b)

    def logpdf(self, u, x, k):
        u = np.asarray(u, dtype=float)
        if np.any((u < 0) | (u > 1)):
            return -np.inf
        a, b = self._shape(x, k)
        return math.log(0.5 + 0.5 * math.exp(float(np.sum(stats.beta.logpdf(u, a, b)))))


def _run_prior_only(outdir, n_iter, proposal, broken=None, min_components=0):
    sampler = RJPTSampler(
        n_shared=0,
        component_dim=PRIOR_P,
        max_components=PRIOR_K,
        min_components=min_components,
        lnlike=_zero_lnlike,
        lnprior=_poisson_lnprior,
        component_proposal=proposal,
        birth_death_weight=0 if broken else 40,
        ntemps=1,
        seed=5,
        outdir=str(outdir),
        verbose=False,
    )
    if broken is not None:
        sampler.add_model_jump(broken(sampler.layout, proposal, sampler.ntemps), 40)
    sampler.sample(np.full(sampler.layout.ndim, 0.5), n_iter, initial_k=1)
    chain = sampler.load_chain()
    burn = n_iter // 10
    return chain["k"][0, burn:], chain["components"][0, burn:]


def _component_z(comps):
    """z of (count below 0.25) and (sum - 0.5 per value), both 0 under U(0, 1)."""
    active = ~np.isnan(comps[..., 0])
    n_values = PRIOR_P * active.sum(axis=1)
    below = np.where(active[..., None], comps < 0.25, False).sum(axis=(1, 2)) - 0.25 * n_values
    centred = np.where(active[..., None], comps, 0.0).sum(axis=(1, 2)) - 0.5 * n_values
    return batch_mean_z(below, 0.0), batch_mean_z(centred, 0.0)


def _check_prior_only(ks, comps, bound, probs=POISSON):
    z_k = _k_z_scores(ks, probs)
    z_c = np.array(_component_z(comps))
    message = f"k z {np.round(z_k, 2)}; component z {np.round(z_c, 2)}; {_k_report(ks, probs)}"
    print(message)
    assert np.max(np.abs(z_k)) < bound, message
    assert np.max(np.abs(z_c)) < bound, message


@pytest.mark.parametrize("n_iter, bound", RUNS)
@pytest.mark.parametrize("proposal", [PRIOR_BIRTHS, MIXED_BIRTHS], ids=["prior", "mixed"])
def test_prior_only(tmp_path, proposal, n_iter, bound):
    ks, comps = _run_prior_only(tmp_path, n_iter, proposal)
    _check_prior_only(ks, comps, bound)


def test_prior_only_with_a_lower_bound_on_k(tmp_path):
    """min_components = 1: the truncated Poisson renormalized on 1..PRIOR_K (smoke only)."""
    ks, comps = _run_prior_only(tmp_path, SMOKE, PRIOR_BIRTHS, min_components=1)
    target = np.where(np.arange(PRIOR_K + 1) >= 1, POISSON, 0.0)
    target /= target.sum()
    assert ks.min() >= 1
    _check_prior_only(ks, comps, SMOKE_BOUND, probs=target)


@pytest.mark.parametrize("n_iter, bound", RUNS)
def test_state_dependent_component_proposal(tmp_path, n_iter, bound):
    proposal = CentredBeta(RJLayout(0, PRIOR_P, PRIOR_K))
    ks, comps = _run_prior_only(tmp_path, n_iter, proposal)
    _check_prior_only(ks, comps, bound)


def _check_detected(ks, probs):
    z_k = _k_z_scores(ks, probs)
    message = f"k z {np.round(z_k, 2)}; {_k_report(ks, probs)}"
    print(message)
    assert np.max(np.abs(z_k)) > POWER, message


@pytest.mark.slow
def test_prior_only_detects_a_missing_selection_ratio(tmp_path):
    ks, _ = _run_prior_only(tmp_path, FULL, PRIOR_BIRTHS, broken=NoSelectionRatioBirthDeath)
    _check_detected(ks, POISSON)


@pytest.mark.slow
def test_prior_only_detects_a_missing_birth_density(tmp_path):
    """Mixed births with a uniform logpdf: the q != p term is dropped.

    p(k) cannot see this (the birth density integrates to 1 over the prior's
    support), so only the component statistics detect it.
    """
    proposal = IndependentComponentProposal(_mixed_draw, _uniform_logpdf)
    ks, comps = _run_prior_only(tmp_path, FULL, proposal)
    z_c = np.array(_component_z(comps))
    message = f"component z {np.round(z_c, 2)}; {_k_report(ks, POISSON)}"
    print(message)
    assert np.max(np.abs(z_c)) > POWER, message


# --- Analytic posterior on k: y_n ~ N(sum_{i<k} a_i, 1), a_i ~ N(0, 1), uniform k ----------

Y = np.array([2.1, 2.9, 2.4, 2.6])
GAUSS_K = 6


def _log_evidence(k):
    """log N(Y; 0, I + k 11^T) up to a k-independent constant (Sherman-Morrison)."""
    c, n = float(k), len(Y)
    return -0.5 * (math.log1p(n * c) + Y @ Y - c * Y.sum() ** 2 / (1.0 + n * c))


_LOG_Z = np.array([_log_evidence(k) for k in range(GAUSS_K + 1)])
GAUSS_POSTERIOR = np.exp(_LOG_Z - _LOG_Z.max())
GAUSS_POSTERIOR /= GAUSS_POSTERIOR.sum()


def _gauss_lnlike(x, k):
    s = float(np.sum(x[:k]))
    return -0.5 * float(np.sum((Y - s) ** 2))


def _gauss_lnprior(x, k):
    return normal_logpdf(x[:k])  # uniform p(k) is a constant


def _normal_draw(rng):
    return rng.standard_normal(1)


STD_NORMAL = IndependentComponentProposal(_normal_draw, normal_logpdf)


def _run_gauss(outdir, n_iter, broken=None, model_jump=None, log_jacobian=0.0):
    sampler = RJPTSampler(
        n_shared=0,
        component_dim=1,
        max_components=GAUSS_K,
        lnlike=_gauss_lnlike,
        lnprior=_gauss_lnprior,
        component_proposal=STD_NORMAL,
        birth_death_weight=0 if (broken or model_jump) else 40,
        ntemps=3,
        seed=9,
        outdir=str(outdir),
        verbose=False,
    )
    if broken is not None:
        sampler.add_model_jump(broken(sampler.layout, STD_NORMAL, sampler.ntemps), 40)
    if model_jump is not None:
        sampler.add_model_jump(model_jump, 40, log_jacobian=log_jacobian)
    sampler.sample(np.zeros(GAUSS_K), n_iter, initial_k=1)
    return sampler.load_chain()["k"][0, n_iter // 10 :]


def _check_gauss(ks, bound):
    z_k = _k_z_scores(ks, GAUSS_POSTERIOR)
    message = f"k z {np.round(z_k, 2)}; {_k_report(ks, GAUSS_POSTERIOR)}"
    print(message)
    assert np.max(np.abs(z_k)) < bound, message


def test_analytic_posterior_is_spread_over_k():
    assert GAUSS_POSTERIOR[0] < 1e-3 and GAUSS_POSTERIOR[GAUSS_K] > 0.15


@pytest.mark.parametrize("n_iter, bound", RUNS)
def test_analytic_k_posterior(tmp_path, n_iter, bound):
    ks = _run_gauss(tmp_path, n_iter)
    _check_gauss(ks, bound)


@pytest.mark.slow
def test_analytic_k_posterior_detects_kill_last(tmp_path):
    ks = _run_gauss(tmp_path, FULL, broken=KillLastBirthDeath)
    _check_detected(ks, GAUSS_POSTERIOR)


# --- Constant Jacobian: a user model jump theta = 0 + 3u, log|J| = log 3 ------------------


@pytest.mark.parametrize("n_iter, bound", RUNS)
def test_constant_jacobian_model_jump(tmp_path, n_iter, bound):
    jump = AffineBirthDeath(mu=[0.0], sigma=[3.0])
    ks = _run_gauss(tmp_path, n_iter, model_jump=jump, log_jacobian=math.log(3.0))
    _check_gauss(ks, bound)


@pytest.mark.slow
def test_constant_jacobian_detects_a_missing_jacobian(tmp_path):
    jump = AffineBirthDeath(mu=[0.0], sigma=[3.0])
    ks = _run_gauss(tmp_path, FULL, model_jump=jump, log_jacobian=0.0)
    _check_detected(ks, GAUSS_POSTERIOR)


# --- component_redraw alone at fixed k = 3 ------------------------------------------------


@pytest.mark.parametrize("n_iter, bound", RUNS)
def test_component_redraw_alone_at_fixed_k(tmp_path, n_iter, bound):
    sampler = RJPTSampler(
        n_shared=0,
        component_dim=1,
        max_components=3,
        min_components=3,
        lnlike=_gauss_lnlike,
        lnprior=_gauss_lnprior,
        component_proposal=STD_NORMAL,
        am_weight=0,
        scam_weight=0,
        de_weight=0,
        component_redraw_weight=1,
        ntemps=1,
        seed=13,
        outdir=str(tmp_path),
        verbose=False,
    )
    sampler.sample(np.zeros(3), n_iter, initial_k=3)
    # Zero total weight would fall back to uniform selection; pin that only the redraw ran.
    calls = {name: r["calls"] for name, r in sampler.proposal_acceptance_rates().items()}
    print(f"proposal calls: {calls}")
    assert calls == {"rj_am": 0, "rj_scam": 0, "rj_de": 0, "component_redraw": n_iter}
    total = sampler.load_chain()["samples"][0, n_iter // 10 :].sum(axis=1)
    var = 1.0 / (1.0 / 3.0 + len(Y))  # posterior of sum(a): prior N(0, 3), 4 unit-noise data
    mean = var * Y.sum()
    z = np.array([batch_mean_z(total, mean), batch_mean_z((total - mean) ** 2, var)])
    message = f"z of (mean, variance) of sum(a): {np.round(z, 2)}"
    print(message)
    assert np.max(np.abs(z)) < bound, message
