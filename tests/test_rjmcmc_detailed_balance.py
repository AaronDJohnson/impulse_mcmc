"""
Detailed-balance regression tests for the RJMCMC birth/death proposals.

These tests pin down the model-index posterior produced by the birth/death
moves against an *analytically known* target, so that a future change to the
acceptance ratio that re-introduces a trans-dimensional bias is caught.

Construction
------------
With a likelihood that depends only on the model index, ``logL = c[nmodel]``,
proper normalized per-source priors, and the default (uniform) model-index
prior, the marginal posterior over the model index is analytic::

    P(nmodel = k)  proportional to  exp(c[k])

(the continuous per-source priors integrate to one regardless of their shape).
A birth/death chain that satisfies detailed balance must reproduce this,
including the uniform case ``c == 0`` -> ``P(k) = 1 / num_sources``.

Move selection goes through the PRODUCTION mechanism: a ``JumpProposals``
mixture with *constant* selection weights containing the combined
``BirthDeathProposal`` kernel (exactly how ``PTSampler.from_rjmcmc`` registers
it).  This is essential coverage: the sub-proposals' ``qxy`` terms contain the
``prob_schedule`` selection ratio, which is only the correct Hastings factor
when birth vs death really is selected with the schedule probabilities.
Registering birth and death as SEPARATE constant-weight jumps (the pre-fix
production wiring) selects them 50/50 independent of the model index and
forces ``pi(boundary)/pi(interior) = 2`` under the default schedule: the K=3
flat-prior stationary distribution comes out ``[0.4, 0.2, 0.4]`` instead of
uniform.  (K=2 is insensitive — both states are boundaries — so the uniform
cases below use K >= 3.)

Historical note: an earlier bug put an unmatched ``1 / (nmodel + 2)`` factor
in the birth acceptance (and ``nmodel + 1`` in death), biasing the posterior
toward fewer sources; ``test_birth_death_qxy_has_no_source_count_term``
guards that.  The non-flat-prior cases additionally guard the cancellation of
the overwritten slot's full-prior contribution.

A second family of tests uses a PARAMETER-DEPENDENT exchangeable likelihood
(see ``test_full_mixture_parameter_dependent_likelihood``): a model-index-only
likelihood cannot detect inactive-slot memory effects, where a
value-preserving death leaves the killed source's posterior-distributed
parameters in the inactive slot and a later ``nmodel_jump`` re-activates them
with ``qxy = 0`` — valid only for prior-distributed inactive slots.

A third family pins the KILL-LAST death.  A death that kills a victim chosen
uniformly among the active sources (compacting the last active source into
the hole) while its ``qxy`` carries no kill-choice factor is NOT the reverse
of the append-last birth: the multiset-cancellation argument balances only
one of the two flow pairings, leaving a residual stationary bias toward
fewer sources.  The defect is parameter-dependent and configuration
sensitive: K=3 tilted configs are INSENSITIVE (predicted residual ~9e-5)
while K=4 with ``a=2, b=-1`` shows maxerr ~0.003 (z up to 5.6 across seeded
validation runs).  Coverage is two-fold: an EXACT finite-state enumeration
of the real kernels (``test_exact_enumeration_stationarity`` — deterministic,
machine-precision, runs in seconds) and a high-power Monte Carlo run of the
sensitive K=4 config (``test_full_mixture_k4_kill_choice_bias``, marked
slow).

Finally, ``test_full_mixture_with_early_de`` runs the mixture with the
min-fill-gated DE move (``EarlyDE``) that ``PTSampler.from_rjmcmc`` now
registers by default, guarding the newest member of the production mixture
(see that test's docstring for why it cannot bias the model-index marginal).
"""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from impulse.chain_stats import ChainStats
from impulse.proposals import (
    JumpProposals,
    make_early_de,
    make_source_swap_proposal,
)
from impulse.rjmcmc import BirthDeathProductSpace
from impulse.rjmcmc_proposals import (
    BirthProposal,
    DeathProposal,
    NmodelJump,
    default_birth_death_probs,
    make_birth_death_proposal,
)
from impulse.sampler_state import PTState


class _Stats:
    """Minimal ChainStats stand-in: the proposals only use rng + current_sample."""

    def __init__(self, rng, sample):
        self.rng = rng
        self.current_sample = sample


class _Noop:
    """Constant-weight filler standing in for the fixed-dimension proposals
    (am/scam/de/swap): never changes the model index, trivially symmetric."""

    __name__ = "noop"

    def __call__(self, chain_stats):
        return chain_stats.current_sample.copy(), 0.0


# --- per-source priors (proper & normalized on [0, 1]) ---


def _flat_logpdf(p):
    p = np.asarray(p, float)
    return 0.0 if np.all((p >= 0.0) & (p <= 1.0)) else -np.inf


def _beta_logpdf(p):
    """Beta(2, 2): pdf = 6 x (1 - x)."""
    p = np.asarray(p, float)
    if np.any((p <= 0.0) | (p >= 1.0)):
        return -np.inf
    return float(np.sum(np.log(6.0) + np.log(p) + np.log(1.0 - p)))


PRIORS = {
    "flat": (_flat_logpdf, lambda rng, n: rng.uniform(0.0, 1.0, size=n)),
    "beta": (_beta_logpdf, lambda rng, n: rng.beta(2.0, 2.0, size=n)),
}


def _run_birth_death(num_sources, num_params, c, prior, n_iter, seed):
    """Chain on logL = c[nmodel]; moves selected through JumpProposals.

    The combined birth-death kernel competes with a no-op filler at constant
    weights, exactly as in production where it competes with the
    fixed-dimension proposals.
    """
    src_logpdf, src_draw = PRIORS[prior]
    rng = np.random.default_rng(seed)

    # supply the single-source prior density so the birth move can cancel the
    # overwritten slot's prior contribution (BirthDeathProductSpace wires this too).
    log_prior_density = None if prior == "flat" else src_logpdf
    kernel = make_birth_death_proposal(
        num_params,
        num_sources,
        draw_from_prior=lambda r: src_draw(r, num_params),
        log_prior_density=log_prior_density,
    )

    ndim = num_sources * num_params + 1
    ptstate = PTState(ndim=ndim, ntemps=1, min_temp=1.0, max_temp=1.0)
    cs = ChainStats(ndim=ndim, pt_state=ptstate, chain_index=0, rng=rng, buffer_size=50)
    jumps = JumpProposals(cs)
    # constant selection weights, independent of the model index — the
    # production path that the pre-fix separate birth/death jumps got wrong
    jumps.add_jump(kernel, 50)
    jumps.add_jump(_Noop(), 50)

    def logpost(z):
        k = int(round(z[-1]))
        if k < 0 or k >= num_sources:
            return -np.inf
        lp = 0.0
        for i in range(num_sources):  # full prior over all slots
            lp += src_logpdf(z[i * num_params : (i + 1) * num_params])
        return -np.inf if not np.isfinite(lp) else float(c[k]) + lp

    z = np.empty(ndim)
    z[:-1] = src_draw(rng, num_sources * num_params)
    z[-1] = 0
    lp = logpost(z)

    counts = np.zeros(num_sources)
    for _ in range(n_iter):
        y, qxy = jumps(SimpleNamespace(positions=z.reshape(1, -1)))
        ly = logpost(y)
        if np.log(rng.random()) < (ly - lp) + qxy:
            z, lp = y, ly
        # keep ACTIVE slots prior-distributed (as NUTS/MH would); inactive slots
        # are deliberately NOT refreshed, matching the real sampler where birth
        # supplies a fresh draw itself.
        k = int(round(z[-1]))
        z = z.copy()
        z[: (k + 1) * num_params] = src_draw(rng, (k + 1) * num_params)
        lp = logpost(z)
        counts[int(round(z[-1]))] += 1
    return counts / counts.sum()


@pytest.mark.parametrize(
    "prior",
    [
        "flat",
        pytest.param("beta", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "num_sources",
    [
        3,
        pytest.param(4, marks=pytest.mark.slow),
        pytest.param(5, marks=pytest.mark.slow),
    ],
)
def test_uniform_model_posterior(num_sources, prior):
    """Constant likelihood => model-index posterior must be uniform.

    Detects the constant-weight selection bias: with birth/death registered
    as separate jumps this comes out [0.4, 0.2, 0.4] for K=3 (K=2 would not
    detect it — both of its states are schedule boundaries).
    """
    probs = _run_birth_death(
        num_sources,
        num_params=2,
        c=[0.0] * num_sources,
        prior=prior,
        n_iter=250_000,
        seed=12345,
    )
    target = np.full(num_sources, 1.0 / num_sources)
    assert (
        np.max(np.abs(probs - target)) < 0.02
    ), f"[{prior}] biased model posterior {probs} vs uniform {target}"


@pytest.mark.parametrize(
    "prior",
    [
        "flat",
        pytest.param("beta", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize("num_sources", [2, 3])
def test_nonuniform_model_posterior(num_sources, prior):
    """logL = c[k] => posterior must match exp(c) (correct Bayes factors)."""
    c = list(np.arange(num_sources) * 0.7)
    probs = _run_birth_death(
        num_sources,
        num_params=2,
        c=c,
        prior=prior,
        n_iter=250_000,
        seed=2024,
    )
    target = np.exp(np.asarray(c))
    target /= target.sum()
    assert (
        np.max(np.abs(probs - target)) < 0.02
    ), f"[{prior}] biased model posterior {probs} vs analytic {target}"


def test_birth_death_qxy_has_no_source_count_term():
    """For a flat prior the birth/death log-Hastings factor is the move ratio only."""
    num_params, num_sources = 2, 4
    birth = BirthProposal(
        num_params,
        num_sources,
        draw_from_prior=lambda r: r.uniform(0.0, 1.0, size=num_params),
    )
    death = DeathProposal(
        num_params,
        num_sources,
        draw_from_prior=lambda r: r.uniform(0.0, 1.0, size=num_params),
    )
    rng = np.random.default_rng(0)

    z = np.zeros(num_sources * num_params + 1)
    z[-1] = 1  # interior model index (schedule gives p_birth = p_death = 0.5)
    _, qxy_b = birth(_Stats(rng, z))
    # p_death(2) = p_birth(1) = 0.5  ->  log(0.5) - log(0.5) = 0, no -log(k+2)
    assert qxy_b == pytest.approx(0.0, abs=1e-12)

    z2 = np.zeros(num_sources * num_params + 1)
    z2[-1] = 2
    _, qxy_d = death(_Stats(rng, z2))
    # p_birth(1) = p_death(2) = 0.5  ->  log(0.5) - log(0.5) = 0, no +log(k+1)
    assert qxy_d == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Parameter-dependent exchangeable likelihood through the FULL production
# mixture: combined birth-death + nmodel_jump + source_swap + exact
# within-model Gibbs, all selected with constant weights via JumpProposals.
#
# Construction (one parameter per source, prior Uniform[0, 1] per slot):
#
#     logL_m(t_0..t_m) = sum_i (a * t_i + b)                (exchangeable)
#     c = int_0^1 exp(a*t + b) dt = e^b (e^a - 1) / a
#     P(nmodel = m)  proportional to  c^(m+1)               (uniform m prior)
#
# This is the strong regression for the inactive-slot memory bug: a
# value-preserving death deposits the killed source's POSTERIOR-distributed
# parameters in the vacated (inactive) slot, and nmodel_jump later
# re-activates the stored values with qxy = 0, which is only valid when
# inactive slots are PRIOR-distributed.  The model-index-only likelihoods
# above cannot see this (their acceptance never reads the slot values), so
# this section drives a likelihood that does.  Pre-fix bias for K=3 was
# ~[+0.04 more-source weight] (maxerr ~0.04 vs analytic across seeds).
#
# The same harness, at the sensitive config (K=4, a=2, b=-1), also detects
# the uniform-kill death's missing kill-choice factor (residual maxerr
# ~0.003 toward fewer sources); see test_full_mixture_k4_kill_choice_bias.
# ---------------------------------------------------------------------------

_TILT = 3.0


class _ActiveGibbs:
    """Exact conditional refresh of the ACTIVE slots for the exchangeable
    likelihood ``logL = a * sum(t) + b * n``, framed as an independence MH
    proposal.

    Active slots are proposed from the exact full conditional
    ``g(t) propto exp(a * t)`` on [0, 1] (inverse CDF), and ``qxy``
    contains the independence-proposal density ratio, which makes the
    total acceptance log-ratio exactly zero: a Gibbs move.
    """

    __name__ = "active_gibbs"

    def __init__(self, a=_TILT):
        self.a = float(a)

    def __call__(self, chain_stats):
        rng = chain_stats.rng
        q = chain_stats.current_sample.copy()
        k = int(np.rint(q[-1]))
        old = q[: k + 1].copy()
        u = rng.random(k + 1)
        q[: k + 1] = np.log1p(u * (np.exp(self.a) - 1.0)) / self.a
        qxy = self.a * (np.sum(old) - np.sum(q[: k + 1]))
        return q, qxy


def _unit_flat_logprior(params):
    """Flat prior on [0, 1] for any number of slots (log-density 0)."""
    p = np.asarray(params, float)
    return 0.0 if np.all((p >= 0.0) & (p <= 1.0)) else -np.inf


def _unit_uniform_draw(rng):
    return rng.random(1)


def _beta31_draw(rng):
    """Beta(3, 1) birth-proposal draw, deliberately far from the flat prior."""
    return rng.beta(3.0, 1.0, size=1)


def _beta31_logpdf(p):
    """Beta(3, 1): pdf = 3 x^2."""
    p = np.asarray(p, float)
    if np.any((p <= 0.0) | (p >= 1.0)):
        return -np.inf
    return float(np.sum(np.log(3.0) + 2.0 * np.log(p)))


def _exch_analytic(num_sources, a, b):
    """P(m) proportional to c^(m+1), with c = e^b (e^a - 1) / a."""
    c = np.exp(b) * (np.exp(a) - 1.0) / a
    p = c ** (np.arange(num_sources) + 1)
    return p / p.sum()


def _run_full_mixture(num_sources, n_iter, seed, draw_mode, a=_TILT, b=-_TILT, early_de=False):
    """Chain on ``logL = a*sum(t) + b*n`` with the full production mixture.

    The trans-dimensional kernel is built through
    ``BirthDeathProductSpace.get_birth_death_proposal`` (production wiring,
    including the per-source prior-density resolution), and competes at
    constant ``JumpProposals`` weights with the production ``nmodel_jump``
    and ``source_swap`` moves plus an exact within-model Gibbs move.

    With ``early_de=True`` the production min-fill-gated DE move
    (``EarlyDE``, registered by default by ``PTSampler.from_rjmcmc``) is
    added to the mixture at constant weight, its buffer fed each iteration
    from the chain's own history (as ``recursive_update`` does in
    production), and — matching production wiring — its parameter group
    contains only the continuous slots, never the model index.
    """
    rng = np.random.default_rng(seed)

    def loglike(active):
        active = np.asarray(active, float)
        return a * float(np.sum(active)) + b * active.size

    if draw_mode == "prior":
        space = BirthDeathProductSpace(
            loglikelihood=loglike,
            logprior=_unit_flat_logprior,
            num_sources=num_sources,
            num_params=1,
            source_prior_draw=_unit_uniform_draw,
        )
    else:  # birth draws from Beta(3, 1) != prior (q != p Hastings terms)
        space = BirthDeathProductSpace(
            loglikelihood=loglike,
            logprior=_unit_flat_logprior,
            num_sources=num_sources,
            num_params=1,
            source_prior_draw=_beta31_draw,
            source_proposal_logpdf=_beta31_logpdf,
        )
    kernel = space.get_birth_death_proposal()

    ndim = num_sources + 1
    ptstate = PTState(ndim=ndim, ntemps=1, min_temp=1.0, max_temp=1.0)
    cs = ChainStats(
        ndim=ndim,
        pt_state=ptstate,
        chain_index=0,
        rng=rng,
        buffer_size=200 if early_de else 50,
        # production groups exclude the model index (get_default_groups /
        # per-model groups): EarlyDE must only ever touch continuous slots
        groups=[np.arange(num_sources)] if early_de else None,
    )
    jumps = JumpProposals(cs)
    # constant selection weights, as registered by PTSampler.from_rjmcmc
    jumps.add_jump(kernel, 40)
    jumps.add_jump(NmodelJump(num_sources), 20)
    jumps.add_jump(make_source_swap_proposal(1), 20)
    jumps.add_jump(_ActiveGibbs(a), 20)
    if early_de:
        jumps.add_jump(make_early_de(min_fill=100), 20)

    def logpost(z):
        k = int(np.rint(z[-1]))
        if k < 0 or k >= num_sources:
            return -np.inf
        if not np.isfinite(_unit_flat_logprior(z[:num_sources])):
            return -np.inf
        return loglike(z[: k + 1])

    z = np.empty(ndim)
    z[:-1] = rng.random(num_sources)
    z[-1] = 0
    lp = logpost(z)

    counts = np.zeros(num_sources)
    for _ in range(n_iter):
        y, qxy = jumps(SimpleNamespace(positions=z.reshape(1, -1)))
        ly = logpost(y)
        if np.log(rng.random()) < (ly - lp) + qxy:
            z, lp = y, ly
        if early_de:
            # feed the DE buffer from the chain's own history, as
            # ChainStats.recursive_update does in production
            cs.sample_total += 1
            cs.update_buffer(z.reshape(1, -1))
        counts[int(np.rint(z[-1]))] += 1
    return counts / counts.sum()


@pytest.mark.parametrize("draw_mode", ["prior", "proposal"])
def test_full_mixture_parameter_dependent_likelihood(draw_mode):
    """Full production mixture must reproduce the analytic model posterior.

    ``draw_mode="prior"`` pins the inactive-slot memory bug (value-preserving
    death + nmodel_jump); ``draw_mode="proposal"`` additionally pins the
    death-side q != p Hastings term (birth draws from Beta(3, 1) while the
    prior is flat).  Pre-fix maxerr: ~0.042 (prior) and ~0.049 (proposal)
    across seeds; post-fix both are < 0.005.
    """
    probs = _run_full_mixture(
        num_sources=3,
        n_iter=400_000,
        seed=20260703,
        draw_mode=draw_mode,
    )
    target = _exch_analytic(3, _TILT, -_TILT)
    assert (
        np.max(np.abs(probs - target)) < 0.02
    ), f"[{draw_mode}] biased model posterior {probs} vs analytic {target}"


def test_full_mixture_with_early_de():
    """The default RJ mixture including the min-fill-gated DE (``EarlyDE``,
    registered by ``PTSampler.from_rjmcmc``) must preserve the analytic
    model posterior.

    Why EarlyDE cannot bias the model-index marginal (the argument that
    stands in for extending the exact enumeration, whose scripted rng
    cannot enumerate the move's continuous scale): it is a fixed-dimension
    increment move ``y = x + gamma * (b_mm - b_nn)`` on ONE
    continuous-parameter group, and production groups never contain the
    model index, so ``nmodel`` is never modified and the trans-dimensional
    kernel's exactness is untouched.  The group index, the buffer-row pair
    ``(mm, nn)``, and ``gamma`` are drawn independently of the current
    point from a buffer frozen within the iteration; the reverse move uses
    the swapped pair ``(nn, mm)`` at the same ``gamma``/group with
    identical probability, so ``q(y|x) = q(x|y)`` and ``qxy = 0`` is the
    exact Hastings factor.  Below ``min_fill`` the kernel is the identity,
    trivially in detailed balance.

    What this run adds beyond that argument: EarlyDE draws from the
    chain's OWN growing history (an adaptive proposal), so stationarity is
    verified empirically with the buffer fed every iteration, exactly as
    ``recursive_update`` feeds it in production.
    """
    probs = _run_full_mixture(
        num_sources=3,
        n_iter=400_000,
        seed=20260705,
        draw_mode="prior",
        early_de=True,
    )
    target = _exch_analytic(3, _TILT, -_TILT)
    assert (
        np.max(np.abs(probs - target)) < 0.02
    ), f"[early_de] biased model posterior {probs} vs analytic {target}"


@pytest.mark.slow
def test_full_mixture_k4_kill_choice_bias():
    """K=4, a=2, b=-1: the config sensitive to the uniform-kill death defect.

    A death that picks its victim uniformly among the active sources
    (compacting the last active source into the hole) while carrying no
    kill-choice factor in ``qxy`` is not the reverse of the append-last
    birth; exact enumeration of the production kernels proved a residual
    stationary bias toward FEWER sources.  K=3 tilted configs are
    insensitive (predicted residual ~9e-5) — only this K=4 config has
    detection power: validation runs measured pooled maxerr 0.0031 +/-
    0.0006 (z up to 5.6 over 6 seeds x 320k and 5.2 over 8 x 600k).

    Sized for clear power: at 3M iterations the pre-fix bias (~0.003) sits
    far above the threshold 0.002, which in turn sits far above the
    post-fix Monte Carlo noise (~3 batch-means SEs ~ 0.0007).  The fast
    deterministic companion is test_exact_enumeration_stationarity.
    """
    probs = _run_full_mixture(
        num_sources=4,
        n_iter=3_000_000,
        seed=20260704,
        draw_mode="prior",
        a=2.0,
        b=-1.0,
    )
    target = _exch_analytic(4, 2.0, -1.0)
    assert np.max(np.abs(probs - target)) < 0.002, (
        f"biased model posterior {probs} vs analytic {target} " f"(err {probs - target})"
    )


# ---------------------------------------------------------------------------
# EXACT finite-state enumeration of the REAL kernels (zero MC noise).
#
# Slots take values on a finite grid in [0, 1] with a flat per-slot prior,
# so the full product-space chain (t_0..t_{K-1}, m) is a finite Markov
# chain.  The transition matrix of the mixture
#
#     w_bd * BirthDeathProposal  +  w_nm * NmodelJump  +  w_g * exact Gibbs
#
# is built by driving the REAL package proposals with a scripted rng that
# enumerates every internal random choice, and the target distribution is
# checked to be EXACTLY invariant: ||pi P - pi||_TV at machine precision.
#
# This is the deterministic, seconds-fast regression for the kill-choice
# defect: with the uniform-kill death (victim uniform among actives, last
# active compacted into the hole) and the unchanged qxy, the same
# enumeration yields a residual of order 1e-3 at the sensitive K=4 config
# — the exact counterpart of the ~0.003 stationary bias seen in the Monte
# Carlo runs.  The kill-last death is exactly invariant.  The scripted-rng
# death branch enumerates a kill index only if the kernel actually draws
# one, so this harness measures whichever death kernel is implemented.
# ---------------------------------------------------------------------------


class _ScriptedRng:
    """Replays scripted values to the real proposals.

    ``random()`` pops from ``seq``; ``integers(lo, hi)`` pops from the
    separate ``ints`` queue (so a kernel that draws no integers leaves
    ``ints`` untouched, which the enumeration uses to detect whether the
    death kernel consumes a kill choice).
    """

    def __init__(self, seq, ints=()):
        self.seq = list(seq)
        self.ints = list(ints)

    def random(self):
        return self.seq.pop(0)

    def integers(self, lo, hi):
        v = self.ints.pop(0)
        assert lo <= v < hi, (v, lo, hi)
        return v


class _GridDraw:
    """draw_from_prior stub: returns the value scripted into the fake rng."""

    __name__ = "grid_draw"

    def __call__(self, rng):
        return np.array([rng.seq.pop(0)], dtype=float)


def _exact_transition_matrix(K, grid, loglike_v, w_bd=0.5, w_nm=0.25):
    """Exact transition matrix of w_bd*BD + w_nm*NM + (1-w_bd-w_nm)*Gibbs.

    One parameter per source; slot i of a state is an index into ``grid``
    and the model index m is last.  Per-slot prior: flat on the grid.
    ``loglike_v[g]``: per-active-slot log-likelihood contribution
    (exchangeable: logL_m = sum over the m+1 active slots).  The birth/death
    draw distribution is uniform on the grid (q = p, so the draw-density
    qxy terms vanish and the schedule ratio is isolated).
    """
    G = len(grid)
    logp_slot = -np.log(G)

    states = [s + (m,) for s in itertools.product(range(G), repeat=K) for m in range(K)]
    idx = {s: i for i, s in enumerate(states)}
    N = len(states)

    logpis = np.array([K * logp_slot + sum(loglike_v[g] for g in s[: s[-1] + 1]) for s in states])

    def state_of(x_arr):
        gs = tuple(int(np.argmin(np.abs(grid - v))) for v in x_arr[:-1])
        assert all(np.isclose(grid[g], v) for g, v in zip(gs, x_arr[:-1]))
        return gs + (int(np.rint(x_arr[-1])),)

    bd = make_birth_death_proposal(1, K, _GridDraw())
    nm = NmodelJump(K)
    q_pmf = 1.0 / G  # uniform draw over the grid

    P = np.zeros((N, N))
    for i, s in enumerate(states):
        m = s[-1]
        x = np.array([grid[g] for g in s[:-1]] + [float(m)])

        def add(p, qxy, yi):
            if p <= 0.0:
                return
            a = min(1.0, np.exp(min(700.0, logpis[yi] - logpis[i] + qxy)))
            P[i, yi] += p * a
            P[i, i] += p * (1.0 - a)

        # ---- BD kernel: enumerate selection x (kill choice) x draw -------
        pb, pd = default_birth_death_probs(m, K)
        tot = pb + pd
        if pb > 0.0:
            for gw in range(G):
                # rng.random() = 0.0 < pb/tot selects birth; the queued grid
                # value is the birth draw
                rng = _ScriptedRng([0.0, grid[gw]])
                y, qxy = bd(_Stats(rng, x))
                add(w_bd * (pb / tot) * q_pmf, qxy, idx[state_of(y)])
        if pd > 0.0:
            for j in range(m + 1):
                # rng.random() = 0.999999 >= pb/tot selects death; the
                # queued grid value is the slot-refresh draw; j is consumed
                # only by a kernel that draws a kill index
                used_kill = False
                for gw in range(G):
                    rng = _ScriptedRng([0.999999, grid[gw]], ints=[j])
                    y, qxy = bd(_Stats(rng, x))
                    used_kill = not rng.ints
                    kill_w = 1.0 / (m + 1) if used_kill else 1.0
                    add(w_bd * (pd / tot) * kill_w * q_pmf, qxy, idx[state_of(y)])
                if not used_kill:
                    break  # single-channel (kill-last) death: one branch

        # ---- NmodelJump: real code, enumerate the model draw --------------
        for mp in range(K):
            y, qxy = nm(_Stats(_ScriptedRng([], ints=[mp]), x))
            add(w_nm * (1.0 / K), qxy, idx[state_of(y)])

        # ---- exact single-slot Gibbs (pi-invariant filler) ----------------
        w_g = 1.0 - w_bd - w_nm
        for slot in range(K):
            logw = np.array([logp_slot + (loglike_v[g] if slot <= m else 0.0) for g in range(G)])
            w = np.exp(logw - logw.max())
            w /= w.sum()
            for g in range(G):
                y = list(s)
                y[slot] = g
                P[i, idx[tuple(y)]] += w_g * (1.0 / K) * w[g]

    return states, logpis, P


@pytest.mark.parametrize(
    "K,a,b,G",
    [
        (3, 3.0, -3.0, 5),
        (4, 2.0, -1.0, 4),  # the config sensitive to the kill-choice defect
    ],
)
def test_exact_enumeration_stationarity(K, a, b, G):
    """The exact target must be EXACTLY invariant under the real kernels.

    Deterministic, machine-precision counterpart of the Monte Carlo tests:
    with the kill-last death the residual is ~1e-16; with the uniform-kill
    death (same qxy) the K=4 config leaves a genuine detailed-balance
    violation of order 1e-3, so this test fails loudly on that kernel.
    """
    grid = (np.arange(G) + 0.5) / G
    loglike_v = a * grid + b
    states, logpis, P = _exact_transition_matrix(K, grid, loglike_v)

    # sanity: rows are probability distributions
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)

    pi = np.exp(logpis - logpis.max())
    pi /= pi.sum()

    # sanity: the target's model marginal is the analytic grid posterior
    # P(m) proportional to c^(m+1) with c = mean_g exp(loglike_v[g])
    marg = np.zeros(K)
    for s, p in zip(states, pi):
        marg[s[-1]] += p
    c = float(np.mean(np.exp(loglike_v)))
    target = c ** (np.arange(K) + 1.0)
    target /= target.sum()
    np.testing.assert_allclose(marg, target, atol=1e-12)

    # THE invariance check: pi P == pi exactly
    piP = pi @ P
    resid = 0.5 * float(np.abs(piP - pi).sum())

    # diagnostic for the failure message: the kernel's true stationary model
    # marginal (power iteration; converges immediately when pi is invariant)
    v = piP
    for _ in range(20_000):
        v_next = v @ P
        if float(np.abs(v_next - v).sum()) < 1e-15:
            v = v_next
            break
        v = v_next
    stat_marg = np.zeros(K)
    for s, p in zip(states, v):
        stat_marg[s[-1]] += p

    assert resid < 1e-12, (
        f"target not invariant under the kernel: TV(pi P, pi) = {resid:.3e};"
        f" stationary model marginal {stat_marg} vs analytic {target}"
        f" (err {stat_marg - target})"
    )
