"""Tests for the SBC driver loops in impulse.validation.

``run_sbc_continuous`` and ``run_sbc_model_selection`` are both exported in
``impulse.__all__`` but had zero references anywhere in tests/ -- their only
callers were notebooks, which docs/conf.py never executes. That left the
tooling this package ships for *proving the sampler unbiased* as the least
proven code in it (module coverage 19%).

These tests drive the loops with stub samplers so they exercise the driver's
own logic -- cold-chain selection, burn/thin, column slicing, bookkeeping,
tmpdir cleanup -- without paying for real MCMC. One end-to-end test with a real
PTSampler covers the integration, and two statistical tests confirm the drivers
report uniform ranks/PIT when handed a correctly calibrated posterior.
"""

import os

import numpy as np
import pytest

from impulse.validation import run_sbc_continuous, run_sbc_model_selection

# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------


class _StubSampler:
    """Minimal stand-in: the drivers only ever call load_chain()."""

    def __init__(self, samples):
        self._samples = np.asarray(samples, dtype=float)

    def load_chain(self):
        return {"samples": self._samples}


class _StubSpace:
    """Minimal stand-in for BirthDeathProductSpace."""

    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=float)
        self.seen_burn = None

    def model_posterior_probs(self, samples, burn=0):
        self.seen_burn = burn
        return self._probs


# ---------------------------------------------------------------------------
# run_sbc_continuous
# ---------------------------------------------------------------------------


class TestRunSBCContinuous:
    def test_output_shapes_and_keys(self):
        ndim, n_sims, nsamp = 3, 5, 400

        def factory(data, rng, outdir):
            # (ntemps, nsamples, ndim) -- the PT shape the driver must reduce
            return _StubSampler(rng.standard_normal((4, nsamp, ndim)))

        out = run_sbc_continuous(
            sampler_factory=factory,
            prior_draw=lambda rng: rng.standard_normal(ndim),
            data_generator=lambda theta, rng: theta,
            n_simulations=n_sims,
            burn=100,
            seed=0,
        )

        assert set(out) == {"ranks", "quantiles", "true_values", "n_posterior"}
        assert out["ranks"].shape == (n_sims, ndim)
        assert out["quantiles"].shape == (n_sims, ndim)
        assert out["true_values"].shape == (n_sims, ndim)
        assert out["n_posterior"] == nsamp - 100

    def test_burn_and_thin_are_applied(self):
        def factory(data, rng, outdir):
            return _StubSampler(rng.standard_normal((2, 1000, 2)))

        out = run_sbc_continuous(
            sampler_factory=factory,
            prior_draw=lambda rng: rng.standard_normal(2),
            data_generator=lambda theta, rng: theta,
            n_simulations=2,
            burn=200,
            thin=4,
            seed=1,
        )
        # (1000 - 200) / 4
        assert out["n_posterior"] == 200

    def test_accepts_2d_samples_from_a_non_pt_sampler(self):
        """A (nsamples, ndim) chain must work as well as the 3-D PT shape."""

        def factory(data, rng, outdir):
            return _StubSampler(rng.standard_normal((500, 2)))

        out = run_sbc_continuous(
            sampler_factory=factory,
            prior_draw=lambda rng: rng.standard_normal(2),
            data_generator=lambda theta, rng: theta,
            n_simulations=3,
            burn=50,
            seed=2,
        )
        assert out["ranks"].shape == (3, 2)
        assert out["n_posterior"] == 450

    def test_extra_columns_are_ignored(self):
        """Only the first ndim columns are scored (model index etc. excluded)."""
        ndim = 2

        def factory(data, rng, outdir):
            # 5 columns, only the first 2 are parameters
            return _StubSampler(rng.standard_normal((1, 300, 5)))

        out = run_sbc_continuous(
            sampler_factory=factory,
            prior_draw=lambda rng: rng.standard_normal(ndim),
            data_generator=lambda theta, rng: theta,
            n_simulations=2,
            burn=0,
            seed=3,
        )
        assert out["ranks"].shape == (2, ndim)

    def test_factory_called_once_per_simulation_with_a_real_tmpdir(self):
        calls = []

        def factory(data, rng, outdir):
            calls.append(outdir)
            assert os.path.isdir(outdir), "driver must hand the factory a real directory"
            return _StubSampler(rng.standard_normal((1, 200, 1)))

        run_sbc_continuous(
            sampler_factory=factory,
            prior_draw=lambda rng: rng.standard_normal(1),
            data_generator=lambda theta, rng: theta,
            n_simulations=4,
            burn=0,
            seed=4,
        )

        assert len(calls) == 4
        assert len(set(calls)) == 4, "each simulation needs its own outdir"
        # and every one of them is cleaned up afterwards
        assert not any(os.path.exists(d) for d in calls)

    def test_seed_makes_the_run_reproducible(self):
        def factory(data, rng, outdir):
            return _StubSampler(rng.standard_normal((1, 300, 2)))

        kw = dict(
            sampler_factory=factory,
            prior_draw=lambda rng: rng.standard_normal(2),
            data_generator=lambda theta, rng: theta,
            n_simulations=3,
            burn=0,
        )
        a = run_sbc_continuous(seed=99, **kw)
        b = run_sbc_continuous(seed=99, **kw)
        np.testing.assert_array_equal(a["ranks"], b["ranks"])
        np.testing.assert_array_equal(a["true_values"], b["true_values"])

    def test_calibrated_posterior_gives_uniform_ranks(self):
        """The property SBC exists to detect.

        The "posterior" here is exact: draws from the same N(0, 1) the true
        value came from. SBC ranks must then be uniform on [0, n_posterior].
        """
        ndim, n_sims, nsamp = 1, 400, 200

        def factory(data, rng, outdir):
            return _StubSampler(rng.standard_normal((1, nsamp, ndim)))

        out = run_sbc_continuous(
            sampler_factory=factory,
            prior_draw=lambda rng: rng.standard_normal(ndim),
            data_generator=lambda theta, rng: theta,
            n_simulations=n_sims,
            burn=0,
            seed=1234,
        )

        u = out["ranks"][:, 0] / nsamp
        counts, _ = np.histogram(u, bins=10, range=(0, 1))
        frac = counts / counts.sum()
        assert np.max(np.abs(frac - 0.1)) < 0.05, f"ranks not uniform: {frac}"

    def test_biased_posterior_gives_non_uniform_ranks(self):
        """Sensitivity check: a shifted posterior must NOT look calibrated."""
        nsamp = 200

        def factory(data, rng, outdir):
            return _StubSampler(rng.standard_normal((1, nsamp, 1)) + 1.5)

        out = run_sbc_continuous(
            sampler_factory=factory,
            prior_draw=lambda rng: rng.standard_normal(1),
            data_generator=lambda theta, rng: theta,
            n_simulations=300,
            burn=0,
            seed=5,
        )
        u = out["ranks"][:, 0] / nsamp
        counts, _ = np.histogram(u, bins=10, range=(0, 1))
        frac = counts / counts.sum()
        assert np.max(np.abs(frac - 0.1)) > 0.05, "a biased posterior should show up"


# ---------------------------------------------------------------------------
# run_sbc_model_selection
# ---------------------------------------------------------------------------


class TestRunSBCModelSelection:
    def test_output_shapes_and_keys(self):
        num_models, n_sims = 3, 6
        probs = np.array([0.2, 0.5, 0.3])

        def factory(true_nmodel, true_params, data, rng, outdir):
            assert os.path.isdir(outdir)
            return _StubSampler(np.zeros((2, 100, 4))), _StubSpace(probs)

        out = run_sbc_model_selection(
            sampler_factory=factory,
            model_prior_draw=lambda rng: int(rng.integers(0, num_models)),
            param_prior_draw=lambda k, rng: rng.standard_normal(2),
            data_generator=lambda k, p, rng: p,
            num_models=num_models,
            n_simulations=n_sims,
            burn=10,
            seed=0,
        )

        assert set(out) == {"pit_values", "true_models", "posterior_probs"}
        assert out["pit_values"].shape == (n_sims,)
        assert out["true_models"].shape == (n_sims,)
        assert out["posterior_probs"].shape == (n_sims, num_models)
        assert np.all((out["pit_values"] >= 0) & (out["pit_values"] <= 1))
        assert np.all((out["true_models"] >= 0) & (out["true_models"] < num_models))

    def test_burn_is_forwarded_to_the_space(self):
        spaces = []

        def factory(true_nmodel, true_params, data, rng, outdir):
            sp = _StubSpace(np.array([0.5, 0.5]))
            spaces.append(sp)
            return _StubSampler(np.zeros((1, 50, 3))), sp

        run_sbc_model_selection(
            sampler_factory=factory,
            model_prior_draw=lambda rng: 0,
            param_prior_draw=lambda k, rng: rng.standard_normal(1),
            data_generator=lambda k, p, rng: p,
            num_models=2,
            n_simulations=3,
            burn=777,
            seed=1,
        )
        assert [s.seen_burn for s in spaces] == [777, 777, 777]

    def test_tmpdirs_are_cleaned_up(self):
        seen = []

        def factory(true_nmodel, true_params, data, rng, outdir):
            seen.append(outdir)
            return _StubSampler(np.zeros((1, 20, 2))), _StubSpace(np.array([1.0, 0.0]))

        run_sbc_model_selection(
            sampler_factory=factory,
            model_prior_draw=lambda rng: 0,
            param_prior_draw=lambda k, rng: rng.standard_normal(1),
            data_generator=lambda k, p, rng: p,
            num_models=2,
            n_simulations=3,
            burn=0,
            seed=2,
        )
        assert len(set(seen)) == 3
        assert not any(os.path.exists(d) for d in seen)

    def test_calibrated_model_posterior_gives_uniform_pit(self):
        """With the true model drawn from the same distribution the posterior
        reports, the randomized PIT must be uniform on [0, 1]."""
        num_models = 3
        probs = np.array([0.2, 0.5, 0.3])

        def factory(true_nmodel, true_params, data, rng, outdir):
            return _StubSampler(np.zeros((1, 10, 2))), _StubSpace(probs)

        out = run_sbc_model_selection(
            sampler_factory=factory,
            # draw the true model FROM the posterior the space reports
            model_prior_draw=lambda rng: int(rng.choice(num_models, p=probs)),
            param_prior_draw=lambda k, rng: rng.standard_normal(1),
            data_generator=lambda k, p, rng: p,
            num_models=num_models,
            n_simulations=3000,
            burn=0,
            seed=7,
        )

        counts, _ = np.histogram(out["pit_values"], bins=10, range=(0, 1))
        frac = counts / counts.sum()
        assert np.max(np.abs(frac - 0.1)) < 0.03, f"PIT not uniform: {frac}"


# ---------------------------------------------------------------------------
# integration with a real sampler
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_run_sbc_continuous_with_a_real_sampler(temp_dir):
    """End-to-end: the driver drives an actual PTSampler.

    Small and seeded -- this asserts the plumbing (factory contract, load_chain
    shape, burn) rather than calibration, which the stub tests cover.
    """
    from impulse import PTSampler

    sigma0 = 1.0

    def factory(data, rng, outdir):
        def lnlike(x):
            x = np.asarray(x)
            if x.ndim == 1:
                return -0.5 * np.sum((x - data) ** 2)
            return -0.5 * np.sum((x - data) ** 2, axis=1)

        def lnprior(x):
            x = np.asarray(x)
            if x.ndim == 1:
                return -0.5 * np.sum(x**2)
            return -0.5 * np.sum(x**2, axis=1)

        sampler = PTSampler(
            ndim=1,
            lnlike=lnlike,
            lnprior=lnprior,
            ntemps=2,
            seed=int(rng.integers(0, 2**31)),
            outdir=outdir,
            save_freq=1200,
        )
        sampler.sample([0.0], num_iterations=1200)
        return sampler

    out = run_sbc_continuous(
        sampler_factory=factory,
        prior_draw=lambda rng: rng.standard_normal(1),
        data_generator=lambda theta, rng: theta + sigma0 * rng.standard_normal(1),
        n_simulations=3,
        burn=200,
        seed=11,
    )

    assert out["ranks"].shape == (3, 1)
    assert out["n_posterior"] == 1000
    assert np.all(out["ranks"] >= 0) and np.all(out["ranks"] <= out["n_posterior"])
