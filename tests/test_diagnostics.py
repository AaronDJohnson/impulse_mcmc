import numpy as np
import pytest
from scipy.stats import norm

from impulse.diagnostics import (
    _acf_fft,
    _ims_tau_from_gamma,
    _ips_tau_from_gamma,
    _next_fast_len,
    _pair_sums_gamma,
    _pava_monotone_nonincreasing,
    autocorr_length_ips_ims,
    effective_sample_size,
    grubin,
)


class TestNextFastLen:
    """Test suite for _next_fast_len function"""

    def test_next_fast_len_basic(self):
        """Test basic functionality"""
        assert _next_fast_len(100) == 256  # Next power of 2 >= 200
        assert _next_fast_len(500) == 1024  # Next power of 2 >= 1000
        assert _next_fast_len(1000) == 2048  # Next power of 2 >= 2000

    def test_next_fast_len_exact_powers_of_two(self):
        """Test with exact powers of two"""
        assert _next_fast_len(512) == 1024  # 2*512 = 1024
        assert _next_fast_len(256) == 512  # 2*256 = 512

    def test_next_fast_len_small_numbers(self):
        """Test with small numbers"""
        assert _next_fast_len(1) == 2
        assert _next_fast_len(2) == 4
        assert _next_fast_len(3) == 8

    def test_next_fast_len_zero(self):
        """Test with zero"""
        assert _next_fast_len(0) == 1  # 2*0 = 0, but m starts at 1


class TestAcfFft:
    """Test suite for _acf_fft function"""

    def test_acf_fft_constant_series(self):
        """Test ACF of constant series"""
        x = np.ones(100)
        rho = _acf_fft(x)
        # Constant series has zero variance, should return all NaN
        assert np.all(np.isnan(rho))

    def test_acf_fft_white_noise(self):
        """Test ACF of white noise"""
        np.random.seed(42)
        x = np.random.randn(1000)
        rho = _acf_fft(x)

        # First lag should be 1.0
        assert np.isclose(rho[0], 1.0)

        # Other lags should be close to 0 for white noise
        assert np.abs(rho[1:10]).mean() < 0.1

    def test_acf_fft_ar1_process(self):
        """Test ACF of AR(1) process"""
        np.random.seed(42)
        n = 1000
        phi = 0.8

        # Generate AR(1) process
        x = np.zeros(n)
        x[0] = np.random.randn()
        for i in range(1, n):
            x[i] = phi * x[i - 1] + np.random.randn()

        rho = _acf_fft(x)

        # Should be close to theoretical ACF: rho[k] = phi^k
        assert np.isclose(rho[0], 1.0)
        assert 0.6 < rho[1] < 0.9  # Should be around 0.8
        assert 0.4 < rho[2] < 0.8  # Should be around 0.64

    def test_acf_fft_length(self):
        """Test that ACF has correct length"""
        for n in [10, 50, 100]:
            x = np.random.randn(n)
            rho = _acf_fft(x)
            assert len(rho) == n

    def test_acf_fft_single_value(self):
        """Test ACF with single value"""
        x = np.array([5.0])
        rho = _acf_fft(x)
        assert len(rho) == 1
        assert np.isnan(rho[0])  # Zero variance


class TestPairSumsGamma:
    """Test suite for _pair_sums_gamma function"""

    def test_pair_sums_gamma_basic(self):
        """Test basic functionality"""
        rho = np.array([1.0, 0.8, 0.6, 0.4, 0.2])  # lag 0,1,2,3,4
        gamma = _pair_sums_gamma(rho)

        # gamma[0] = rho[1] + rho[2] = 0.8 + 0.6 = 1.4
        # gamma[1] = rho[3] + rho[4] = 0.4 + 0.2 = 0.6
        expected = np.array([1.4, 0.6])
        np.testing.assert_array_almost_equal(gamma, expected)

    def test_pair_sums_gamma_odd_length(self):
        """Test with odd number of lags"""
        rho = np.array([1.0, 0.8, 0.6, 0.4])  # lag 0,1,2,3
        gamma = _pair_sums_gamma(rho)

        # Only one pair: gamma[0] = rho[1] + rho[2] = 0.8 + 0.6 = 1.4
        expected = np.array([1.4])
        np.testing.assert_array_almost_equal(gamma, expected)

    def test_pair_sums_gamma_short_input(self):
        """Test with very short input"""
        rho = np.array([1.0, 0.5])
        gamma = _pair_sums_gamma(rho)
        # Only lag 0 and 1, no pairs to form
        assert len(gamma) == 0

    def test_pair_sums_gamma_single_element(self):
        """Test with single element"""
        rho = np.array([1.0])
        gamma = _pair_sums_gamma(rho)
        assert len(gamma) == 0


class TestIpsTauFromGamma:
    """Test suite for _ips_tau_from_gamma function"""

    def test_ips_tau_basic(self):
        """Test basic IPS tau calculation"""
        gamma = np.array([0.8, 0.4, 0.2])
        tau = _ips_tau_from_gamma(gamma)
        # tau = 1 + 2*(0.8 + 0.4 + 0.2) = 1 + 2*1.4 = 3.8
        assert np.isclose(tau, 3.8)

    def test_ips_tau_with_negative(self):
        """Test IPS tau with negative values"""
        gamma = np.array([0.8, 0.4, -0.1, 0.2])
        tau = _ips_tau_from_gamma(gamma)
        # Should stop at first negative: 1 + 2*(0.8 + 0.4) = 3.4
        assert np.isclose(tau, 3.4)

    def test_ips_tau_all_negative(self):
        """Test IPS tau with all negative values"""
        gamma = np.array([-0.1, -0.2])
        tau = _ips_tau_from_gamma(gamma)
        assert tau == 1.0

    def test_ips_tau_empty_gamma(self):
        """Test IPS tau with empty gamma"""
        gamma = np.array([])
        tau = _ips_tau_from_gamma(gamma)
        assert tau == 1.0


class TestPavaMonotoneNonincreasing:
    """Test suite for _pava_monotone_nonincreasing function"""

    def test_pava_already_monotone(self):
        """Test with already monotone nonincreasing sequence"""
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = _pava_monotone_nonincreasing(y)
        np.testing.assert_array_almost_equal(result, y)

    def test_pava_needs_adjustment(self):
        """Test with sequence that needs adjustment"""
        y = np.array([5.0, 3.0, 4.0, 2.0])  # 3.0 < 4.0 violates nonincreasing
        result = _pava_monotone_nonincreasing(y)

        # Result should be nonincreasing
        assert np.all(np.diff(result) <= 1e-10)  # Allow for numerical precision

    def test_pava_single_element(self):
        """Test with single element"""
        y = np.array([5.0])
        result = _pava_monotone_nonincreasing(y)
        np.testing.assert_array_equal(result, y)

    def test_pava_empty_array(self):
        """Test with empty array"""
        y = np.array([])
        result = _pava_monotone_nonincreasing(y)
        np.testing.assert_array_equal(result, y)

    def test_pava_with_weights(self):
        """Test PAVA with custom weights"""
        y = np.array([5.0, 3.0, 4.0, 2.0])
        w = np.array([1.0, 2.0, 1.0, 1.0])
        result = _pava_monotone_nonincreasing(y, w)

        # Result should be nonincreasing
        assert np.all(np.diff(result) <= 1e-10)


class TestImsTauFromGamma:
    """Test suite for _ims_tau_from_gamma function"""

    def test_ims_tau_basic(self):
        """Test basic IMS tau calculation"""
        gamma = np.array([0.8, 0.6, 0.4])  # Already nonincreasing
        tau = _ims_tau_from_gamma(gamma)
        # Should be same as IPS: 1 + 2*(0.8 + 0.6 + 0.4) = 4.6
        assert np.isclose(tau, 4.6)

    def test_ims_tau_needs_monotonization(self):
        """Test IMS tau with non-monotonic gamma"""
        gamma = np.array([0.6, 0.8, 0.4])  # 0.6 < 0.8 violates nonincreasing
        tau = _ims_tau_from_gamma(gamma)

        # Should be >= 1.0 and finite
        assert tau >= 1.0
        assert np.isfinite(tau)

    def test_ims_tau_empty_gamma(self):
        """Test IMS tau with empty gamma"""
        gamma = np.array([])
        tau = _ims_tau_from_gamma(gamma)
        assert tau == 1.0


class TestAutocorrLengthIpsIms:
    """Test suite for autocorr_length_ips_ims function"""

    def test_autocorr_length_white_noise(self):
        """Test autocorrelation length for white noise"""
        np.random.seed(42)
        chain = np.random.randn(1000, 3)

        tau_ips, tau_ims = autocorr_length_ips_ims(chain)

        # Both should be close to 1 for white noise (allow wider range due to statistical variation)
        assert np.all(tau_ips > 0.3) and np.all(tau_ips < 4.0)
        assert np.all(tau_ims > 0.3) and np.all(tau_ims < 4.0)
        assert tau_ips.shape == (3,)
        assert tau_ims.shape == (3,)

    def test_autocorr_length_ar1(self):
        """Test autocorrelation length for AR(1) process"""
        np.random.seed(42)
        n_samples = 2000
        n_params = 2
        phi = 0.8

        chain = np.zeros((n_samples, n_params))
        chain[0] = np.random.randn(n_params)

        for i in range(1, n_samples):
            chain[i] = phi * chain[i - 1] + 0.6 * np.random.randn(n_params)

        tau_ips, tau_ims = autocorr_length_ips_ims(chain)

        # Should be greater than 1 for autocorrelated process
        assert np.all(tau_ips > 1.0)
        assert np.all(tau_ims > 1.0)
        # Theoretical value for AR(1): (1+phi)/(1-phi) = 1.8/0.2 = 9
        # Should be in reasonable range
        assert np.all(tau_ips < 20)
        assert np.all(tau_ims < 20)

    def test_autocorr_length_wrong_shape(self):
        """Test error with wrong input shape"""
        chain_1d = np.random.randn(100)
        with pytest.raises(ValueError, match="Expected chain with shape \\(T, D\\)"):
            autocorr_length_ips_ims(chain_1d)

    def test_autocorr_length_constant_chain(self):
        """Test with constant chain (zero variance)"""
        chain = np.ones((100, 2))
        tau_ips, tau_ims = autocorr_length_ips_ims(chain)

        # Should return NaN for zero variance
        assert np.all(np.isnan(tau_ips))
        assert np.all(np.isnan(tau_ims))


class TestEffectiveSampleSize:
    """Test suite for effective_sample_size function"""

    def test_ess_white_noise(self):
        """Test ESS for white noise"""
        np.random.seed(42)
        n_samples = 1000
        chain = np.random.randn(n_samples, 2)

        ess = effective_sample_size(chain)

        # ESS should be close to n_samples for white noise
        assert np.all(ess > 500)  # Should be reasonably high
        assert np.all(ess <= n_samples)  # Can't exceed total samples

    def test_ess_correlated_chain(self):
        """Test ESS for correlated chain"""
        np.random.seed(42)
        n_samples = 1000
        chain = np.zeros((n_samples, 1))
        chain[0] = np.random.randn()

        # High autocorrelation
        for i in range(1, n_samples):
            chain[i] = 0.9 * chain[i - 1] + 0.1 * np.random.randn()

        ess = effective_sample_size(chain)

        # ESS should be much less than n_samples
        assert ess[0] < n_samples / 2
        assert ess[0] > 0

    def test_ess_wrong_shape(self):
        """Test ESS with wrong input shape"""
        chain_1d = np.random.randn(100)
        with pytest.raises(ValueError, match="Expected chain with shape \\(T, D\\)"):
            effective_sample_size(chain_1d)

    def test_ess_constant_chain(self):
        """Test ESS with constant chain"""
        chain = np.ones((100, 2))
        ess = effective_sample_size(chain)

        # Should return NaN for zero variance
        assert np.all(np.isnan(ess))


class TestGrubin:
    """Test suite for grubin function (Gelman-Rubin diagnostic).

    grubin takes parameters ONLY, shape (T, D) -- the same contract as
    effective_sample_size, and exactly what load_chain returns in
    chain["samples"][k].

    These tests previously padded every input with two constant columns
    (np.ones((n, 2))) to feed the old "drop the last two columns" behavior.
    Because the padding was constant, the tests passed whether or not the
    slicing was correct, which is how a wrong column convention survived a
    suite at 84% coverage. Inputs here are parameters only, so the returned
    R-hat length is now a real assertion about the contract.
    """

    def test_grubin_converged_chains(self):
        """Test R-hat for converged chains"""
        np.random.seed(42)
        n_samples = 1000
        n_params = 2

        # Generate well-mixed chains from same distribution
        chains = []
        for _ in range(4):
            chain = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_samples)
            chains.append(chain)

        # Test with concatenated chains
        combined_chain = np.vstack(chains)
        rhat, idx = grubin(combined_chain, M=4)

        # R-hat should be close to 1 for converged chains
        assert len(rhat) == n_params
        assert np.all(rhat < 1.2)  # Good convergence
        assert np.all(rhat >= 1.0)  # R-hat is always >= 1

    def test_grubin_poorly_mixed(self):
        """Test R-hat for poorly mixed chains"""
        np.random.seed(42)
        n_samples = 500

        # Create chains with different means (poor mixing)
        chain1 = np.random.randn(n_samples, 1) + 0  # mean 0
        chain2 = np.random.randn(n_samples, 1) + 5  # mean 5

        rhat, idx = grubin([chain1, chain2], M=2)

        # R-hat should be much greater than 1
        assert len(rhat) == 1
        assert rhat[0] > 1.5  # Poor convergence

    def test_grubin_threshold_detection(self):
        """Test that grubin correctly identifies problematic parameters"""
        np.random.seed(42)
        n_samples = 500

        # Create mixed scenario: one parameter converged, one not
        good_param = np.random.randn(n_samples, 1)
        bad_param1 = np.random.randn(n_samples // 2, 1) + 0
        bad_param2 = np.random.randn(n_samples // 2, 1) + 3
        bad_param = np.vstack([bad_param1, bad_param2])

        chain_data = np.column_stack([good_param, bad_param])

        rhat, idx = grubin(chain_data, M=2, threshold=1.1)

        # Should identify the second parameter as problematic
        assert len(rhat) == 2
        assert len(idx) >= 1  # At least one parameter above threshold
        assert 1 in idx  # Second parameter should be flagged

    def test_grubin_custom_burn(self):
        """Test grubin with custom burn-in"""
        np.random.seed(42)

        # Create chain with burn-in period
        burn_in = np.random.randn(200, 1) + 10  # High initial values
        converged = np.random.randn(800, 1) + 0  # Converged values
        chain_data = np.vstack([burn_in, converged])

        # Test with and without burn-in
        rhat_no_burn, _ = grubin(chain_data, M=4, burn=0)
        rhat_with_burn, _ = grubin(chain_data, M=4, burn=300)

        # R-hat should be better (closer to 1) with proper burn-in
        assert rhat_with_burn[0] < rhat_no_burn[0]

    def test_grubin_single_chain_split(self):
        """Test grubin with single chain that gets split"""
        np.random.seed(42)
        n_samples = 1000

        # Single well-mixed chain
        chain_data = np.random.randn(n_samples, 2)

        rhat, idx = grubin(chain_data, M=4)

        # Should show good convergence when split
        assert len(rhat) == 2
        assert np.all(rhat < 1.2)
        assert len(idx) == 0  # No parameters above default threshold

    def test_grubin_returns_one_rhat_per_column(self):
        """Every supplied column is a parameter; none is silently dropped.

        Regression test: grubin used to slice off the last two columns, so a
        D-parameter chain came back with D-2 R-hat values -- silently omitting
        two real parameters from the convergence check.
        """
        rng = np.random.default_rng(0)
        for d in (1, 2, 3, 5):
            rhat, _ = grubin(rng.standard_normal((400, d)), M=2)
            assert len(rhat) == d

    def test_grubin_accepts_load_chain_output(self):
        """The documented user path -- load_chain()["samples"][k] -- works directly."""
        ndim = 3
        rng = np.random.default_rng(1)
        # load_chain returns parameters only, shape (nsamples, ndim)
        samples = rng.standard_normal((600, ndim))

        rhat, idx = grubin(samples, M=2)

        assert len(rhat) == ndim
        assert np.all(np.isfinite(rhat))

    def test_grubin_rejects_non_2d(self):
        """A 1-D array is a usage error, not something to reinterpret."""
        with pytest.raises(ValueError, match=r"\(T, D\)"):
            grubin(np.random.default_rng(2).standard_normal(500), M=2)
