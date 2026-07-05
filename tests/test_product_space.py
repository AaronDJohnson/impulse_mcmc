import numpy as np
import pytest

from impulse.product_space import NestedProductSpace, ProductSpace


class TestProductSpace:
    """Test suite for ProductSpace class"""

    def test_product_space_init_basic(self):
        """Test basic ProductSpace initialization"""
        model_names = ["linear", "quadratic"]

        def linear_loglike(x):
            return -0.5 * np.sum((x - np.array([1.0, 2.0])) ** 2)

        def quad_loglike(x):
            return -0.5 * np.sum((x - np.array([0.0, 1.0, 2.0])) ** 2)

        def linear_prior(x):
            return 0.0 if len(x) == 2 else -np.inf

        def quad_prior(x):
            return 0.0 if len(x) == 3 else -np.inf

        loglikelihoods = [linear_loglike, quad_loglike]
        logpriors = [linear_prior, quad_prior]
        param_names = [["a", "b"], ["a", "b", "c"]]

        ps = ProductSpace(model_names, loglikelihoods, logpriors, param_names)

        assert ps.model_names == model_names
        assert ps.loglikelihoods == loglikelihoods
        assert ps.logpriors == logpriors
        assert ps.num_models == 2
        np.testing.assert_array_equal(ps.nmodels, np.array([0, 1]))

        # Check parameter names
        expected_params = [
            "a_linear",
            "b_linear",
            "a_quadratic",
            "b_quadratic",
            "c_quadratic",
            "nmodel",
        ]
        assert ps.all_params == expected_params
        assert ps.ndim == 6

        # Check model parameter indices
        assert ps.model_params[0] == [0, 1]  # linear model uses params 0, 1
        assert ps.model_params[1] == [2, 3, 4]  # quad model uses params 2, 3, 4

    def test_product_space_single_model(self):
        """Test ProductSpace with single model"""
        model_names = ["simple"]

        def simple_loglike(x):
            return -np.sum(x**2)

        def simple_prior(x):
            return 0.0

        loglikelihoods = [simple_loglike]
        logpriors = [simple_prior]
        param_names = [["param1"]]

        ps = ProductSpace(model_names, loglikelihoods, logpriors, param_names)

        assert ps.num_models == 1
        assert ps.ndim == 2  # 1 parameter + nmodel
        assert ps.all_params == ["param1_simple", "nmodel"]

    def test_product_space_loglikelihood_model_0(self):
        """Test ProductSpace loglikelihood for model 0"""
        model_names = ["linear", "quadratic"]

        def linear_loglike(x):
            return -0.5 * np.sum(x**2)

        def quad_loglike(x):
            return -0.25 * np.sum(x**2)

        def dummy_prior(x):
            return 0.0

        loglikelihoods = [linear_loglike, quad_loglike]
        logpriors = [dummy_prior, dummy_prior]
        param_names = [["a", "b"], ["a", "b", "c"]]

        ps = ProductSpace(model_names, loglikelihoods, logpriors, param_names)

        # Test with model 0 (linear)
        x = np.array([1.0, 2.0, 999.0, 999.0, 999.0, 0.0])  # nmodel = 0
        result = ps.loglikelihood(x)

        # Should use linear_loglike with params [1.0, 2.0]
        expected = linear_loglike(np.array([1.0, 2.0]))
        assert np.isclose(result, expected)

    def test_product_space_loglikelihood_model_1(self):
        """Test ProductSpace loglikelihood for model 1"""
        model_names = ["linear", "quadratic"]

        def linear_loglike(x):
            return -0.5 * np.sum(x**2)

        def quad_loglike(x):
            return -0.25 * np.sum(x**2)

        def dummy_prior(x):
            return 0.0

        loglikelihoods = [linear_loglike, quad_loglike]
        logpriors = [dummy_prior, dummy_prior]
        param_names = [["a", "b"], ["a", "b", "c"]]

        ps = ProductSpace(model_names, loglikelihoods, logpriors, param_names)

        # Test with model 1 (quadratic)
        x = np.array([999.0, 999.0, 1.0, 2.0, 3.0, 1.0])  # nmodel = 1
        result = ps.loglikelihood(x)

        # Should use quad_loglike with params [1.0, 2.0, 3.0]
        expected = quad_loglike(np.array([1.0, 2.0, 3.0]))
        assert np.isclose(result, expected)

    def test_product_space_logprior_model_0(self):
        """Test ProductSpace logprior for model 0"""
        model_names = ["linear", "quadratic"]

        def dummy_loglike(x):
            return 0.0

        def linear_prior(x):
            return 0.0 if np.all(x >= 0) else -np.inf

        def quad_prior(x):
            return 0.0 if np.all(x <= 1) else -np.inf

        loglikelihoods = [dummy_loglike, dummy_loglike]
        logpriors = [linear_prior, quad_prior]
        param_names = [["a", "b"], ["a", "b", "c"]]

        ps = ProductSpace(model_names, loglikelihoods, logpriors, param_names)

        # Test with model 0 and valid parameters
        x = np.array([1.0, 2.0, 999.0, 999.0, 999.0, 0.0])  # nmodel = 0, positive values
        result = ps.logprior(x)
        assert result == 0.0

        # Test with model 0 and invalid parameters
        x = np.array([-1.0, 2.0, 999.0, 999.0, 999.0, 0.0])  # nmodel = 0, negative value
        result = ps.logprior(x)
        assert result == -np.inf

    def test_product_space_logprior_invalid_model(self):
        """Test ProductSpace logprior with invalid model index"""
        model_names = ["model1"]

        def dummy_func(x):
            return 0.0

        loglikelihoods = [dummy_func]
        logpriors = [dummy_func]
        param_names = [["param1"]]

        ps = ProductSpace(model_names, loglikelihoods, logpriors, param_names)

        # Test with invalid model index
        x = np.array([1.0, 2.0])  # nmodel = 2, but only model 0 exists
        result = ps.logprior(x)
        assert result == -np.inf

    def test_product_space_logprior_inactive_model_ignored(self):
        """Test ProductSpace logprior only evaluates the active model's prior"""
        model_names = ["model1", "model2"]

        def dummy_loglike(x):
            return 0.0

        def good_prior(x):
            return 0.0

        def bad_prior(x):
            return -np.inf  # Always reject

        loglikelihoods = [dummy_loglike, dummy_loglike]
        logpriors = [good_prior, bad_prior]  # Second model always rejects
        param_names = [["a"], ["b"]]

        ps = ProductSpace(model_names, loglikelihoods, logpriors, param_names)

        # With model 0 selected, inactive model 1's prior should not matter
        x = np.array([1.0, 999.0, 0.0])  # nmodel = 0
        result = ps.logprior(x)
        assert result == 0.0

        # With model 1 selected, its prior rejects everything
        x = np.array([1.0, 999.0, 1.0])  # nmodel = 1
        result = ps.logprior(x)
        assert result == -np.inf

    def test_product_space_complex_scenario(self):
        """Test ProductSpace with more complex realistic scenario"""
        model_names = ["polynomial_1", "polynomial_2", "exponential"]

        def poly1_loglike(x):  # a*x
            a = x[0]
            return -(a**2)  # Prefer a close to 0

        def poly2_loglike(x):  # a*x + b*x^2
            a, b = x[0], x[1]
            return -(a**2 + b**2)

        def exp_loglike(x):  # a*exp(b*x)
            a, b = x[0], x[1]
            return -(a**2 + b**2)

        def uniform_prior(x):
            return 0.0 if np.all(np.abs(x) <= 5) else -np.inf

        loglikelihoods = [poly1_loglike, poly2_loglike, exp_loglike]
        logpriors = [uniform_prior, uniform_prior, uniform_prior]
        param_names = [["a"], ["a", "b"], ["a", "b"]]

        ps = ProductSpace(model_names, loglikelihoods, logpriors, param_names)

        # Test dimensions
        assert ps.num_models == 3
        assert ps.ndim == 6  # a_poly1 + a_poly2 + b_poly2 + a_exp + b_exp + nmodel = 6

        # Test model parameter mapping
        assert ps.model_params[0] == [0]  # poly1: a_polynomial_1
        assert ps.model_params[1] == [1, 2]  # poly2: a_polynomial_2, b_polynomial_2
        assert ps.model_params[2] == [3, 4]  # exp: a_exponential, b_exponential

        # Test likelihood evaluation for each model
        x_poly1 = np.array([2.0, 999.0, 999.0, 999.0, 999.0, 0.0])  # model 0
        x_poly2 = np.array([999.0, 1.0, 2.0, 999.0, 999.0, 1.0])  # model 1
        x_exp = np.array([999.0, 999.0, 999.0, 1.0, 2.0, 2.0])  # model 2

        like_poly1 = ps.loglikelihood(x_poly1)
        like_poly2 = ps.loglikelihood(x_poly2)
        like_exp = ps.loglikelihood(x_exp)

        assert np.isclose(like_poly1, -4.0)  # -(2^2)
        assert np.isclose(like_poly2, -5.0)  # -(1^2 + 2^2)
        assert np.isclose(like_exp, -5.0)  # -(1^2 + 2^2)


class TestNestedProductSpace:
    """Test suite for NestedProductSpace class"""

    def test_nested_product_space_init(self):
        """Test NestedProductSpace initialization"""

        def mock_loglike(x):
            return -np.sum(x**2)

        def mock_logprior(x):
            return 0.0

        nps = NestedProductSpace(
            loglikelihood=mock_loglike, logprior=mock_logprior, num_sources=3, num_params=2
        )

        assert nps.loglikelihood is mock_loglike
        assert nps.logprior is mock_logprior
        assert nps.num_models == 3
        np.testing.assert_array_equal(nps.nmodels, np.array([0, 1, 2]))
        assert nps.num_params == 2
        assert nps.ndim == 7  # 3 sources * 2 params + 1 model index

    def test_nested_product_space_get_loglikelihood_basic(self):
        """Test NestedProductSpace get_loglikelihood basic functionality"""

        def mock_loglike(x):
            return -np.sum(x**2)

        def mock_logprior(x):
            return 0.0

        nps = NestedProductSpace(mock_loglike, mock_logprior, num_sources=2, num_params=2)

        # Test with 1 source (nmodel = 0)
        params = np.array([1.0, 2.0, 999.0, 999.0, 0.0])  # 2 active params + unused + nmodel
        result = nps.get_loglikelihood(params)

        # Should use first 2 parameters: [1.0, 2.0]
        expected = mock_loglike(np.array([1.0, 2.0]))
        assert np.isclose(result, expected)

    def test_nested_product_space_get_loglikelihood_multiple_sources(self):
        """Test NestedProductSpace with multiple active sources"""

        def mock_loglike(x):
            return -np.sum(x**2)

        def mock_logprior(x):
            return 0.0

        nps = NestedProductSpace(mock_loglike, mock_logprior, num_sources=3, num_params=2)

        # Test with 2 sources (nmodel = 1, so 0 and 1 are active)
        params = np.array([1.0, 2.0, 3.0, 4.0, 999.0, 999.0, 1.0])
        result = nps.get_loglikelihood(params)

        # Should use first 4 parameters: [1.0, 2.0, 3.0, 4.0]
        expected = mock_loglike(np.array([1.0, 2.0, 3.0, 4.0]))
        assert np.isclose(result, expected)

    def test_nested_product_space_get_loglikelihood_all_sources(self):
        """Test NestedProductSpace with all sources active"""

        def mock_loglike(x):
            return -np.sum(x**2)

        def mock_logprior(x):
            return 0.0

        nps = NestedProductSpace(mock_loglike, mock_logprior, num_sources=2, num_params=3)

        # Test with all sources (nmodel = 1, so sources 0 and 1 are active)
        params = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0])
        result = nps.get_loglikelihood(params)

        # Should use all 6 parameters (2 sources * 3 params)
        expected = mock_loglike(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        assert np.isclose(result, expected)

    def test_nested_product_space_get_logprior_basic(self):
        """Test NestedProductSpace get_logprior only uses active source params"""

        def mock_loglike(x):
            return 0.0

        def mock_logprior(x):
            return -np.sum(x**2) if len(x) > 0 else 0.0

        nps = NestedProductSpace(mock_loglike, mock_logprior, num_sources=2, num_params=2)

        # Test with nmodel = 0 (1 active source, first 2 params)
        params = np.array([1.0, 2.0, 3.0, 4.0, 0.0])
        result = nps.get_logprior(params)
        expected = mock_logprior(np.array([1.0, 2.0]))
        assert np.isclose(result, expected)

        # Test with nmodel = 1 (2 active sources, first 4 params)
        params = np.array([1.0, 2.0, 3.0, 4.0, 1.0])
        result = nps.get_logprior(params)
        expected = mock_logprior(np.array([1.0, 2.0, 3.0, 4.0]))
        assert np.isclose(result, expected)

    def test_nested_product_space_get_logprior_invalid_model(self):
        """Test NestedProductSpace get_logprior with invalid model"""

        def mock_loglike(x):
            return 0.0

        def mock_logprior(x):
            return 0.0

        nps = NestedProductSpace(mock_loglike, mock_logprior, num_sources=2, num_params=2)

        # Test with invalid nmodel (outside valid range)
        params = np.array([1.0, 2.0, 3.0, 4.0, 3.0])  # nmodel = 3, but only 0,1 are valid
        result = nps.get_logprior(params)

        assert result == -np.inf

    def test_nested_product_space_edge_cases(self):
        """Test NestedProductSpace edge cases"""

        def mock_loglike(x):
            return -np.sum(x**2) if len(x) > 0 else 0.0

        def mock_logprior(x):
            return 0.0 if np.all(np.abs(x) <= 5) else -np.inf

        # Test with single source, single parameter
        nps = NestedProductSpace(mock_loglike, mock_logprior, num_sources=1, num_params=1)

        assert nps.ndim == 2  # 1 source * 1 param + 1 model index

        # Test with 0 sources active (nmodel = -1, but this would be invalid)
        # Actually, nmodel should be in range [0, num_sources-1]
        params = np.array([1.0, 0.0])  # 1 param + nmodel = 0

        # With nmodel = 0, should use first 1 parameter
        like_result = nps.get_loglikelihood(params)
        prior_result = nps.get_logprior(params)

        expected_like = mock_loglike(np.array([1.0]))
        expected_prior = mock_logprior(np.array([1.0]))

        assert np.isclose(like_result, expected_like)
        assert np.isclose(prior_result, expected_prior)

    def test_nested_product_space_realistic_scenario(self):
        """Test NestedProductSpace with realistic astrophysical scenario"""

        def multi_source_loglike(params):
            """Mock likelihood for multiple point sources"""
            # Each source has [x, y, flux] parameters
            n_sources = len(params) // 3
            total_like = 0.0

            for i in range(n_sources):
                x, y, flux = params[3 * i : 3 * (i + 1)]
                # Penalize sources far from center and negative flux
                total_like += -(x**2 + y**2) - max(0, -flux) ** 2

            return total_like

        def multi_source_logprior(params):
            """Mock prior for multiple sources"""
            n_sources = len(params) // 3

            for i in range(n_sources):
                x, y, flux = params[3 * i : 3 * (i + 1)]
                # Position constraints
                if abs(x) > 10 or abs(y) > 10:
                    return -np.inf
                # Flux must be positive
                if flux <= 0:
                    return -np.inf

            return 0.0

        # Up to 3 sources, 3 parameters each
        nps = NestedProductSpace(
            loglikelihood=multi_source_loglike,
            logprior=multi_source_logprior,
            num_sources=3,
            num_params=3,
        )

        assert nps.ndim == 10  # 3*3 + 1

        # Test 1 source scenario
        params_1src = np.array([1.0, 2.0, 5.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 0.0])
        like_1src = nps.get_loglikelihood(params_1src)
        prior_1src = nps.get_logprior(params_1src)

        # Should be finite (valid source)
        assert np.isfinite(like_1src)
        assert prior_1src == 0.0

        # Test 2 source scenario
        params_2src = np.array([1.0, 2.0, 5.0, -1.0, 3.0, 2.0, 999.0, 999.0, 999.0, 1.0])
        like_2src = nps.get_loglikelihood(params_2src)
        prior_2src = nps.get_logprior(params_2src)

        assert np.isfinite(like_2src)
        assert prior_2src == 0.0

        # Test invalid scenario (negative flux)
        params_bad = np.array([1.0, 2.0, -1.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 0.0])
        prior_bad = nps.get_logprior(params_bad)

        assert prior_bad == -np.inf
