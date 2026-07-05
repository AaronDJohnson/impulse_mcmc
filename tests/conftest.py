import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from impulse.chain_stats import ChainStats
from impulse.sampler_state import PTState, SamplerState


@pytest.fixture
def temp_dir():
    """Temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def simple_likelihood():
    """Simple quadratic likelihood function"""

    def log_likelihood(x):
        x = np.asarray(x)
        if x.ndim == 1:
            return -0.5 * np.sum(x**2)
        else:
            return -0.5 * np.sum(x**2, axis=1)

    return log_likelihood


@pytest.fixture
def simple_prior():
    """Simple uniform prior"""

    def log_prior(x):
        x = np.asarray(x)
        if x.ndim == 1:
            return 0.0 if np.all(np.abs(x) <= 5) else -np.inf
        else:
            result = np.zeros(x.shape[0])
            mask = np.any(np.abs(x) > 5, axis=1)
            result[mask] = -np.inf
            return result

    return log_prior


@pytest.fixture
def vectorized_likelihood():
    """Vectorized likelihood function"""

    def log_likelihood(x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return -0.5 * np.sum(x**2, axis=1)

    return log_likelihood


@pytest.fixture
def vectorized_prior():
    """Vectorized prior function"""

    def log_prior(x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        result = np.zeros(x.shape[0])
        mask = np.any(np.abs(x) > 5, axis=1)
        result[mask] = -np.inf
        return result

    return log_prior


@pytest.fixture
def ptstate_2d():
    """2D PTState fixture"""
    return PTState(ndim=2, ntemps=3, min_temp=1.0, max_temp=4.0)


@pytest.fixture
def ptstate_3d():
    """3D PTState fixture"""
    return PTState(ndim=3, ntemps=5, min_temp=1.0, max_temp=16.0)


@pytest.fixture
def sample_state_2d(ptstate_2d):
    """Sample SamplerState for 2D case"""
    positions = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
    lnlikes = np.array([-1.0, -1.1, -1.2])
    lnpriors = np.array([0.0, 0.0, 0.0])
    temps = ptstate_2d.ladder
    lnprobs = lnpriors + lnlikes / temps
    accepted = np.array([1, 0, 1])

    return SamplerState(positions, lnlikes, lnpriors, lnprobs, accepted, temps)


@pytest.fixture
def sample_state_3d(ptstate_3d):
    """Sample SamplerState for 3D case"""
    positions = np.random.randn(5, 3) * 0.5
    lnlikes = -0.5 * np.sum(positions**2, axis=1)
    lnpriors = np.zeros(5)
    temps = ptstate_3d.ladder
    lnprobs = lnpriors + lnlikes / temps
    accepted = np.ones(5, dtype=int)

    return SamplerState(positions, lnlikes, lnpriors, lnprobs, accepted, temps)


@pytest.fixture
def chain_stats_2d(ptstate_2d):
    """ChainStats fixture for 2D case"""
    rng = np.random.default_rng(42)
    return ChainStats(ndim=2, pt_state=ptstate_2d, chain_index=0, rng=rng, buffer_size=100)


@pytest.fixture
def sample_chain_data():
    """Sample MCMC chain data for testing diagnostics"""
    np.random.seed(42)
    n_samples = 1000
    n_params = 3

    # Generate autocorrelated samples
    samples = np.zeros((n_samples, n_params))
    samples[0] = np.random.randn(n_params)

    for i in range(1, n_samples):
        # AR(1) process with correlation 0.8
        samples[i] = 0.8 * samples[i - 1] + 0.6 * np.random.randn(n_params)

    return samples
