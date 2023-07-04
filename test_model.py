import numpy as np


def make_sin_data(num_pts, omega_real=None, a_real=None, phi_real=None):
    rng = np.random.default_rng()

    if omega_real is None:
        omega_real = 3 * rng.uniform(0, 1)
    else:
        omega_real = omega_real
    if a_real is None:
        a_real = 5 * rng.uniform(0, 1)
    else:
        a_real = a_real
    if phi_real is None:
        phi_real = np.pi * rng.uniform(0, 1)
    else:
        phi_real = phi_real

    x = np.linspace(0, 8*np.pi, num_pts)
    data = rng.standard_normal(num_pts) + a_real * np.sin(omega_real * x + phi_real)
    return x, data, omega_real, a_real, phi_real


# standard least squares likelihood
class LnLikelihood():
    def __init__(self, x, data, sigma=1):
        self.x = x
        self.sigma = sigma
        self.data = data

    def __call__(self, params):
        omega = params[0]
        # a = params[1]
        # phi = params[2]
        a = params[1]
        phi = params[2]
        func = a * np.sin(omega * self.x + phi)
        result = np.sum(-0.5 * (func - self.data)**2)
        if np.isfinite(result):
            return result
        else:
            return -np.inf


# uniform prior
class LnPrior():
    def __init__(self, mins, maxes):
        """
        mins: vector of minima on uniform prior
        maxes: vector of maxima on uniform prior
        """
        self.rng = np.random.default_rng()
        self.mins = mins
        self.maxes = maxes

    def __call__(self, params):
        if np.any(params < self.mins) or np.any(params > self.maxes):
            return -np.inf
        else:
            return 0

    def initial_sample(self):
        return self.rng.uniform(self.mins, self.maxes)


class LnPosterior():
    def __init__(self, mins, maxes, x, data, sigma=1):
        self.lnlike = LnLikelihood(x, data, sigma)
        self.lnprior = LnPrior(mins, maxes)

    def __call__(self, params):
        return self.lnlike(params) + self.lnprior(params)
