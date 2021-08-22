import numpy as np


def make_sin_data(num_pts):
    rng = np.random.default_rng()

    omega_real = 3 * rng.uniform(0, 1)
    a_real = 5 * rng.uniform(0, 1)
    phi_real = 2 * np.pi * rng.uniform(0, 1)

    x = np.linspace(0, 2*np.pi, num_pts)
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
        a = params[1]
        phi = params[2]
        func = a * np.sin(omega * self.x + phi)
        result = np.sum(-(func - self.data)**2)
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
