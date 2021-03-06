import numpy as np
from impulse.random_nums import rng
from numba import njit


class PTSwap():

    def __init__(self, ndim, ntemps, tmin=1, tmax=None, tstep=None,
                 tinf=False, adapt_t0=100, adapt_nu=10,
                 ladder=None):
        self.ndim = ndim
        self.ntemps = ntemps
        self.tmin = tmin
        self.tmax = tmax
        self.tstep = tstep
        if ladder is None:
            self.ladder = self.temp_ladder()
        else:
            self.ladder = ladder
            self.ntemps = len(ladder)
        if tinf:
            self.ladder[-1] = np.inf
        self.swap_accept = np.zeros(ntemps - 1)  # swap acceptance between chains
        self.adapt_t0 = adapt_t0
        self.adapt_nu = adapt_nu
        self.nswaps = 0


    def temp_ladder(self):
        """
        Method to compute temperature ladder. At the moment this uses
        a geometrically spaced temperature ladder with a temperature
        spacing designed to give 25 % temperature swap acceptance rate.
        """
        if self.tstep is None and self.tmax is None:
            self.tstep = 1 + np.sqrt(2 / self.ndim)
        elif self.tstep is None and self.tmax is not None:
            self.tstep = np.exp(np.log(self.tmax / self.tmin) / (self.ntemps - 1))
        ii = np.arange(self.ntemps)
        ladder = self.tmin * self.tstep**ii
        return ladder


    def adapt_ladder(self):
        """
        Adapt temperatures according to arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>.
        """
        # Modulate temperature adjustments with a hyperbolic decay.
        decay = self.adapt_t0 / (self.nswaps + self.adapt_t0)  # t0 / (t + t0)
        kappa = decay / self.adapt_nu  # 1 / nu
        # Construct temperature adjustments.
        accept_ratio = self.compute_accept_ratio()
        dSs = kappa * (accept_ratio[:-1] - accept_ratio[1:])  # delta acceptance ratios for chains
        # Compute new ladder (hottest and coldest chains don't move).
        deltaTs = np.diff(self.ladder[:-1])
        deltaTs *= np.exp(dSs)
        self.ladder[1:-1] = (np.cumsum(deltaTs) + self.ladder[0])


    def compute_accept_ratio(self):
        return self.swap_accept / self.nswaps


    def __call__(self, chain, lnlike, lnprob, swap_idx):  # propose swaps!
        self.nswaps += 1
        for i in range(1, self.ntemps):
            dbeta = (1 / self.ladder[i - 1] - 1 / self.ladder[i])
            raccept = np.log(rng.random())
            paccept = dbeta * (lnlike[swap_idx, i] - lnlike[swap_idx, i - 1])

            if paccept > raccept:
                self.swap_accept[i - 1] += 1
                chain_temp = np.copy(chain[swap_idx, :, i])
                logl_temp = np.copy(lnlike[swap_idx, i])
                # logp_temp = np.copy(lnprob[swap_idx, i])

                chain[swap_idx, :, i] = chain[swap_idx, :, i - 1]
                lnlike[swap_idx, i] = lnlike[swap_idx, i - 1]
                # lnprob[swap_idx, i] = lnprob[swap_idx, i - 1] - dbeta * lnlike[swap_idx, i - 1]

                chain[swap_idx, :, i - 1] = chain_temp
                lnlike[swap_idx, i - 1] = logl_temp
                # lnprob[swap_idx, i - 1] = logp_temp + dbeta * logl_temp
        return chain, lnlike, lnprob
