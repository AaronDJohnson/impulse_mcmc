import numpy as np
from dataclasses import dataclass
import ray

@dataclass
class PTState():
    """
    Parallel Tempering State
    Contains the PT ladder, the swap acceptance rate, and the number of swaps
    """
    ndim: int
    ntemps: int
    swap_steps: int = 100
    min_temp: float = 1.0
    max_temp: float = None
    temp_step: float = None
    swap_accept: float = 0.0
    nswaps: int = 1  # start at 1 to avoid divide by zero errors
    ladder: np.ndarray = None
    inf_temp: bool = False
    # adaptive temperature ladder parameters:
    adapt_t0: float = 100
    adapt_nu: float = 10

    def __post_init__(self):
        if self.ladder is None:
            self.ladder = self.temp_ladder()

    def compute_accept_ratio(self):
        return self.swap_accept / self.nswaps

    def temp_ladder(self):
        """
        Method to compute temperature ladder. At the moment this uses
        a geometrically spaced temperature ladder with a temperature
        spacing designed to give 25% temperature swap acceptance rate
        on a multi-variate Gaussian.
        """
        if self.inf_temp:
            self.ntemps -= 1  # remove top value from ladder
        if self.temp_step is None and self.max_temp is None:
            self.temp_step = 1 + np.sqrt(2 / self.ndim)
        elif self.temp_step is None and self.max_temp is not None:
            self.temp_step = np.exp(np.log(self.max_temp / self.min_temp) / (self.ntemps - 1))
        temp_idxs = np.arange(self.ntemps)
        if self.inf_temp:
            self.ntemps += 1 # add empty top value back to ladder
            ladder = self.min_temp * self.temp_step**temp_idxs  # compute ladder
            ladder = np.concatenate([ladder, [np.inf]])  # add inf value as top of ladder
        else:
            ladder = self.min_temp * self.temp_step**temp_idxs
        return ladder

    def adapt_ladder(self):
        """
        Adapt temperatures according to arXiv:1501.05823 <https://arxiv.org/abs/1501.05823>.
        """
        # Temperature adjustments with a hyperbolic decay.
        decay = self.adapt_t0 / (self.nswaps + self.adapt_t0)  # t0 / (t + t0)
        kappa = decay / self.adapt_nu  # 1 / nu
        # Construct temperature adjustments.
        accept_ratio = self.compute_accept_ratio()
        dscaled_accept = kappa * (accept_ratio[:-1] - accept_ratio[1:])  # delta acceptance ratios for chains
        # Compute new ladder (hottest and coldest chains don't move).
        delta_temps = np.diff(self.ladder[:-1])
        delta_temps *= np.exp(dscaled_accept)
        self.ladder[1:-1] = (np.cumsum(delta_temps) + self.ladder[0])

@ray.remote
class Collection():
    def __init__(self):
        self.p0s = []
        self.log_Ls = []

    def append_p0(self, item):
        self.p0s.append(item)

    def append_log_L(self, item):
        self.log_Ls.append(item)

    def get_all(self):
        return self.p0s, self.log_Ls

    def clear(self):
        self.p0s = []
        self.log_Ls = []


@ray.remote
class PTSwap():

    def __init__(self, ndim, ntemps, rng, tmin=1, tmax=None, tstep=None,
                 tinf=False, adapt_t0=100, adapt_nu=10, swap_steps=100,
                 ladder=None):
        self.ndim = ndim
        self.ntemps = ntemps
        self.tmin = tmin
        self.tmax = tmax
        self.tstep = tstep
        self.rng = rng
        if ladder is None:
            self.ladder = self.temp_ladder()
        else:
            self.ladder = ladder
            self.ntemps = len(ladder)
        if tinf:
            self.ladder[-1] = np.inf
        self.swap_steps = swap_steps
        self.swap_accept = np.zeros(ntemps - 1)  # swap acceptance between chains
        self.adapt_t0 = adapt_t0
        self.adapt_nu = adapt_nu
        self.nswaps = 0

        self.collection = Collection.remote()

    def get_temp_ladder(self):
        return self.ladder

    def get_swap_steps(self):
        return self.swap_steps

    def compute_accept_ratio(self):
        return self.swap_accept / self.nswaps

    def swap(self, p0s, log_Ls):
        """
        Repurposed from Neil Cornish/Bence Becsy's code:
        """
        # self.collection.append_p0.remote(p0)
        # self.collection.append_log_L.remote(log_L)
        # p0s, log_Ls = ray.get(self.collection.get_all.remote())

        Ts = self.ladder

        # set up map to help keep track of swaps
        swap_map = list(range(self.ntemps))

        # loop through and propose a swap at each chain (starting from hottest chain and going down in T)
        # and keep track of results in swap_map
        for swap_chain in reversed(range(self.ntemps - 1)):
            log_acc_ratio = -log_Ls[swap_map[swap_chain]] / Ts[swap_chain]
            log_acc_ratio += -log_Ls[swap_map[swap_chain + 1]] / Ts[swap_chain + 1]
            log_acc_ratio += log_Ls[swap_map[swap_chain + 1]] / Ts[swap_chain]
            log_acc_ratio += log_Ls[swap_map[swap_chain]] / Ts[swap_chain + 1]

            acc_ratio = np.exp(log_acc_ratio)
            if self.rng.uniform() <= acc_ratio:
                swap_map[swap_chain], swap_map[swap_chain + 1] = swap_map[swap_chain + 1], swap_map[swap_chain]
                self.swap_accept[swap_chain] += 1
                self.nswaps += 1
            else:
                self.nswaps += 1

        # loop through the chains and record the new samples and log_Ls
        for jj in range(self.ntemps):
            p0s[jj] = p0s[swap_map[jj]]
            log_Ls[jj] = log_Ls[swap_map[jj]]

        return p0s, log_Ls

    # def __call__(self, chain, lnlike, lnprob, swap_idx):  # propose swaps!
    #     self.nswaps += 1
    #     for i in range(1, self.ntemps):
    #         dbeta = (1 / self.ladder[i - 1] - 1 / self.ladder[i])
    #         raccept = np.log(rng.random())
    #         paccept = dbeta * (lnlike[swap_idx, i] - lnlike[swap_idx, i - 1])

    #         if paccept > raccept:
    #             self.swap_accept[i - 1] += 1
    #             chain_temp = np.copy(chain[swap_idx, :, i])
    #             logl_temp = np.copy(lnlike[swap_idx, i])
    #             # logp_temp = np.copy(lnprob[swap_idx, i])

    #             chain[swap_idx, :, i] = chain[swap_idx, :, i - 1]
    #             lnlike[swap_idx, i] = lnlike[swap_idx, i - 1]
    #             # lnprob[swap_idx, i] = lnprob[swap_idx, i - 1] - dbeta * lnlike[swap_idx, i - 1]

    #             chain[swap_idx, :, i - 1] = chain_temp
    #             lnlike[swap_idx, i - 1] = logl_temp
    #             # lnprob[swap_idx, i - 1] = logp_temp + dbeta * logl_temp
    #     return chain, lnlike, lnprob
