import numpy as np
from numpy.random import SeedSequence, default_rng
from tqdm import tqdm
from loguru import logger
import ray

from impulse.ptsampler import PTSwap
from impulse.proposals import JumpProposals, am, scam, de
from impulse.save_data import SaveData
from impulse.mhsampler import mh_step

class Sampler():
    def __init__(self,
                 ndim,
                 logl,
                 logp,
                 ncores=1,
                 ntemps=1,
                 tmin=1,
                 tmax=None,
                 tstep=None,
                 tinf=False,
                 adapt=False,
                 adapt_t0=100,
                 adapt_nu=10,
                 ladder=None,
                 seed=None,
                 buf_size=50_000,
                 mean=None,
                 cov=None,
                 groups=None,
                 loglargs=[],
                 loglkwargs={},
                 logpargs=[],
                 logpkwargs={},
                 cov_update=1000,
                 save_freq=1000,
                 SCAMweight=30,
                 AMweight=15,
                 DEweight=50,
                 outdir="./chains"):

        self.ndim = ndim
        self.ncores = ncores
        self.ntemps = ntemps
        self.tmin = tmin
        self.tmax = tmax
        self.tstep = tstep
        self.tinf = tinf
        self.adapt = adapt
        self.adapt_t0 = adapt_t0
        self.adapt_nu = adapt_nu
        self.ladder = ladder

        # setup random number generator
        stream = SeedSequence(seed)
        seeds = stream.generate_state(ntemps + 1)
        self.rng = [default_rng(s) for s in seeds]

        # setup parallel tempering
        self.ptswap = PTSwap(self.ndim, self.ntemps, self.rng[-1], tmin=self.tmin, tmax=self.tmax, tstep=self.tstep,
                             tinf=self.tinf, adapt_t0=self.adapt_t0, adapt_nu=self.adapt_nu, ladder=self.ladder)

        # setup samplers:
        samplers = [PTSampler.remote(self.ndim, logl, logp, ii, self.ptswap.ladder[ii], buf_size, mean, cov, groups,
                                     loglargs, loglkwargs, logpargs, logpkwargs, cov_update, save_freq, SCAMweight,
                                     AMweight, DEweight, outdir, self.rng[ii], self.ptswap) for ii in range(self.ntemps)]


        def sample(x0, num_samples, thin=1, resume=False):
            [sampler.sample.remote(x0, num_samples, thin=thin) for sampler in samplers]

        # # swap sometimes
        # if self.counter % self.swap_steps == 0 and self.counter > 1:
        #     self.x0s, self.lnlike0s = self.ptswap.swap(self.chain[kk, :, :], self.lnlike_arr[kk, :])
        #     for jj in range(self.ntemps):
        #         self.x0[jj] = self.x0s[jj]
        #         self.lnlike0[jj] = self.lnlike0s[jj]
        #         self.lnprob0[jj] = self.logp(self.x0[jj]) + self.lnlike0[jj]

        #         self.chain[kk, :, jj], self.lnlike_arr[kk, jj] = self.x0s[jj], self.lnlike0s[jj]
        #         self.lnprob0[jj] = 1 / self.ptswap.ladder[jj] * self.lnlike0[jj] + self.logp(self.x0[jj])
        #         self.lnprob_arr[kk, jj] = self.lnprob0[jj]

        # # adaptive temperature spacing
        # if self.adapt:
        #     self.ptswap.adapt_ladder()
        #     self.save.save_swap_data(self.ptswap)

        def save_state(self):
            pass

        def load_state(self):
            pass
        

@ray.remote(num_cpus=1)
class PTSampler():
    def __init__(self,
                 ndim,
                 logl,
                 logp,
                 chain_num=1,
                 temperature=1,
                 buf_size=50_000,
                 mean=None,
                 cov=None,
                 groups=None,
                 loglargs=[],
                 loglkwargs={},
                 logpargs=[],
                 logpkwargs={},
                 cov_update=1000,
                 save_freq=1000,
                 SCAMweight=30,
                 AMweight=15,
                 DEweight=50,
                 outdir="./chains",
                 rng=np.random.default_rng(),):

        # setup loglikelihood and logprior functions
        self.logl = _function_wrapper(logl, loglargs, loglkwargs)
        self.logp = _function_wrapper(logp, logpargs, logpkwargs)

        # PTSwap parameters
        self.chain_num = chain_num  # number of this chain
        self.temp = temperature  # temperature of this chain

        # other important bits
        self.outdir = outdir
        self.ndim = ndim
        self.cov_update = cov_update
        self.save_freq = save_freq
        self.rng = rng

        # sample counter
        self.counter = 0

        # setup standard jump proposals
        self.mix = JumpProposals(self.ndim, buf_size=buf_size, groups=groups, cov=cov, mean=mean)
        self.mix.add_proposal(scam, SCAMweight)
        self.mix.add_proposal(am, AMweight)
        self.mix.add_proposal(de, DEweight)


    def sample(self, x0, num_samples, thin=1):

        # initial sample
        self.x0 = np.array(x0)  # (ntemps, ndim)
        self.lnlike0 = self.logl(self.x0)
        self.lnprob0 = self.logp(self.x0) + self.lnlike0

        # setup save
        self.filename = '/chain_{}.txt'.format(self.chain_num) # temps change (label by chain number)
        self.save = SaveData(outdir=self.outdir, filename=self.filename, thin=thin)

        # setup arrays
        self.chain = np.zeros((self.save_freq, self.ndim))
        self.lnlike_arr = np.zeros(self.save_freq)
        self.lnprob_arr = np.zeros(self.save_freq)
        self.accept_arr = np.zeros(self.save_freq)

        # start sampling!
        for ii in tqdm(range(num_samples)):
            kk = ii % self.save_freq
            self.counter += 1

            # metropolis hastings step + update chains
            self.x0, self.lnlike0, self.lnprob0, self.accept = mh_step(self.x0, self.lnlike0, self.lnprob0, self.logl,
                                                                       self.logp, self.mix, self.rng, self.temp)
            self.chain[kk, :] = self.x0
            self.lnlike_arr[kk] = self.lnlike0
            self.lnprob_arr[kk] = self.lnprob0
            self.accept_arr[kk] = self.accept

            # update covariance matrix
            if self.counter % self.cov_update == 0 and self.counter > 1:
                self.mix.recursive_update(self.counter, self.chain)

            # save and save state of PTSampler
            if self.counter % self.save_freq == 0 and self.counter > 1:
                self.save(self.chain, self.lnlike_arr, self.lnprob_arr, self.accept_arr)


class _function_wrapper(object):

    """
    This is a hack to make the likelihood function pickleable when ``args``
    or ``kwargs`` are also included.
    """

    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        return self.f(x, *self.args, **self.kwargs)
