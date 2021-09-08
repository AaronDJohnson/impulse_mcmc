import numpy as np
import os
from multiprocessing import Pool
from itertools import repeat

from impulse.sampler import MHSampler
from impulse.proposals import pre_proposals, stock_proposals
from impulse.save_hdf import save_h5


# This is a work in progress....

def temp_ladder(ndim, nchain, temp_min, temp_max=None, temp_step=None):
    """
    Method to compute temperature ladder. At the moment this uses
    a geometrically spaced temperature ladder with a temperature
    spacing designed to give 25 % temperature swap acceptance rate.
    """

    # TODO: make options to do other temperature ladders

    if nchain > 1:
        if temp_step is None and temp_max is None:
            temp_step = 1 + np.sqrt(2 / ndim)
        elif temp_step is None and temp_max is not None:
            temp_step = np.exp(np.log(temp_max / temp_min) / (nchain - 1))
        ladder = np.zeros(nchain)
        for ii in range(nchain - 1):
            ladder[ii] = temp_min * temp_step**ii
        ladder[-1] = 1e80
    else:
        ladder = np.array([1])

    return ladder


def pt_save_samples(temp, chain, lnprob, accept_rate, outdir):
    # make directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    filename = outdir + '/chain_{}.txt'.format(temp)
    data = np.column_stack((chain, lnprob, accept_rate))
    with open(filename, 'a+') as fname:
        np.savetxt(fname, data)


def pt_pre_sample(post, x0, mix, num_samples, loop_iterations, full_chain):
    sampler = MHSampler(x0, post, mix, iterations=loop_iterations)
    chain, accept, lnprob = sampler.sample()
    pt_save_samples(temp, chain, lnprob, accept_rate, outdir)
    full_chain[chain_num, count:count + loop_iterations, :] = chain
    mix.update(loop_iterations, chain)
    x0 = chain[-1]
    return x0, full_chain, mix


def pre_worker(tasks):
    post, x0, mix, num_samples, loop_iterations = tasks
    x0, full_chain, mix = pt_pre_sample(post, x0, mix, num_samples, loop_iterations, full_chain)
    return x0, full_chain, mix


# def pt_sample_step(post, x0, mix, num_samples, loop_iterations):
    



class PTPosterior():
    def __init__(self, temp, lnlikelihood, lnprior):
        self.temp = temp
        self.lnlikelihood = lnlikelihood
        self.lnprior = lnprior

    def __call__(self, x):
        ptposterior = self.lnlikelihood(x) * (1 / self.temp) + self.lnprior(x)
        return ptposterior



def PTSampler():
    def __init__(self, ndim, nchain, lnlikelihood, lnprior, temp_min=1, temp_max=None, temp_step=None, pool=None):
        self.pool = pool
        self.ndim = ndim
        temps = temp_ladder(ndim, nchain, temp_min, temp_max=temp_max, temp_step=temp_step)
        self.post_list = []
        for temp in temps:
            self.post_list.append(PTPosterior(temp, lnlikelihood, lnprior))

    def sample(self, post, x0, num_samples=100_000, loop_iterations=1000, save=True, outdir='./test', compress=True):
        mix = pre_proposals(self.ndim)
        full_chain = np.zeros((nchain, num_samples, ndim))
        count = 0
        while count < int(num_samples / 100):
            with self.pool as pool:
                pool.starmap()
                x0, full_chain, mix = pt_pre_sample(post, x0, mix, num_samples, loop_iterations)
                pool.join()
        count += loop_iterations
        # add DE jump:
        mix = stock_proposals(ndim, full_chain)
        for i in tqdm(range(int(num_samples / 100), int(num_samples), loop_iterations)):
            sampler = MHSampler(x0, post, mix, iterations=loop_iterations)
            chain, accept, lnprob = sampler.sample()
            sampler.save_samples(outdir)
            full_chain[count:count + loop_iterations, :] = chain
            mix.update(loop_iterations, chain, full_chain=full_chain)
            x0 = chain[-1]
            count += loop_iterations


# def worker(task):
#     post, ndim, x0 = task
#     return sample(post, ndim, x0)


# def test(post, ndim, x0):
#     tasks = zip(post, repeat(ndim), repeat(x0))
#     with Pool(processes=2) as pool:
#         res = pool.map(worker, tasks)


# def make_ladder(ndim, ntemps=None, Tmax=None):
#     """
#     Returns a ladder of :math:`\\beta \\equiv 1/T` under a geometric spacing that is determined by the
#     arguments ``ntemps`` and ``Tmax``.  The temperature selection algorithm works as follows:

#     Ideally, ``Tmax`` should be specified such that the tempered posterior looks like the prior at
#     this temperature.  If using adaptive parallel tempering, per `arXiv:1501.05823
#     <http://arxiv.org/abs/1501.05823>`_, choosing ``Tmax = inf`` is a safe bet, so long as
#     ``ntemps`` is also specified.

#     :param ndim:
#         The number of dimensions in the parameter space.

#     :param ntemps: (optional)
#         If set, the number of temperatures to generate.

#     :param Tmax: (optional)
#         If set, the maximum temperature for the ladder.

#     Temperatures are chosen according to the following algorithm:

#     * If neither ``ntemps`` nor ``Tmax`` is specified, raise an exception (insufficient
#       information).
#     * If ``ntemps`` is specified but not ``Tmax``, return a ladder spaced so that a Gaussian
#       posterior would have a 25% temperature swap acceptance ratio.
#     * If ``Tmax`` is specified but not ``ntemps``:

#       * If ``Tmax = inf``, raise an exception (insufficient information).
#       * Else, space chains geometrically as above (for 25% acceptance) until ``Tmax`` is reached.

#     * If ``Tmax`` and ``ntemps`` are specified:

#       * If ``Tmax = inf``, place one chain at ``inf`` and ``ntemps-1`` in a 25% geometric spacing.
#       * Else, use the unique geometric spacing defined by ``ntemps`` and ``Tmax``.

#     """

#     if type(ndim) != int or ndim < 1:
#         raise ValueError('Invalid number of dimensions specified.')
#     if ntemps is None and Tmax is None:
#         raise ValueError('Must specify one of ``ntemps`` and ``Tmax``.')
#     if Tmax is not None and Tmax <= 1:
#         raise ValueError('``Tmax`` must be greater than 1.')
#     if ntemps is not None and (type(ntemps) != int or ntemps < 1):
#         raise ValueError('Invalid number of temperatures specified.')

#     tstep = np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
#                       2.71225, 2.49879, 2.34226, 2.22198, 2.12628,
#                       2.04807, 1.98276, 1.92728, 1.87946, 1.83774,
#                       1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
#                       1.66657, 1.64647, 1.62795, 1.61083, 1.59494,
#                       1.58014, 1.56632, 1.55338, 1.54123, 1.5298,
#                       1.51901, 1.50881, 1.49916, 1.49, 1.4813,
#                       1.47302, 1.46512, 1.45759, 1.45039, 1.4435,
#                       1.4369, 1.43056, 1.42448, 1.41864, 1.41302,
#                       1.40761, 1.40239, 1.39736, 1.3925, 1.38781,
#                       1.38327, 1.37888, 1.37463, 1.37051, 1.36652,
#                       1.36265, 1.35889, 1.35524, 1.3517, 1.34825,
#                       1.3449, 1.34164, 1.33847, 1.33538, 1.33236,
#                       1.32943, 1.32656, 1.32377, 1.32104, 1.31838,
#                       1.31578, 1.31325, 1.31076, 1.30834, 1.30596,
#                       1.30364, 1.30137, 1.29915, 1.29697, 1.29484,
#                       1.29275, 1.29071, 1.2887, 1.28673, 1.2848,
#                       1.28291, 1.28106, 1.27923, 1.27745, 1.27569,
#                       1.27397, 1.27227, 1.27061, 1.26898, 1.26737,
#                       1.26579, 1.26424, 1.26271, 1.26121,
#                       1.25973])

#     if ndim > tstep.shape[0]:
#         # An approximation to the temperature step at large
#         # dimension
#         tstep = 1.0 + 2.0*np.sqrt(np.log(4.0))/np.sqrt(ndim)
#     else:
#         tstep = tstep[ndim-1]

#     appendInf = False
#     if Tmax == np.inf:
#         appendInf = True
#         Tmax = None
#         ntemps = ntemps - 1

#     if ntemps is not None:
#         if Tmax is None:
#             # Determine Tmax from ntemps.
#             Tmax = tstep ** (ntemps - 1)
#     else:
#         if Tmax is None:
#             raise ValueError('Must specify at least one of ``ntemps'' and '
#                              'finite ``Tmax``.')

#         # Determine ntemps from Tmax.
#         ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

#     betas = np.logspace(0, -np.log10(Tmax), ntemps)
#     if appendInf:
#         # Use a geometric spacing, but replace the top-most temperature with
#         # infinity.
#         betas = np.concatenate((betas, [0]))

#     return betas



