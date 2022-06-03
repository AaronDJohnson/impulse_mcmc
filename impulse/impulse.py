import numpy as np
from tqdm import tqdm
import os
import ray
from impulse.sampler import MHSampler
from impulse.proposals import JumpProposals, am, scam, de
from impulse.save_hdf import save_h5


def sample(lnlike, lnprior, ndim, x0, num_samples=100_000, buf_size=10000,
           amweight=30, scamweight=15, deweight=50,
           loop_iterations=1000, save=True, outdir='./test', filename='/chain_1.txt', compress=True):
    # set up proposals for burn-in:
    mix = JumpProposals(ndim, buf_size=buf_size)
    mix.add_jump(am, amweight)
    mix.add_jump(scam, scamweight)
    # make empty full chain
    full_chain = np.zeros((num_samples, ndim))
    # set up count and iterations between loops
    count = 0
    while count < buf_size:  # fill the buffer
        sampler = MHSampler(x0, lnlike, lnprior, mix, iterations=loop_iterations)
        chain, accept, lnprob = sampler.sample()
        sampler.save_samples(outdir, filename=filename)
        full_chain[count:count + loop_iterations, :] = chain
        mix.recursive_update(count, chain)
        x0 = chain[-1]
        count += loop_iterations
    # add DE jump:
    mix.add_jump(de, deweight)
    for i in tqdm(range(buf_size, int(num_samples), loop_iterations)):
        sampler = MHSampler(x0, lnlike, lnprior, mix, iterations=loop_iterations)
        mix.recursive_update(count, chain)
        chain, accept, lnprob = sampler.sample()
        sampler.save_samples(outdir, filename=filename)
        full_chain[count:count + loop_iterations, :] = chain
        mix.recursive_update(count, chain)
        x0 = chain[-1]
        count += loop_iterations
    # save compressed file and delete others
    if save and compress:
        save_h5(outdir + filename, full_chain)
        os.remove(outdir + filename)
    return full_chain


@ray.remote
def ray_sample(lnlike, lnprior, ndim, x0, num_samples=100_000, buf_size=10000,
               amweight=30, scamweight=15, deweight=50,
               loop_iterations=1000, save=True, outdir='./test', filename='/chain_1.txt', compress=False):
    return sample(lnlike, lnprior, ndim, x0, num_samples=num_samples, buf_size=buf_size,
                  amweight=amweight, scamweight=scamweight, deweight=deweight,
                  loop_iterations=loop_iterations, save=save, outdir=outdir, filename=filename, compress=compress)


def parallel_sample(nchains, ncores, lnlike, lnprior, ndim, x0, num_samples=300_000):
    if not ray.is_initialized():
        ray.init(num_cpus=ncores)
    ids = [ray_sample.remote(lnlike, lnprior, ndim, x0[ii], num_samples=num_samples, filename='/chain_1_{}.txt'.format(ii)) for ii in range(nchains)]
    return ray.get(ids)
