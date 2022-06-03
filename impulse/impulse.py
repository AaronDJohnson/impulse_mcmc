import numpy as np
from tqdm import tqdm
import os

from impulse.sampler import MHSampler
from impulse.proposals import JumpProposals, am, scam, de
from impulse.save_hdf import save_h5


def sample(lnlike, lnprior, ndim, x0, num_samples=100_000, buf_size=10000,
           loop_iterations=1000, save=True, outdir='./test', compress=True):
    # set up proposals for burn-in:
    mix = JumpProposals(ndim, buf_size=buf_size)
    mix.add_jump(am, 30)
    mix.add_jump(scam, 15)
    # make empty full chain
    full_chain = np.zeros((num_samples, ndim))
    # set up count and iterations between loops
    count = 0
    while count < buf_size:  # fill the buffer
        sampler = MHSampler(x0, lnlike, lnprior, mix, iterations=loop_iterations)
        chain, accept, lnprob = sampler.sample()
        sampler.save_samples(outdir)
        full_chain[count:count + loop_iterations, :] = chain
        mix.recursive_update(count, chain)
        x0 = chain[-1]
        count += loop_iterations
    # add DE jump:
    mix.add_jump(de, 50)
    for i in tqdm(range(buf_size, int(num_samples), loop_iterations)):
        sampler = MHSampler(x0, lnlike, lnprior, mix, iterations=loop_iterations)
        mix.recursive_update(count, chain)
        chain, accept, lnprob = sampler.sample()
        sampler.save_samples(outdir)
        full_chain[count:count + loop_iterations, :] = chain
        mix.recursive_update(count, chain)
        x0 = chain[-1]
        count += loop_iterations
    # save compressed file and delete others
    if save and compress:
        save_h5(outdir + '/chain_1.h5', full_chain)
        os.remove(outdir + '/chain_1.txt')
    return full_chain

