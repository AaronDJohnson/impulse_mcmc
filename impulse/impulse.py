import numpy as np
# from schwimmbad import MultiPool
from tqdm import tqdm
import os

from impulse.sampler import MHSampler
from impulse.proposals import pre_proposals, stock_proposals
from impulse.save_hdf import save_h5


def sample(post, ndim, x0, num_samples=100_000, loop_iterations=1000, save=True, outdir='./test', compress=True):
    # set up proposals for burn-in:
    mix = pre_proposals(ndim)
    # make empty full chain
    full_chain = np.zeros((num_samples, ndim))
    # set up count and iterations between loops
    count = 0
    while count < int(num_samples / 100):
        sampler = MHSampler(x0, post, mix, iterations=loop_iterations)
        chain, accept, lnprob = sampler.sample()
        sampler.save_samples(outdir)
        full_chain[count:count + loop_iterations, :] = chain
        mix.update(loop_iterations, chain)
        x0 = chain[-1]
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
    # save compressed file and delete others
    if compress:
        save_h5(outdir + '/chain_1.h5', full_chain)
        os.remove(outdir + '/chain_1.txt')
    return full_chain

