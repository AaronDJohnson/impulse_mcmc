import numpy as np
from tqdm import tqdm
import os
import ray
from impulse.sampler import MHSampler
from impulse.ptsampler import PTSampler, temp_ladder, propose_swaps
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


def pt_sample(lnlike, lnprior, ndim, x0, num_samples=100_000, buf_size=10000,
              amweight=30, scamweight=15, deweight=50, ntemps=2, tmin=1, tmax=None, tstep=None,
              swap_count=100,
              loop_iterations=1000, save=True, outdir='./test', filename='/chain_1.txt', compress=True):
    ladder = temp_ladder(tmin, ndim, ntemps, tmax=tmax, tstep=tstep)
    # make empty full chain
    full_chain = np.zeros((num_samples, ndim, ntemps))

    chain = np.zeros((loop_iterations, ndim, ntemps))
    lnlike_arr = np.zeros((loop_iterations, ntemps))
    # set up proposals for burn-in:
    mixes = []
    for ii in range(len(ladder)):
        mixes.append(JumpProposals(ndim, buf_size=buf_size))
        mixes[ii].add_jump(am, amweight)
        mixes[ii].add_jump(scam, scamweight)
        full_chain[0, :, ii] = x0[ii]
        print(full_chain[0, :, ii])
    
    # set up count and iterations between loops
    count = 0
    while count < buf_size:  # fill the buffer
        swap_tot = 0

        while swap_tot == 0 or loop_iterations / swap_tot != 1:
            print(full_chain[count + swap_tot, :, ii])
            samplers = [PTSampler(full_chain[count + swap_tot, :, ii], lnlike, lnprior, mixes[ii], ladder[ii], iterations=swap_count) for ii in range(len(ladder))]
            for ii, sampler in enumerate(samplers):
                chain[swap_tot:swap_tot + swap_count, :, ii], lnlike_arr[swap_tot:swap_tot + swap_count, ii] = sampler.sample()
            chain = propose_swaps(chain, lnlike_arr, ladder, swap_tot)
            swap_tot += swap_count
        full_chain[count:count + loop_iterations, :, :] = chain
        for ii in range(len(ladder)):
            samplers[ii].save_samples(outdir, filename='/chain_{0}.txt'.format(ladder[ii]))
            mixes[ii].recursive_update(count, chain[:, :, ii])
            x0[ii] = chain[-1, :, ii]
        count += loop_iterations
    return full_chain
    # add DE jump:
    # mix.add_jump(de, deweight)
    # for i in tqdm(range(buf_size, int(num_samples), loop_iterations)):
    #     sampler = MHSampler(x0, lnlike, lnprior, mix, iterations=loop_iterations)
    #     mix.recursive_update(count, chain)
    #     chain, accept, lnprob = sampler.sample()
    #     sampler.save_samples(outdir, filename=filename)
    #     full_chain[count:count + loop_iterations, :] = chain
    #     mix.recursive_update(count, chain)
    #     x0 = chain[-1]
    #     count += loop_iterations
    # # save compressed file and delete others
    # if save and compress:
    #     save_h5(outdir + filename, full_chain)
    #     os.remove(outdir + filename)
    # return full_chain






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
