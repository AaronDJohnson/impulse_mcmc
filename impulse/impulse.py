import numpy as np
from tqdm import tqdm
from impulse.mhsampler import MHSampler
from impulse.ptsampler import PTSwap
from impulse.proposals import JumpProposals, am, scam, de
from impulse.save_data import SaveData


def sample(lnlike, lnprior, ndim, x0, num_samples=1_000_000, buf_size=50_000,
           amweight=30, scamweight=15, deweight=50, loop_iterations=1000,
           save=True, outdir='./chains', filename='/chain_1.txt'):
    save = SaveData(outdir=outdir, filename=filename)
    # set up proposals:
    mix = JumpProposals(ndim, buf_size=buf_size)
    mix.add_jump(am, amweight)
    mix.add_jump(scam, scamweight)
    mix.add_jump(de, deweight)
    # make empty full chain
    full_chain = np.zeros((num_samples, ndim))
    # set up count and iterations between loops
    sampler = MHSampler(x0, lnlike, lnprior, mix, iterations=loop_iterations)
    count = 0
    for _ in tqdm(range(0, int(num_samples), loop_iterations)):
        chain, like, prob, accept = sampler.sample()
        save(chain, like, prob, accept)
        full_chain[count:count + loop_iterations, :] = chain
        mix.recursive_update(count, chain)
        count += loop_iterations
    return full_chain


def pt_sample(lnlike, lnprior, ndim, x0, num_samples=1_000_000, buf_size=50_000,
              amweight=30, scamweight=15, deweight=50, ntemps=2, tmin=1, tmax=None, tstep=None,
              swap_count=200, ladder=None, tinf=False, adapt_time=100, adapt_lag=1000,
              loop_iterations=1000, outdir='./chains'):

    ptswap = PTSwap(ndim, ntemps, tmin=tmin, tmax=tmax, tstep=tstep,
                    tinf=tinf, adaptation_time=adapt_time,
                    adaptation_lag=adapt_lag, ladder=ladder)
    saves = [SaveData(outdir=outdir, filename='/chain_{}.txt'.format(ptswap.ladder[ii])) for ii in range(ntemps)]

    # make empty full chain
    full_chain = np.zeros((num_samples, ndim, ntemps))
    chain = np.zeros((loop_iterations, ndim, ntemps))
    lnlike_arr = np.zeros((loop_iterations, ntemps))
    lnprob_arr = np.zeros((loop_iterations, ntemps))
    accept_arr = np.zeros((loop_iterations, ntemps))
    temps_arr = np.zeros((loop_iterations, ntemps))
    # set up proposals
    mixes = []
    for ii in range(ntemps):
        mixes.append(JumpProposals(ndim, buf_size=buf_size))
        mixes[ii].add_jump(am, amweight)
        mixes[ii].add_jump(scam, scamweight)
        mixes[ii].add_jump(de, deweight)

    # make set of samplers (1 for each temp)
    samplers = [MHSampler(x0[ii], lnlike, lnprior, mixes[ii], iterations=swap_count, init_temp=ptswap.ladder[ii]) for ii in range(ntemps)]

    # set up count and iterations between loops
    count = 0
    for _ in tqdm(range(0, int(num_samples), loop_iterations)):
        swap_tot = 0
        for _ in range(loop_iterations // swap_count):
            for ii, sampler in enumerate(samplers):
                (chain[swap_tot:swap_tot + swap_count, :, ii],
                 lnlike_arr[swap_tot:swap_tot + swap_count, ii],
                 lnprob_arr[swap_tot:swap_tot + swap_count, ii],
                 accept_arr[swap_tot:swap_tot + swap_count, ii]) = sampler.sample()
            chain, lnlike_arr, logprob_arr = ptswap(chain, lnlike_arr, lnprob_arr, swap_tot)
            # ptswap.adapt_ladder(samplers[0].num_samples, adaptation_lag=adapt_lag,
            #                     adaptation_time=adapt_time)
            [samplers[ii].set_x0(chain[swap_tot, :, ii], logprob_arr[swap_tot, ii], temp=ptswap.ladder[ii]) for ii in range(ntemps)]
            swap_tot += swap_count
        for ii in range(ntemps):
            saves[ii](chain[:, :, ii], lnlike_arr[:, ii], lnprob_arr[:, ii], accept_arr[:, ii], temps_arr[:, ii])
            mixes[ii].recursive_update(count, chain[:, :, ii])
        full_chain[count:count + loop_iterations, :, :] = chain
        count += loop_iterations
    return full_chain


# @ray.remote
# def ray_sample(lnlike, lnprior, ndim, x0, num_samples=100_000, buf_size=20000,
#                amweight=30, scamweight=15, deweight=50,
#                loop_iterations=1000, save=True, outdir='./test', filename='/chain_1.txt', compress=False):
#     return sample(lnlike, lnprior, ndim, x0, num_samples=num_samples, buf_size=buf_size,
#                   amweight=amweight, scamweight=scamweight, deweight=deweight,
#                   loop_iterations=loop_iterations, save=save, outdir=outdir, filename=filename, compress=compress)

# @ray.remote  # TODO reduce reused code (clean this one up)
# def ray_pt_sample(lnlike, lnprior, ndim, x0, num_samples=100_000, buf_size=20000,
#                      amweight=30, scamweight=15, deweight=50, ntemps=2, tmin=1, tmax=None, tstep=None,
#                      swap_count=100, ladder=None, core_num=0, inf_temp=False,
#                      loop_iterations=1000, outdir='./test'):

#     # make empty full chain
#     full_chain = np.zeros((num_samples, ndim, ntemps))
#     chain = np.zeros((loop_iterations, ndim, ntemps))
#     lnlike_arr = np.zeros((loop_iterations, ntemps))
#     # set up proposals
#     mixes = []
#     for ii in range(len(ladder)):
#         mixes.append(JumpProposals(ndim, buf_size=buf_size))
#         mixes[ii].add_jump(am, amweight)
#         mixes[ii].add_jump(scam, scamweight)
#         mixes[ii].add_jump(de, deweight)

#     # make set of samplers (1 for each temp)
#     samplers = [PTSampler(ndim, lnlike, lnprior, mixes[ii], ladder[ii], iterations=swap_count) for ii in range(len(ladder))]

#     # set up count and iterations between loops
#     count = 0
#     for _ in tqdm(range(0, int(num_samples), loop_iterations)):
#         swap_tot = 0
#         for _ in range(loop_iterations // swap_count):
#             for ii, sampler in enumerate(samplers):
#                 chain[swap_tot:swap_tot + swap_count, :, ii], lnlike_arr[swap_tot:swap_tot + swap_count, ii], x0[ii] = sampler.sample(x0[ii])
#             chain = propose_swaps(chain, lnlike_arr, ladder, swap_tot)
#             swap_tot += swap_count
#         for ii in range(len(ladder)):
#             samplers[ii].save_samples(outdir, filename='/chain_{0}_{1}.txt'.format(ladder[ii], core_num))
#             mixes[ii].recursive_update(count, chain[:, :, ii])
#         full_chain[count:count + loop_iterations, :, :] = chain
#         count += loop_iterations
#     return full_chain


# def parallel_sample(nchains, ncores, lnlike, lnprior, ndim, x0, num_samples=300_000):
#     if not ray.is_initialized():
#         ray.init(num_cpus=ncores)
#     ids = [ray_sample.remote(lnlike, lnprior, ndim, x0[ii], num_samples=num_samples, filename='/chain_1_{}.txt'.format(ii)) for ii in range(nchains)]
#     return ray.get(ids)


# def parallel_pt_sample(nchains, ncores, ntemps, lnlike, lnprior, ndim, x0, num_samples=300_000, tmin=1, tmax=None, tstep=None, inf_temp=False):
#     if not ray.is_initialized():
#         ray.init(num_cpus=ncores)
#     ids = [ray_pt_sample.remote(lnlike, lnprior, ndim, x0[ii], num_samples=num_samples, inf_temp=inf_temp,
#                                 ntemps=ntemps, tmin=tmin, tmax=tmax, tstep=tstep, outdir='./test',
#                                 core_num=ii) for ii in range(nchains)]
#     return ray.get(ids)
