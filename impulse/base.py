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


# TODO: make this a class and combine these two functions with better counting built in
def pt_sample(lnlike, lnprior, ndim, x0, num_samples=1_000_000, buf_size=50_000,
              amweight=30, scamweight=15, deweight=50, ntemps=2, tmin=1, tmax=None, tstep=None,
              swap_count=100, ladder=None, tinf=False, adapt_t0=100, adapt_nu=10,
              loop_iterations=1000, outdir='./chains'):

    ptswap = PTSwap(ndim, ntemps, tmin=tmin, tmax=tmax, tstep=tstep,
                    tinf=tinf, adapt_t0=adapt_t0,
                    adapt_nu=adapt_nu, ladder=ladder)
    saves = [SaveData(outdir=outdir, filename='/chain_{}.txt'.format(ptswap.ladder[ii])) for ii in range(ntemps)]

    # make empty full chain
    full_chain = np.zeros((num_samples, ndim, ntemps))
    chain = np.zeros((loop_iterations, ndim, ntemps))
    lnlike_arr = np.zeros((loop_iterations, ntemps))
    lnprob_arr = np.zeros((loop_iterations, ntemps))
    accept_arr = np.zeros((loop_iterations, ntemps))
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
            swap_idx = samplers[0].num_samples % loop_iterations - 1
            chain, lnlike_arr, logprob_arr = ptswap(chain, lnlike_arr, lnprob_arr, swap_idx)
            ptswap.adapt_ladder(adapt_t0=adapt_t0, adapt_nu=adapt_nu)
            [samplers[ii].set_x0(chain[swap_idx, :, ii], logprob_arr[swap_idx, ii], temp=ptswap.ladder[ii]) for ii in range(ntemps)]
            # print(samplers[0].x0)
            swap_tot += swap_count
        for ii in range(ntemps):
            saves[ii](chain[:, :, ii], lnlike_arr[:, ii], lnprob_arr[:, ii], accept_arr[:, ii])
            mixes[ii].recursive_update(count, chain[:, :, ii])
        full_chain[count:count + loop_iterations, :, :] = chain
        count += loop_iterations
    return full_chain
