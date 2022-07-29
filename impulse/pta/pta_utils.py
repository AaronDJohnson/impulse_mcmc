import os
import platform

import numpy as np
from loguru import logger

from enterprise_extensions import __version__
from enterprise.signals.parameter import sample as sample_params

from impulse.pta.pta_jumps import ExtraProposals
from impulse.base import PTSampler

# PTA specific functions

def initial_sample(pta, ntemps=None, nchains=None):
    if ntemps is not None and nchains is not None:  # both pt and parallel
        x0 = [[np.array(list(sample_params(pta.params).values())) for _ in range(ntemps)] for _ in range(nchains)]
        ndim = len(x0[0][0])
    elif ntemps is None and nchains is not None:  # no ptsampling
        x0 = [np.array(list(sample_params(pta.params).values())) for _ in range(nchains)]
        ndim = len(x0[0])
    elif ntemps is not None and nchains is None:  # no parallel
        x0 = [np.array(list(sample_params(pta.params).values())) for _ in range(ntemps)]
        ndim = len(x0[0])
    else:  # normal sampling without pt or parallel
        x0 = np.array(list(sample_params(pta.params).values()))
        ndim = len(x0)
    return ndim, x0

# PTA specific jumps (and their utilities)

def get_global_parameters(pta):
    """Utility function for finding global parameters."""
    pars = []
    for sc in pta._signalcollections:
        pars.extend(sc.param_names)

    gpars = list(set(par for par in pars if pars.count(par) > 1))
    ipars = [par for par in pars if par not in gpars]

    # gpars = np.unique(list(filter(lambda x: pars.count(x)>1, pars)))
    # ipars = np.array([p for p in pars if p not in gpars])

    return np.array(gpars), np.array(ipars)


def get_parameter_groups(pta):
    """Utility function to get parameter groupings for sampling."""
    params = pta.param_names
    ndim = len(params)
    groups = [list(np.arange(0, ndim))]

    # get global and individual parameters
    gpars, ipars = get_global_parameters(pta)
    if gpars.size:
        # add a group of all global parameters
        groups.append([params.index(gp) for gp in gpars])

    # make a group for each signal, with all non-global parameters
    for sc in pta._signalcollections:
        for signal in sc._signals:
            ind = [params.index(p) for p in signal.param_names if not gpars.size or p not in gpars]
            if ind:
                groups.append(ind)

    return groups


def get_psr_groups(pta):
    groups = []
    for psr in pta.pulsars:
        grp = [pta.param_names.index(par)
               for par in pta.param_names if psr in par]
        groups.append(grp)
    return groups


def get_cw_groups(pta):
    """Utility function to get parameter groups for CW sampling.
    These groups should be appended to the usual get_parameter_groups()
    output.
    """
    ang_pars = ['costheta', 'phi', 'cosinc', 'phase0', 'psi']
    mfdh_pars = ['log10_Mc', 'log10_fgw', 'log10_dL', 'log10_h']
    freq_pars = ['log10_Mc', 'log10_fgw', 'pdist', 'pphase']

    groups = []
    for pars in [ang_pars, mfdh_pars, freq_pars]:
        groups.append(group_from_params(pta, pars))

    return groups


def group_from_params(pta, params):
    gr = []
    for p in params:
        for q in pta.param_names:
            if p in q:
                gr.append(pta.param_names.index(q))
    return gr


def save_runtime_info(pta, outdir='chains', human=None):
    """save system info, enterprise PTA.summary, and other metadata to file
    """
    # save system info and enterprise PTA.summary to single file
    sysinfo = {}
    if human is not None:
        sysinfo.update({"human": human})
    sysinfo.update(platform.uname()._asdict())

    with open(os.path.join(outdir, "runtime_info.txt"), "w") as fout:
        for field, data in sysinfo.items():
            fout.write(field + " : " + data + "\n")
        fout.write("\n")
        fout.write("enterprise_extensions v" + __version__ +"\n")
        fout.write(pta.summary())

    # save paramter list
    with open(os.path.join(outdir, "pars.txt"), "w") as fout:
        for pname in pta.param_names:
            fout.write(pname + "\n")

    # save list of priors
    with open(os.path.join(outdir, "priors.txt"), "w") as fout:
        for pp in pta.params:
            fout.write(pp.__repr__() + "\n")


def setup_sampler(pta, outdir='chains', empirical_distr=None, hypermodel=False,
                  groups=None, human=None, save_ext_dists=False, ntemps=1,
                  ncores=1, num_samples=1_000_000, ret_chain=False,
                  prior_sample=False, resume=False, sample_nmodel=True):
    """
    Sets up an instance of PTMCMC sampler.

    We initialize the sampler the likelihood and prior function
    from the PTA object. We set up an initial jump covariance matrix
    with fairly small jumps as this will be adapted as the MCMC runs.

    We will setup an output directory in `outdir` that will contain
    the chain (first n columns are the samples for the n parameters
    and last 4 are log-posterior, log-likelihood, acceptance rate, and
    an indicator variable for parallel tempering but it doesn't matter
    because we aren't using parallel tempering).

    We then add several custom jump proposals to the mix based on
    whether or not certain parameters are in the model. These are
    all either draws from the prior distribution of parameters or
    draws from uniform distributions.

    save_ext_dists: saves distributions that have been extended to
    cover priors as a pickle to the outdir folder. These can then
    be loaded later as distributions to save a minute at the start
    of the run.
    """

    # dimension of parameter space
    params = pta.param_names
    ndim = len(params)

    # initial jump covariance matrix
    if os.path.exists(outdir+'/cov.npy'):
        cov = np.load(outdir+'/cov.npy')
    else:
        cov = np.diag(np.ones(ndim))

    if hypermodel:
        x0 = [pta.initial_sample() for __ in range(ntemps)]
        logger.debug(x0)
    else:
        ndim, x0 = initial_sample(pta, ntemps)
    if not len(x0) == ntemps:
        x0 = x0[0]

    # parameter groupings
    if groups is None and not hypermodel:
        groups = get_parameter_groups(pta)
    elif groups is None and hypermodel:
        groups = pta.get_parameter_groups()

    if prior_sample:
        sampler = PTSampler(pta.get_lnprior, lambda x:0, x0, num_samples=num_samples, groups=groups,
                            ntemps=ntemps, ncores=ncores, ret_chain=ret_chain, resume=resume, cov=cov)
    else:
        sampler = PTSampler(pta.get_lnlikelihood, pta.get_lnprior, x0, num_samples=num_samples, groups=groups,
                            ntemps=ntemps, ncores=ncores, ret_chain=ret_chain, resume=resume, cov=cov)

    save_runtime_info(pta, sampler.outdir, human)

    # additional jump proposals
    if hypermodel:
        jp = ExtraProposals(pta, snames=pta.snames, empirical_distr=empirical_distr, save_ext_dists=save_ext_dists, outdir=outdir)
    else:
        jp = ExtraProposals(pta, empirical_distr=empirical_distr, save_ext_dists=save_ext_dists, outdir=outdir)
            

    # always add draw from prior
    sampler.add_jump(jp.draw_from_prior, 5)

    if sample_nmodel and hypermodel:
        print('Adding nmodel uniform distribution draws...\n')
        sampler.add_jump(jp.draw_from_nmodel_prior, 25)

    # try adding empirical proposals
    if empirical_distr is not None:
        logger.info('Attempting to add empirical proposals...\n')
        sampler.add_jump(jp.draw_from_empirical_distr, 10)

    # Red noise prior draw
    if 'red noise' in jp.snames:
        logger.info('Adding red noise prior draws...\n')
        sampler.add_jump(jp.draw_from_red_prior, 10)

    # DM GP noise prior draw
    if 'dm_gp' in jp.snames:
        logger.info('Adding DM GP noise prior draws...\n')
        sampler.add_jump(jp.draw_from_dm_gp_prior, 10)

    # DM annual prior draw
    if 'dm_s1yr' in jp.snames:
        logger.info('Adding DM annual prior draws...\n')
        sampler.add_jump(jp.draw_from_dm1yr_prior, 10)

    # DM dip prior draw
    if 'dmexp' in jp.snames:
        logger.info('Adding DM exponential dip prior draws...\n')
        sampler.add_jump(jp.draw_from_dmexpdip_prior, 10)

    # DM cusp prior draw
    if 'dm_cusp' in jp.snames:
        logger.info('Adding DM exponential cusp prior draws...\n')
        sampler.add_jump(jp.draw_from_dmexpcusp_prior, 10)

    # DMX prior draw
    if 'dmx_signal' in jp.snames:
        logger.info('Adding DMX prior draws...\n')
        sampler.add_jump(jp.draw_from_dmx_prior, 10)

    # Ephemeris prior draw
    if 'd_jupiter_mass' in pta.param_names:
        logger.info('Adding ephemeris model prior draws...\n')
        sampler.add_jump(jp.draw_from_ephem_prior, 10)

    # GWB uniform distribution draw
    if np.any([('gw' in par and 'log10_A' in par) for par in pta.param_names]):
        logger.info('Adding GWB uniform distribution draws...\n')
        sampler.add_jump(jp.draw_from_gwb_log_uniform_distribution, 10)

    # Dipole uniform distribution draw
    if 'dipole_log10_A' in pta.param_names:
        logger.info('Adding dipole uniform distribution draws...\n')
        sampler.add_jump(jp.draw_from_dipole_log_uniform_distribution, 10)

    # Monopole uniform distribution draw
    if 'monopole_log10_A' in pta.param_names:
        logger.info('Adding monopole uniform distribution draws...\n')
        sampler.add_jump(jp.draw_from_monopole_log_uniform_distribution, 10)

    # Altpol uniform distribution draw
    if 'log10Apol_tt' in pta.param_names:
        logger.info('Adding alternative GW-polarization uniform distribution draws...\n')
        sampler.add_jump(jp.draw_from_altpol_log_uniform_distribution, 10)

    # BWM prior draw
    if 'bwm_log10_A' in pta.param_names:
        logger.info('Adding BWM prior draws...\n')
        sampler.add_jump(jp.draw_from_bwm_prior, 10)

    # FDM prior draw
    if 'fdm_log10_A' in pta.param_names:
        logger.info('Adding FDM prior draws...\n')
        sampler.add_jump(jp.draw_from_fdm_prior, 10)

    # CW prior draw
    if 'cw_log10_h' in pta.param_names:
        logger.info('Adding CW strain prior draws...\n')
        sampler.add_jump(jp.draw_from_cw_log_uniform_distribution, 10)
    if 'cw_log10_Mc' in pta.param_names:
        logger.info('Adding CW prior draws...\n')
        sampler.add_jump(jp.draw_from_cw_distribution, 10)

    return sampler


