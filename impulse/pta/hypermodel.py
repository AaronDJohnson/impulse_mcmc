import numpy as np
from scipy import linalg as sl
from impulse.pta.pta_utils import get_parameter_groups
import enterprise.constants as const


class HyperModel(object):
    """
    Class to define hyper-model that is the concatenation of all models.
    """

    def __init__(self, models, log_weights=None):
        self.models = models
        self.num_models = len(self.models)
        self.log_weights = log_weights

        #########
        self.param_names, ind = np.unique(np.concatenate([p.param_names
                                                          for p in self.models.values()]),
                                          return_index=True)
        self.param_names = self.param_names[np.argsort(ind)]
        self.param_names = np.append(self.param_names, 'nmodel').tolist()
        #########

        self.pulsars = np.unique(np.concatenate([p.pulsars
                                                 for p in self.models.values()]))
        self.pulsars = np.sort(self.pulsars)

        #########
        self.params = [p for p in self.models[0].params]  # start of param list
        uniq_params = [str(p) for p in self.models[0].params]  # which params are unique
        for model in self.models.values():
            # find differences between next model and concatenation of previous
            param_diffs = np.setdiff1d([str(p) for p in model.params], uniq_params)
            mask = np.array([str(p) in param_diffs for p in model.params])
            # concatenate for next loop iteration
            uniq_params = np.union1d([str(p) for p in model.params], uniq_params)
            # extend list of unique parameters
            self.params.extend([pp for pp in np.array(model.params)[mask]])
        #########

        #########
        # get signal collections
        self.snames = dict.fromkeys(np.unique(sum(sum([[[qq.signal_name for qq in pp._signals]
                                                        for pp in self.models[mm]._signalcollections]
                                                       for mm in self.models], []), [])))
        for key in self.snames:
            self.snames[key] = []

        for mm in self.models:
            for sc in self.models[mm]._signalcollections:
                for signal in sc._signals:
                    self.snames[signal.signal_name].extend(signal.params)
        for key in self.snames:
            self.snames[key] = list(set(self.snames[key]))

        for key in self.snames:
            uniq_params, ind = np.unique([p.name for p in self.snames[key]],
                                         return_index=True)
            uniq_params = uniq_params[np.argsort(ind)].tolist()
            all_params = [p.name for p in self.snames[key]]

            self.snames[key] = np.array(self.snames[key])[[all_params.index(q)
                                                           for q in uniq_params]].tolist()
        #########

    def get_lnlikelihood(self, x):

        # find model index variable
        idx = list(self.param_names).index('nmodel')
        nmodel = int(np.rint(x[idx]))

        # find parameters of active model
        q = []
        for par in self.models[nmodel].param_names:
            idx = self.param_names.index(par)
            q.append(x[idx])

        # only active parameters enter likelihood
        active_lnlike = self.models[nmodel].get_lnlikelihood(q)

        if self.log_weights is not None:
            active_lnlike += self.log_weights[nmodel]

        return active_lnlike

    def get_lnprior(self, x):

        # find model index variable
        idx = list(self.param_names).index('nmodel')
        nmodel = int(np.rint(x[idx]))

        if nmodel not in self.models.keys():
            return -np.inf
        else:
            lnP = 0
            for p in self.models.values():
                q = []
                for par in p.param_names:
                    idx = self.param_names.index(par)
                    q.append(x[idx])
                lnP += p.get_lnprior(np.array(q))

            return lnP

    def get_parameter_groups(self):

        groups = []
        for p in self.models.values():
            groups.extend(get_parameter_groups(p))
        list(np.unique(groups))

        groups.extend([[len(self.param_names)-1]])  # nmodel

        return groups

    def initial_sample(self):
        """
        Draw an initial sample from within the hyper-model prior space.
        """

        x0 = [np.array(p.sample()).ravel().tolist() for p in self.models[0].params]
        uniq_params = [str(p) for p in self.models[0].params]

        for model in self.models.values():
            param_diffs = np.setdiff1d([str(p) for p in model.params], uniq_params)
            mask = np.array([str(p) in param_diffs for p in model.params])
            x0.extend([np.array(pp.sample()).ravel().tolist() for pp in np.array(model.params)[mask]])

            uniq_params = np.union1d([str(p) for p in model.params], uniq_params)

        x0.extend([[0.1]])

        return np.array([p for sublist in x0 for p in sublist])


    def get_process_timeseries(self, psr, chain, burn, comp='DM',
                               mle=False, model=0):
        """
        Construct a time series realization of various constrained processes.

        :param psr: enterprise pulsar object
        :param chain: MCMC chain from sampling all models
        :param burn: desired number of initial samples to discard
        :param comp: which process to reconstruct? (red noise or DM) [default=DM]
        :param mle: create time series from ML of GP hyper-parameters? [default=False]
        :param model: which sub-model within the super-model to reconstruct from? [default=0]

        :return ret: time-series of the reconstructed process
        """

        wave = 0
        pta = self.models[model]
        model_chain = chain[np.rint(chain[:, -5])==model, :]

        # get parameter dictionary
        if mle:
            ind = np.argmax(model_chain[:, -4])
        else:
            ind = np.random.randint(burn, model_chain.shape[0])
        params = {par: model_chain[ind, ct]
                  for ct, par in enumerate(self.param_names)
                  if par in pta.param_names}

        # deterministic signal part
        wave += pta.get_delay(params=params)[0]

        # get linear parameters
        # Nvec = pta.get_ndiag(params)[0] # Not currently used in code
        phiinv = pta.get_phiinv(params, logdet=False)[0]
        T = pta.get_basis(params)[0]

        d = pta.get_TNr(params)[0]
        TNT = pta.get_TNT(params)[0]

        # Red noise piece
        Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

        try:
            u, s, _ = sl.svd(Sigma)
            mn = np.dot(u, np.dot(u.T, d)/s)
            Li = u * np.sqrt(1/s)
        except np.linalg.LinAlgError:

            Q, R = sl.qr(Sigma)
            Sigi = sl.solve(R, Q.T)
            mn = np.dot(Sigi, d)
            u, s, _ = sl.svd(Sigi)
            Li = u * np.sqrt(1/s)

        b = mn + np.dot(Li, np.random.randn(Li.shape[0]))

        # find basis indices
        pardict = {}
        for sc in pta._signalcollections:
            ntot = 0
            for sig in sc._signals:
                if sig.signal_type == 'basis':
                    basis = sig.get_basis(params=params)
                    nb = basis.shape[1]
                    pardict[sig.signal_name] = np.arange(ntot, nb+ntot)
                    ntot += nb

        # DM quadratic + GP
        if comp == 'DM':
            idx = pardict['dm_gp']
            wave += np.dot(T[:, idx], b[idx])
            ret = wave * (psr.freqs**2 * const.DM_K * 1e12)
        elif comp == 'scattering':
            idx = pardict['scattering_gp']
            wave += np.dot(T[:, idx], b[idx])
            ret = wave * (psr.freqs**4)  # * const.DM_K * 1e12)
        elif comp == 'red':
            idx = pardict['red noise']
            wave += np.dot(T[:, idx], b[idx])
            ret = wave
        elif comp == 'FD':
            idx = pardict['FD']
            wave += np.dot(T[:, idx], b[idx])
            ret = wave
        elif comp == 'all':
            wave += np.dot(T, b)
            ret = wave
        else:
            ret = wave

        return ret

    def summary(self, to_stdout=False):
        """generate summary string for HyperModel, including all PTAs

        :param to_stdout: [bool]
            print summary to `stdout` instead of returning it
        :return: [string]

        """

        summary = ""
        for ii, pta in self.models.items():
            summary += "model " + str(ii) + "\n"
            summary += "=" * 9 + "\n\n"
            summary += pta.summary()
            summary += "=" * 90 + "\n\n"
        if to_stdout:
            print(summary)
        else:
            return summary
