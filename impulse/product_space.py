import numpy as np

class ProductSpace:
    def __init__(self,
                 model_names: list,
                 loglikelihoods: list,
                 logpriors: list,
                 param_names: list[list[str]]):
        self.model_names = model_names
        self.loglikelihoods = loglikelihoods
        self.logpriors = logpriors
        self.num_models = len(self.model_names)
        self.nmodels = np.arange(self.num_models)

        # get unique parameters for output
        self.all_params = []
        for jj, param_list in enumerate(param_names):
            for i in range(len(param_list)):
                self.all_params.append(param_list[i] + '_' + model_names[jj])
        self.all_params.append('nmodel')
        self.ndim = len(self.all_params)

        # get indices for each model
        self.model_params = []
        for i in range(self.num_models):
            self.model_params.append([])
            for param in param_names[i]:
                self.model_params[i].append(self.all_params.index(param + '_' + model_names[i]))

    def loglikelihood(self, x):
        # find model index variable
        idx = list(self.all_params).index('nmodel')
        nmodel = int(np.rint(x[idx]))

        # only active parameters enter likelihood
        active_lnlike = self.loglikelihoods[nmodel]
        return active_lnlike(x[self.model_params[nmodel]])

    def logprior(self, x):
        # find model index variable
        idx = list(self.all_params).index('nmodel')
        nmodel = int(np.rint(x[idx]))

        if nmodel not in self.nmodels:
            return -np.inf
        vals = np.array([self.logpriors[i](x[self.model_params[i]]) for i in self.nmodels])
        if np.any(np.isinf(vals)):
            return -np.inf

        active_lnprior = self.logpriors[nmodel]
        return active_lnprior(x[self.model_params[nmodel]])