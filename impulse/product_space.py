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
        all_params = []
        for param_list in param_names:
            for i in range(len(param_list)):
                all_params.append(param_list[i])
        all_params.append('nmodel')

        self.unique_params, unique_param_idxs = np.unique(all_params, return_index=True)
        self.unique_params = self.unique_params[np.argsort(unique_param_idxs)]
        self.ndim = len(self.unique_params)

        # get correspondence between unique parameters and model parameters
        self.model_params = []  # should be as long as number of models
        for param_list in param_names:
            temp_param_idxs = []
            for i in range(len(param_list)):
                temp_param_idxs.append(list(self.unique_params).index(param_list[i]))
            self.model_params.append(np.array(temp_param_idxs))
        
        self.model_ndims = [len(self.model_params[i]) for i in range(self.num_models)]

    def loglikelihood(self, x):
        # find model index variable
        idx = list(self.unique_params).index('nmodel')
        nmodel = int(np.rint(x[idx]))

        # only active parameters enter likelihood
        active_lnlike = self.loglikelihoods[nmodel]
        return active_lnlike(x[self.model_params[nmodel]])

    def logprior(self, x):
        # find model index variable
        idx = list(self.unique_params).index('nmodel')
        nmodel = int(np.rint(x[idx]))

        if nmodel not in self.nmodels:
            return -np.inf

        active_lnprior = self.logpriors[nmodel]
        return active_lnprior(x[self.model_params[nmodel]])
