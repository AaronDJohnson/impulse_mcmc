import numpy as np
from scipy import stats
import deepcopy

def propose_birth_move(RJState):
    if np.sum(self.configuration) == self.N_possible_knots:
        return (-np.inf, -np.inf, self.configuration, self.current_heights)
    else:
        idx_to_add = np.random.choice(np.where(~self.configuration)[0])
        new_heights = deepcopy(self.current_heights)
        new_config = deepcopy(self.configuration)
        new_config[idx_to_add] = True

    randnum = np.random.rand()
    
    # proposal height
    height_from_model = self.evaluate_interp_model(self.available_knots[idx_to_add],
                                                    self.current_heights, self.configuration)
    if randnum < self.birth_uniform_frac:
        # uniform draw
        new_heights[idx_to_add] = np.random.rand() * (self.yhigh - self.ylow) + self.ylow
    else:
        # gaussian draw around height
        new_heights[idx_to_add] = stats.norm.rvs(loc=height_from_model, scale=self.birth_gauss_scalefac, size=1)
    
    log_qx = 0
    
    log_qy = np.log(self.birth_uniform_frac / self.yrange + \
                    (1 - self.birth_uniform_frac) * stats.norm.pdf(new_heights[idx_to_add], loc=height_from_model,
                                                                scale=self.birth_gauss_scalefac))
    
    log_px = 0

    log_py = self.get_height_log_prior(new_heights[idx_to_add])
    
    new_ll = self.ln_likelihood(new_config, new_heights)



class RJState:
    def __init__(self, max_ndim, ):
        self.max_ndim = max_ndim
