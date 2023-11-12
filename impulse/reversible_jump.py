import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass
class RJState:
    """
    Reversible Jump State
    """
    models: dict
    model_chain_stats: dict
    model_short_chains: dict
    ndim: int
    num_params_increment: int

def reversible_birth(rjstate: RJState):
    """
    Birth move for reversible jump. Adds a new model to the model dictionary.
    """
    # randomly select a model to copy
    model_idx = np.random.randint(0, len(rjstate.models))
    # make a copy of the model
    new_model = rjstate.models[model_idx].copy()
    # add the new model to the dictionary
    rjstate.models[len(rjstate.models)] = new_model
    # increment the number of parameters
    rjstate.ndim += rjstate.num_params_increment

        

    



