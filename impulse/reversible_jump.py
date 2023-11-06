import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass
class RJState:
    """
    Reversible Jump State
    """
    ndim: int
    current_loglike: Callable

    def update_state(self, ndim, new_loglike):
        self.ndim = ndim
        self.current_loglike = new_loglike

