from dataclasses import dataclass
import numpy as np
import pathlib
import os
from impulse.sampler_state import SamplerState
from impulse.utils import prepare_files

@dataclass
class ShortChain:
    """
    Ring buffer to hold short chains and save to file periodically.
    """
    ndim: int
    ntemps: int
    short_iters: int
    iteration: int = 0
    outdir: str = './chains/'
    resume: bool = False
    thin: int = 1

    def __post_init__(self):
        if self.thin > self.short_iters:
            raise ValueError("There are not enough samples to thin. Increase save_freq.")
        self.samples = np.zeros((self.ntemps, self.short_iters, self.ndim))
        self.lnprob = np.zeros((self.ntemps, self.short_iters))
        self.lnlike = np.zeros((self.ntemps, self.short_iters))
        self.accept = np.zeros((self.ntemps, self.short_iters))
        self.var_temp = np.zeros((self.ntemps, self.short_iters))
        self.filenames = [f'chain_{nchain}.txt' for nchain in range(self.ntemps)]
        self.filepaths = [os.path.join(self.outdir, filename) for filename in self.filenames]
        prepare_files(self.filepaths, resume=self.resume)

    def add_state(self,
                  new_state: SamplerState):
        save_iter = self.iteration % self.short_iters
        self.samples[:, save_iter] = new_state.positions
        self.lnprob[:, save_iter] = new_state.lnprobs
        self.lnlike[:, save_iter] = new_state.lnlikes
        self.accept[:, save_iter] = new_state.accepted
        self.var_temp[:, save_iter] = new_state.temps
        self.iteration += 1

    def exists(self, outdir, filename):
        return pathlib.Path(os.path.join(outdir, filename)).exists()

    def save_chain(self):
        for temp_idx, filepath in enumerate(self.filepaths):
            to_save = np.column_stack([self.samples[temp_idx], self.lnlike[temp_idx], self.lnprob[temp_idx], self.accept[temp_idx], self.var_temp[temp_idx]])[::self.thin]
            with open(filepath, 'a') as fp:
                np.savetxt(fp, to_save, fmt='%s', delimiter=' ')
