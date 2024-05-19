from dataclasses import dataclass
import os
import pathlib
import numpy as np

@dataclass
class ShortChain:
    """
    class to hold a short chain of save_freq iterations
    """
    ndim: int
    short_iters: int
    iteration: int = 1
    outdir: str = './chains/'
    resume: bool = False
    thin: int = 1
    nchain: float = 1

    def __post_init__(self):
        if self.thin > self.short_iters:
            raise ValueError("There are not enough samples to thin. Increase save_freq.")
        self.samples = np.zeros((self.short_iters, self.ndim))
        self.lnprob = np.zeros((self.short_iters))
        self.lnlike = np.zeros((self.short_iters))
        self.accept = np.zeros((self.short_iters))
        self.var_temp = np.zeros((self.short_iters))
        self.filename = f'chain_{self.nchain}.txt'
        self.filepath = os.path.join(self.outdir, self.filename)

        pathlib.Path(self.outdir).mkdir(parents=True, exist_ok=True)
        if self.exists(self.outdir, self.filename) and not self.resume:
            with open(self.filepath, 'w') as _:
                pass

    def add_state(self,
                  new_state: MHState):
        self.samples[self.iteration % self.short_iters] = new_state.position
        self.lnprob[self.iteration % self.short_iters] = new_state.lnprob
        self.lnlike[self.iteration % self.short_iters] = new_state.lnlike
        self.accept[self.iteration % self.short_iters] = new_state.accepted
        self.var_temp[self.iteration % self.short_iters] = new_state.temp
        self.iteration += 1

    def set_filepath(self, outdir, filename):
        self.filepath = os.path.join(outdir, filename)

    def exists(self, outdir, filename):
        return pathlib.Path(os.path.join(outdir, filename)).exists()

    def save_chain(self):
        to_save = np.column_stack([self.samples, self.lnlike, self.lnprob, self.accept, self.var_temp])[::self.thin]
        with open(self.filepath, 'a+') as f:
            np.savetxt(f, to_save)