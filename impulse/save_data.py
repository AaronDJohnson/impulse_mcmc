import h5py
import pathlib
import numpy as np

class SaveData():
    def __init__(self, outdir='./chains/', filename='chain_1.txt',
                 tempdir='/temp_data/', resume=False, thin=1):
        self.thin = thin
        self.outdir = outdir
        self.tempdir = tempdir
        self.filepath = outdir + filename
        self.swap_accept_file = outdir + tempdir + 'accept.txt'
        self.ladder_file = outdir + tempdir + 'temps.txt'
        pathlib.Path(self.outdir).mkdir(parents=True, exist_ok=True)
        # pathlib.Path(self.outdir + self.tempdir).mkdir(parents=True, exist_ok=True)
        if self.exists(outdir, filename) and not resume:
            with open(self.filepath, 'w') as _:
                pass
        if self.exists(outdir + tempdir, 'accept.txt') and not resume:
            with open(self.swap_accept_file, 'w') as _:
                pass
        if self.exists(outdir + tempdir, 'temps.txt') and not resume:
            with open(self.ladder_file, 'w') as _:
                pass


    def set_filepath(self, outdir, filename):
        self.filepath = outdir + filename


    def __call__(self, *args):
        with open(self.filepath, 'a+') as f:
            np.savetxt(f, np.column_stack((args))[::self.thin])


    def exists(self, outdir, filename):
        return pathlib.Path(outdir + filename).exists()


    def save_swap_data(self, ptswap):
        """
        :param ptswap: PTSwap object
        """
        ladder = ptswap.ladder
        swap_accept = ptswap.compute_accept_ratio()
        with open(self.swap_accept_file, 'a+') as f:
            np.savetxt(f, swap_accept, newline=" ")
            f.write('\n')
        with open(self.ladder_file, 'a+') as f:
            np.savetxt(f, ladder, newline=" ")
            f.write('\n')


def save_h5(filepath, data):
    hf = h5py.File(filepath, 'w')
    hf.create_dataset('chain', data=data, compression="gzip", compression_opts=9)
    hf.close()
