import h5py
import pathlib
import numpy as np

class SaveData():
    def __init__(self, outdir='./chains/', filename='chain_1.txt'):
        self.outdir = outdir
        self.filepath = outdir + filename
        pathlib.Path(self.outdir).mkdir(parents=True, exist_ok=True)

    def set_filepath(self, outdir, filename):
        self.filepath = outdir + filename

    def __call__(self, *args):
        with open(self.filepath, 'a+') as f:
            np.savetxt(f, np.column_stack((args)))

    def exists(self, outdir, filename):
        return pathlib.Path(outdir + filename).exists()


def save_h5(filepath, data):
    hf = h5py.File(filepath, 'w')
    hf.create_dataset('chain', data=data, compression="gzip", compression_opts=9)
    hf.close()
