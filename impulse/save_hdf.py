import numpy as np
import h5py
import os


def save_h5(filepath, data):
    hf = h5py.File(filepath, 'w')
    hf.create_dataset('chain', data=data, compression="gzip", compression_opts=9)
    hf.close()
