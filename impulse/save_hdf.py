import h5py


def save_h5(filepath, data):
    hf = h5py.File(filepath, 'w')
    hf.create_dataset('chain', data=data, compression="gzip", compression_opts=9)
    hf.close()
