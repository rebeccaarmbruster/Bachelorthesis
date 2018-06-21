import numpy as np

def write_parameters(file, params):
    for k, v in params.items():
        with open(file, 'a') as out:
            print(k, ":", v, file=out)

def get_batches(x, y, mask, rmd, mb_size, shuffle=False, rng_seed=0):
    if shuffle:
        np.random.seed(rng_seed)
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        for i in range(0, len(x) - mb_size + 1, mb_size):
            excerpt = indices[i:i+mb_size]
            yield x[excerpt], y[excerpt], mask[excerpt], rmd[excerpt]
    else:
        n_batches = len(x) // mb_size
        x, y = x[:n_batches * mb_size], y[:n_batches * mb_size]
        for i in range(0, len(x), mb_size):
            yield x[i:i + mb_size], y[i:i + mb_size], mask[i:i + mb_size], rmd[i:i + mb_size]
