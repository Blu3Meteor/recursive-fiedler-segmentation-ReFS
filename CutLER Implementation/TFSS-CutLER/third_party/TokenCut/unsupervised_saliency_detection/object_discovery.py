import numpy as np
from scipy import ndimage


def detect_box(bipartition, seed, dims, scales=None, initial_im_size=None):
    mask = np.asarray(bipartition).reshape(dims).astype(bool)
    seed_y = int(seed) // int(dims[1])
    seed_x = int(seed) % int(dims[1])
    labels, _ = ndimage.label(mask)
    label = labels[seed_y, seed_x]
    if label == 0:
        ys, xs = np.where(mask)
    else:
        ys, xs = np.where(labels == label)
    if len(xs) == 0:
        return 0, 0, 0, (np.array([], dtype=int), np.array([], dtype=int))
    return int(xs.min()), int(ys.min()), int(xs.max()), (ys, xs)

