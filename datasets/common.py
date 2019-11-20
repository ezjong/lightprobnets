from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import torch
import numpy as np
from scipy import ndimage


def numpy2torch(array):
    assert(isinstance(array, np.ndarray))
    if array.ndim == 3:
        array = np.transpose(array, (2, 0, 1))
    else:
        array = np.expand_dims(array, axis=0)
    return torch.from_numpy(array).float()


def deterministic_indices(k, n, seed):
    indices = range(n)
    random.Random(seed).shuffle(indices)
    return sorted(indices[0:k])


def read_flo_as_float32(filename):
    with open(filename, 'rb') as file:
        magic = np.fromfile(file, np.float32, count=1)
        assert(202021.25 == magic), "Magic number incorrect. Invalid .flo file"
        w = np.fromfile(file, np.int32, count=1)[0]
        h = np.fromfile(file, np.int32, count=1)[0]
        data = np.fromfile(file, np.float32, count=2*h*w)
    data2D = np.resize(data, (h, w, 2))
    return data2D


def read_image_as_float32(filename):
    return ndimage.imread(filename).astype(np.float32) / np.float32(255.0)


def read_image_as_byte(filename):
    return ndimage.imread(filename)
