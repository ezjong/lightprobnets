from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class Smooth2D(nn.Module):
    def __init__(self, num_input_channels, width=3, std=1.0, padding_type='replicate'):
        super(Smooth2D, self).__init__()

        padding = int(np.floor(width*0.5))
        self.pad = nn.ReplicationPad2d(padding)
        # self.pad = nn.ConstantPad2d(padding)

        filt = np.zeros([num_input_channels, num_input_channels, width, width], np.float32)
        gauss_2d = self.gaussian_smoothing_weights(shape=[width, width], sigma=std)

        for i in range(num_input_channels):
            filt_2d = np.zeros([num_input_channels, width, width], np.float32)
            filt_2d[i, ...] = gauss_2d
            filt[i, ...] = filt_2d

        weight = torch.from_numpy(filt)
        self.register_buffer('weight', weight)

    def gaussian_smoothing_weights(self, shape, sigma):
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp(-(x*x + y*y) / (2.0*sigma*sigma))
        h[h < np.finfo(h.dtype).eps*h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h.astype(np.float32)

    def forward(self, input):
        return F.conv2d(self.pad(input), Variable(self.weight, requires_grad=False))


class ToGrayscale(nn.Module):
    """ Convert to grayscale:  0.299 * R + 0.587 * G + 0.114 * B """

    def __init__(self):
        super(ToGrayscale, self).__init__()
        # Tested these weights, they work as expected
        w = np.array([0.299, 0.587, 0.114], np.float32).reshape([1,3,1,1])
        weight = torch.from_numpy(w)
        self.register_buffer('weight', weight)

    def forward(self, input):
        return F.conv2d(input, Variable(self.weight, requires_grad=False))
