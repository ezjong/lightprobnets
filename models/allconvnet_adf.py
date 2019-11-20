from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import logging
import torch
from contrib import adf


def keep_variance(x, min_variance):
    return x + min_variance


def finitialize_msra(modules, small=False):
    logging.info("Initializing MSRA")
    for layer in modules:
        if isinstance(layer, adf.Conv2d):  # convolution: bias=0, weight=msra
            nn.init.kaiming_normal_(layer.weight)
            if small:
                layer.weight.data.mul_(0.001)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


def finitialize_xavier(modules, small=False):
    logging.info("Initializing Xavier")
    for layer in modules:
        if isinstance(layer, adf.Conv2d):  # convolution: bias=0, weight=msra
            nn.init.xavier_normal_(layer.weight)
            if small:
                layer.weight.data.mul_(0.001)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


def make_conv(inchannels, outchannels, kernel_size, stride, nonlinear=True, keep_variance_fn=None):
    padding = kernel_size // 2
    if nonlinear:
        return adf.Sequential(
            adf.Conv2d(
                inchannels, outchannels, kernel_size=kernel_size, padding=padding,
                stride=stride, bias=True, keep_variance_fn=keep_variance_fn),
            adf.ReLU(keep_variance_fn=keep_variance_fn)
        )
    else:
        return adf.Conv2d(
            inchannels, outchannels, kernel_size=kernel_size, padding=padding, stride=stride, bias=True,
            keep_variance_fn=keep_variance_fn)


class AllConvNetADF(nn.Module):
    def __init__(self, args, initialize_msra=False, noise_variance=1e-3, min_variance=1e-3, log_variance=False):
        super(AllConvNetADF, self).__init__()
        self._keep_variance_fn = lambda x: keep_variance(x, min_variance=min_variance)
        self._noise_variance = noise_variance

        self.conv1   = make_conv(3, 96, kernel_size=3, stride=1, keep_variance_fn=self._keep_variance_fn)
        self.conv1_1 = make_conv(96, 96, kernel_size=3, stride=1, keep_variance_fn=self._keep_variance_fn)
        self.conv1_2 = make_conv(96, 96, kernel_size=3, stride=2, nonlinear=False, keep_variance_fn=self._keep_variance_fn)
        self.dropout1 = adf.Dropout(0.5, keep_variance_fn=self._keep_variance_fn)

        self.conv2   = make_conv( 96, 192, kernel_size=3, stride=1, keep_variance_fn=self._keep_variance_fn)
        self.conv2_1 = make_conv(192, 192, kernel_size=3, stride=1, keep_variance_fn=self._keep_variance_fn)
        self.conv2_2 = make_conv(192, 192, kernel_size=3, stride=2, nonlinear=False, keep_variance_fn=self._keep_variance_fn)
        self.dropout2 = adf.Dropout(0.5, keep_variance_fn=self._keep_variance_fn)

        self.conv3   = make_conv(192, 192, kernel_size=3, stride=1, keep_variance_fn=self._keep_variance_fn)
        self.conv3_1 = make_conv(192, 192, kernel_size=1, stride=1, keep_variance_fn=self._keep_variance_fn)
        self.conv3_2 = make_conv(192,  10, kernel_size=1, stride=1, nonlinear=False, keep_variance_fn=self._keep_variance_fn)

        if initialize_msra:
            finitialize_msra(self.modules(), small=False)
        else:
            finitialize_xavier(self.modules(), small=False)

    def forward(self, example_dict):
        inputs = example_dict["input1"]
        inputs_mean = inputs
        inputs_variance = torch.zeros_like(inputs_mean) + self._noise_variance
        x = inputs_mean, inputs_variance

        x = self.conv1(*x)
        x = self.conv1_1(*x)
        x = self.conv1_2(*x)
        x = self.dropout1(*x)

        x = self.conv2(*x)
        x = self.conv2_1(*x)
        x = self.conv2_2(*x)
        x = self.dropout2(*x)

        x = self.conv3(*x)
        x = self.conv3_1(*x)
        x = self.conv3_2(*x)
        mu, variance = x

        batch_size = mu.size(0)

        mu = mu.contiguous().view(batch_size, 10, -1)
        variance = variance.contiguous().view(batch_size, 10, -1)

        N = mu.size(2)
        mu = mu.mean(dim=2)
        variance = variance.mean(dim=2) * (1.0 / float(N))

        return {"mean1": mu, "variance1": variance }
