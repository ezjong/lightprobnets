from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import logging
from contrib import adf
import torch


def keep_variance(x, min_variance):
    return x + min_variance


def finitialize_msra(modules, small=False):
    logging.info("Initializing MSRA")
    for layer in modules:
        if isinstance(layer, adf.Conv2d) or isinstance(layer, adf.Linear):  # convolution: bias=0, weight=msra
            nn.init.kaiming_normal_(layer.weight)
            if small:
                layer.weight.data.mul_(0.001)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


def finitialize_xavier(modules, small=False):
    logging.info("Initializing Xavier")
    for layer in modules:
        if isinstance(layer, adf.Conv2d) or isinstance(layer, adf.Linear):  # convolution: bias=0, weight=msra
            nn.init.xavier_normal_(layer.weight)
            if small:
                layer.weight.data.mul_(0.001)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


class LeNetADF(nn.Module):
    def __init__(self, args, noise_variance=1e-3, min_variance=1e-3, initialize_msra=False):
        super(LeNetADF, self).__init__()
        self._keep_variance_fn = lambda x: keep_variance(x, min_variance=min_variance)
        self._noise_variance = noise_variance
        self.conv1 = adf.Conv2d(1, 32, kernel_size=5, keep_variance_fn=self._keep_variance_fn)
        self.relu1 = adf.ReLU(keep_variance_fn=self._keep_variance_fn)
        self.maxpool1 = adf.MaxPool2d(keep_variance_fn=self._keep_variance_fn)
        self.conv2 = adf.Conv2d(32, 64, kernel_size=5, keep_variance_fn=self._keep_variance_fn)
        self.relu2 = adf.ReLU(keep_variance_fn=self._keep_variance_fn)
        self.maxpool2 = adf.MaxPool2d(keep_variance_fn=self._keep_variance_fn)
        self.fc1 = adf.Linear(1024, 1024, keep_variance_fn=self._keep_variance_fn)
        self.fc2 = adf.Linear(1024, 10, keep_variance_fn=self._keep_variance_fn)
        self.relu3 = adf.ReLU(keep_variance_fn=self._keep_variance_fn)

        if initialize_msra:
            finitialize_msra(self.modules())
            # finitialize_msra([self.fc2], small=True)
        else:
            finitialize_xavier(self.modules())
            # finitialize_xavier([self.fc2], small=True)

    def forward(self, example_dict):
        inputs = example_dict["input1"]
        inputs_mean = inputs
        inputs_variance = torch.zeros_like(inputs_mean) + self._noise_variance
        x = inputs_mean, inputs_variance
        x = self.conv1(*x)
        x = self.relu1(*x)
        x = self.maxpool1(*x)
        x = self.conv2(*x)
        x = self.relu2(*x)
        x = self.maxpool2(*x)
        x = [u.view(-1, 1024) for u in x]
        x = self.fc1(*x)
        x = self.relu3(*x)
        mean, variance = self.fc2(*x)
        return {"mean1": mean, "variance1": variance }
