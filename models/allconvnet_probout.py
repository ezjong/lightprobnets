from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import logging
from torch.nn import functional as tf


def finitialize_msra(modules, small=False):
    logging.info("Initializing MSRA")
    for layer in modules:
        if isinstance(layer, nn.Conv2d):  # convolution: bias=0, weight=msra
            nn.init.kaiming_normal_(layer.weight)
            if small:
                layer.weight.data.mul_(0.001)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


def finitialize_xavier(modules, small=False):
    logging.info("Initializing Xavier")
    for layer in modules:
        if isinstance(layer, nn.Conv2d):  # convolution: bias=0, weight=msra
            nn.init.xavier_normal_(layer.weight)
            if small:
                layer.weight.data.mul_(0.001)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


def make_conv(inchannels, outchannels, kernel_size, stride, nonlinear=True):
    padding = kernel_size // 2
    if nonlinear:
        return nn.Sequential(
            nn.Conv2d(
                inchannels, outchannels, kernel_size=kernel_size, padding=padding, stride=stride, bias=True),
            nn.ReLU()
        )
    else:
        return nn.Conv2d(
            inchannels, outchannels, kernel_size=kernel_size, padding=padding, stride=stride, bias=True)


class AllConvNetProbOut(nn.Module):
    def __init__(self, args, initialize_msra=False):
        super(AllConvNetProbOut, self).__init__()

        self.conv1   = make_conv(3, 96, kernel_size=3, stride=1)
        self.conv1_1 = make_conv(96, 96, kernel_size=3, stride=1)
        self.conv1_2 = make_conv(96, 96, kernel_size=3, stride=2, nonlinear=False)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2   = make_conv( 96, 192, kernel_size=3, stride=1)
        self.conv2_1 = make_conv(192, 192, kernel_size=3, stride=1)
        self.conv2_2 = make_conv(192, 192, kernel_size=3, stride=2, nonlinear=False)
        self.dropout2 = nn.Dropout(0.5)
        self.conv3   = make_conv(192, 192, kernel_size=3, stride=1)
        self.conv3_1 = make_conv(192, 192, kernel_size=1, stride=1)
        self.conv3_2_mu = make_conv(192,  10, kernel_size=1, stride=1, nonlinear=False)
        self.conv3_2_vr = make_conv(192,  10, kernel_size=1, stride=1, nonlinear=False)

        if initialize_msra:
            finitialize_msra(self.modules(), small=False)
            finitialize_msra([self.conv3_2_vr], small=True)
        else:
            finitialize_xavier(self.modules(), small=False)
            finitialize_xavier([self.conv3_2_vr], small=True)

    def forward(self, example_dict):
        x = example_dict["input1"]

        x = self.conv1(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.conv3_1(x)
        mu = self.conv3_2_mu(x)
        log_vr = self.conv3_2_vr(x)

        variance = tf.softplus(log_vr)
        batch_size = x.size(0)
        mu = mu.contiguous().view(batch_size, 10, -1).mean(dim=2)
        variance = variance.contiguous().view(batch_size, 10, -1).mean(dim=2)

        return {"mean1": mu, "variance1": variance }
