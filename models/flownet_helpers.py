from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from contrib.interpolation import resize2D_as
import torch.nn.functional as tf
import logging


def conv(in_planes, out_planes, kernel_size, stride, pad, nonlinear, bias):
    if nonlinear:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=pad, bias=bias),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, bias=bias)


def deconv(in_planes, out_planes, kernel_size, stride, pad, nonlinear, bias):
    if nonlinear:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=pad, bias=bias),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, bias=bias)


def concatenate_as(tensor_list, tensor_as, dim, mode="bilinear"):
    tensor_list = [resize2D_as(x, tensor_as, mode=mode) for x in tensor_list]
    return torch.cat(tensor_list, dim=dim)


def upsample2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return tf.upsample(inputs, [h, w], mode=mode, align_corners=True)


def keep_variance(inputs_variance, min_variance):
    return inputs_variance + min_variance


def initialize_msra(modules):
    logging.info("Initializing MSRA")
    for layer in modules:
        if isinstance(layer, nn.Conv2d):  # convolution: bias=0, weight=msra
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.ConvTranspose2d):  # deconvolution: bias=0, weigth=msra
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.LeakyReLU):
            pass

        elif isinstance(layer, nn.Sequential):
            pass

        elif "models" in str(type(layer)) and "FlowNet" in str(type(layer)):
            # these are high-level FlowNet models
            pass
