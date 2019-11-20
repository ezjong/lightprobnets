from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from contrib import adf
from .flownet_helpers import initialize_msra
from contrib.adf import concatenate_as
from .flownet_helpers import upsample2d_as


def keep_variance(x, min_variance):
    return x + min_variance


def conv(in_planes, out_planes, kernel_size, stride, pad, nonlinear, bias, keep_variance_fn=None):
    if nonlinear:
        return adf.Sequential(
            adf.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=pad, bias=bias, keep_variance_fn=keep_variance_fn),
            adf.LeakyReLU(0.1, keep_variance_fn=keep_variance_fn)
        )
    else:
        return adf.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, bias=bias,
            keep_variance_fn=keep_variance_fn)


def deconv(in_planes, out_planes, kernel_size, stride, pad, nonlinear, bias, keep_variance_fn=None):
    if nonlinear:
        return adf.Sequential(
            adf.ConvTranspose2d(
                in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=pad, bias=bias, keep_variance_fn=keep_variance_fn),
            adf.LeakyReLU(0.1, keep_variance_fn=keep_variance_fn)
        )
    else:
        return adf.ConvTranspose2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, bias=bias,
            keep_variance_fn=keep_variance_fn)


class _FlowNetADF(nn.Module):
    def __init__(self, args, num_pred=2, keep_variance_fn=None):
        super(_FlowNetADF, self).__init__()
        self._num_pred = num_pred

        def make_conv(in_planes, out_planes, kernel_size, stride):
            pad = kernel_size // 2
            return conv(in_planes, out_planes, kernel_size=kernel_size,
                        stride=stride, pad=pad, nonlinear=True, bias=True,
                        keep_variance_fn=keep_variance_fn)

        self._conv1   = make_conv(   6,   64, kernel_size=7, stride=2)
        self._conv2   = make_conv(  64,  128, kernel_size=5, stride=2)
        self._conv3   = make_conv( 128,  256, kernel_size=5, stride=2)
        self._conv3_1 = make_conv( 256,  256, kernel_size=3, stride=1)
        self._conv4   = make_conv( 256,  512, kernel_size=3, stride=2)
        self._conv4_1 = make_conv( 512,  512, kernel_size=3, stride=1)
        self._conv5   = make_conv( 512,  512, kernel_size=3, stride=2)
        self._conv5_1 = make_conv( 512,  512, kernel_size=3, stride=1)
        self._conv6   = make_conv( 512, 1024, kernel_size=3, stride=2)
        self._conv6_1 = make_conv(1024, 1024, kernel_size=3, stride=1)

        def make_deconv(in_planes, out_planes):
            return deconv(in_planes, out_planes, kernel_size=4, stride=2, pad=1,
                          nonlinear=True, bias=False, keep_variance_fn=keep_variance_fn)

        self._deconv5 = make_deconv(1024,            512)
        self._deconv4 = make_deconv(1024 + num_pred, 256)
        self._deconv3 = make_deconv( 768 + num_pred, 128)
        self._deconv2 = make_deconv( 384 + num_pred,  64)

        def make_predict(in_planes, out_planes):
            return conv(in_planes, out_planes, kernel_size=3, stride=1, pad=1,
                        nonlinear=False, bias=True, keep_variance_fn=keep_variance_fn)

        self._predict_flow6 = make_predict(1024,            num_pred)
        self._predict_flow5 = make_predict(1024 + num_pred, num_pred)
        self._predict_flow4 = make_predict( 768 + num_pred, num_pred)
        self._predict_flow3 = make_predict( 384 + num_pred, num_pred)
        self._predict_flow2 = make_predict( 192 + num_pred, num_pred)

        def make_upsample(in_planes, out_planes):
            return deconv(in_planes, out_planes, kernel_size=4, stride=2, pad=1,
                          nonlinear=False, bias=False, keep_variance_fn=keep_variance_fn)

        self._upsample_flow6_to_5 = make_upsample(num_pred, num_pred)
        self._upsample_flow5_to_4 = make_upsample(num_pred, num_pred)
        self._upsample_flow4_to_3 = make_upsample(num_pred, num_pred)
        self._upsample_flow3_to_2 = make_upsample(num_pred, num_pred)

        initialize_msra(self.modules())

    def forward(self, *inputs):
        conv1 = self._conv1(*inputs)
        conv2 = self._conv2(*conv1)
        conv3_1 = self._conv3_1(*self._conv3(*conv2))
        conv4_1 = self._conv4_1(*self._conv4(*conv3_1))
        conv5_1 = self._conv5_1(*self._conv5(*conv4_1))
        conv6_1 = self._conv6_1(*self._conv6(*conv5_1))

        predict_flow6        = self._predict_flow6(*conv6_1)

        upsampled_flow6_to_5 = self._upsample_flow6_to_5(*predict_flow6)
        deconv5              = self._deconv5(*conv6_1)
        concat5              = concatenate_as((conv5_1, deconv5, upsampled_flow6_to_5), conv5_1, dim=1)
        predict_flow5        = self._predict_flow5(*concat5)

        upsampled_flow5_to_4 = self._upsample_flow5_to_4(*predict_flow5)
        deconv4              = self._deconv4(*concat5)
        concat4              = concatenate_as((conv4_1, deconv4, upsampled_flow5_to_4), conv4_1, dim=1)
        predict_flow4        = self._predict_flow4(*concat4)

        upsampled_flow4_to_3 = self._upsample_flow4_to_3(*predict_flow4)
        deconv3              = self._deconv3(*concat4)
        concat3              = concatenate_as((conv3_1, deconv3, upsampled_flow4_to_3), conv3_1, dim=1)
        predict_flow3        = self._predict_flow3(*concat3)

        upsampled_flow3_to_2 = self._upsample_flow3_to_2(*predict_flow3)
        deconv2              = self._deconv2(*concat3)
        concat2              = concatenate_as((conv2, deconv2, upsampled_flow3_to_2), conv2, dim=1)
        predict_flow2        = self._predict_flow2(*concat2)

        if self.training:
            return predict_flow2, predict_flow3, predict_flow4, predict_flow5, predict_flow6
        else:
            return predict_flow2


class FlowNetADF(nn.Module):
    def __init__(self, args, noise_variance=1e-3, min_variance=1e-3, div_flow=0.05, log_variance=False):
        super(FlowNetADF, self).__init__()
        self._noise_variance = noise_variance
        self._keep_variance_fn = lambda x: keep_variance(x, min_variance=min_variance)
        self._flownetadf = _FlowNetADF(args, keep_variance_fn=self._keep_variance_fn)
        self._div_flow = div_flow

    def forward(self, input_dict):
        im1 = input_dict['input1']
        im2 = input_dict['input2']
        inputs_mean = torch.cat((im1, im2), dim=1)
        inputs_variance = torch.zeros_like(inputs_mean) + self._noise_variance

        output_dict = {}
        if self.training:
            flow2, flow3, flow4, flow5, flow6 = self._flownetadf(inputs_mean, inputs_variance)
            output_dict['flow2'] = flow2
            output_dict['flow3'] = flow3
            output_dict['flow4'] = flow4
            output_dict['flow5'] = flow5
            output_dict['flow6'] = flow6
        else:
            flow2_mean, flow2_variance = self._flownetadf(inputs_mean, inputs_variance)

            flow1_mean = (1.0 / self._div_flow) * upsample2d_as(flow2_mean, im1, mode="bilinear")
            flow1_variance = (1.0 / self._div_flow)**2 * upsample2d_as(flow2_variance, im1, mode="bilinear")
            output_dict['flow1'] = flow1_mean, flow1_variance

        return output_dict
