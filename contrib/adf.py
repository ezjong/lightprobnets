import operator
from collections import OrderedDict
from itertools import islice

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as tf
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.conv import _ConvTransposeMixin
from torch.nn.modules.utils import _pair

from contrib.interpolation import resize2D_as
from contrib.math import normpdf, normcdf


class ReLU(nn.Module):
    def __init__(self, keep_variance_fn=None):
        super(ReLU, self).__init__()
        self._keep_variance_fn = keep_variance_fn

    def forward(self, features_mean, features_variance):
        features_stddev = torch.sqrt(features_variance)
        div = features_mean / features_stddev
        pdf = normpdf(div)
        cdf = normcdf(div)
        outputs_mean = features_mean * cdf + features_stddev * pdf
        outputs_variance = (features_mean ** 2 + features_variance) * cdf \
                           + features_mean * features_stddev * pdf - outputs_mean ** 2
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01, keep_variance_fn=None):
        super(LeakyReLU, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self._negative_slope = negative_slope

    def forward(self, features_mean, features_variance):
        features_stddev = torch.sqrt(features_variance)
        div = features_mean / features_stddev
        pdf = normpdf(div)
        cdf = normcdf(div)
        negative_cdf = 1.0 - cdf
        mu_cdf = features_mean * cdf
        stddev_pdf = features_stddev * pdf
        squared_mean_variance = features_mean ** 2 + features_variance
        mean_stddev_pdf = features_mean * stddev_pdf
        mean_r = mu_cdf + stddev_pdf
        variance_r = squared_mean_variance * cdf + mean_stddev_pdf - mean_r ** 2
        mean_n = - features_mean * negative_cdf + stddev_pdf
        variance_n = squared_mean_variance * negative_cdf - mean_stddev_pdf - mean_n ** 2
        covxy = - mean_r * mean_n
        outputs_mean = mean_r - self._negative_slope * mean_n
        outputs_variance = variance_r \
                           + self._negative_slope * self._negative_slope * variance_n \
                           - 2.0 * self._negative_slope * covxy
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


class Dropout(nn.Module):
    def __init__(self, p=0.5, keep_variance_fn=None):
        super(Dropout, self).__init__()
        self._keep_variance_fn = keep_variance_fn

    def forward(self, inputs_mean, inputs_variance):
        z = torch.ones_like(inputs_mean)
        outputs_mean = inputs_mean * z
        outputs_variance = inputs_variance * (z**2)
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance

class MaxPool2d(nn.Module):
    def __init__(self, keep_variance_fn=None):
        super(MaxPool2d, self).__init__()
        self._keep_variance_fn = keep_variance_fn

    def _max_pool_internal(self, mu_a, mu_b, var_a, var_b):
        stddev = torch.sqrt(var_a + var_b)
        ab = mu_a - mu_b
        alpha = ab / stddev
        pdf = normpdf(alpha)
        cdf = normcdf(alpha)
        z_mu = stddev * pdf + ab * cdf + mu_b
        z_var = ((mu_a + mu_b) * stddev * pdf +
                 (mu_a ** 2 + var_a) * cdf +
                 (mu_b ** 2 + var_b) * (1.0 - cdf) - z_mu ** 2)
        if self._keep_variance_fn is not None:
            z_var = self._keep_variance_fn(z_var)
        return z_mu, z_var

    def _max_pool_1x2(self, inputs_mean, inputs_variance):
        mu_a = inputs_mean[:, :, :, 0::2]
        mu_b = inputs_mean[:, :, :, 1::2]
        var_a = inputs_variance[:, :, :, 0::2]
        var_b = inputs_variance[:, :, :, 1::2]
        outputs_mean, outputs_variance = self._max_pool_internal(
            mu_a, mu_b, var_a, var_b)
        return outputs_mean, outputs_variance

    def _max_pool_2x1(self, inputs_mean, inputs_variance):
        mu_a = inputs_mean[:, :, 0::2, :]
        mu_b = inputs_mean[:, :, 1::2, :]
        var_a = inputs_variance[:, :, 0::2, :]
        var_b = inputs_variance[:, :, 1::2, :]
        outputs_mean, outputs_variance = self._max_pool_internal(
            mu_a, mu_b, var_a, var_b)
        return outputs_mean, outputs_variance

    def forward(self, inputs_mean, inputs_variance):
        z_mean, z_variance = self._max_pool_1x2(inputs_mean, inputs_variance)
        outputs_mean, outputs_variance = self._max_pool_2x1(z_mean, z_variance)
        return outputs_mean, outputs_variance


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, keep_variance_fn=None):
        super(Linear, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs_mean, inputs_variance):
        outputs_mean = tf.linear(inputs_mean, self.weight, self.bias)
        outputs_variance = tf.linear(inputs_variance, self.weight**2, None)
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 keep_variance_fn=None):
        self._keep_variance_fn = keep_variance_fn
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, inputs_mean, inputs_variance):
        outputs_mean = tf.conv2d(
            inputs_mean, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        outputs_variance = tf.conv2d(
            inputs_variance, self.weight ** 2, None, self.stride, self.padding, self.dilation, self.groups)
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


class ConvTranspose2d(_ConvTransposeMixin, _ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1,
                 keep_variance_fn=None):
        self._keep_variance_fn = keep_variance_fn
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(ConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias)

    def forward(self, inputs_mean, inputs_variance, output_size=None):
        output_padding = self._output_padding(inputs_mean, output_size)
        outputs_mean = tf.conv_transpose2d(
            inputs_mean, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
        outputs_variance = tf.conv_transpose2d(
            inputs_variance, self.weight ** 2, None, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


def concatenate_as(tensor_list, tensor_as, dim, mode="bilinear"):
    means = [resize2D_as(x[0], tensor_as[0], mode=mode) for x in tensor_list]
    variances = [resize2D_as(x[1], tensor_as[0], mode=mode) for x in tensor_list]
    means = torch.cat(means, dim=dim)
    variances = torch.cat(variances, dim=dim)
    return means, variances


class Sequential(nn.Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, inputs, inputs_variance):
        for module in self._modules.values():
            inputs, inputs_variance = module(inputs, inputs_variance)

        return inputs, inputs_variance
