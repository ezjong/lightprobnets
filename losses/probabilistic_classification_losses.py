from __future__ import absolute_import
from __future__ import print_function

import logging

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import functional as tf

from contrib.distributions import Dirichlet
from contrib.distributions import SmoothOneHot


def _log(name, tensor):
    logging.info("\n%s=[%4.4f, %4.4f, %4.4f]" % (name, tensor.min(), tensor.mean(), tensor.max()))


class DirichletProbOutLoss(nn.Module):
    def __init__(self, args, label_smoothing=-1.0, mult=True, random_off_targets=True, topk=(1, 2, 3)):
        super(DirichletProbOutLoss, self).__init__()
        self._mult = mult
        self._nll = torch.nn.NLLLoss(size_average=True, reduce=True)
        self._cross_entropy = torch.nn.CrossEntropyLoss(
            size_average=True, reduce=True)

        self._crossentropy = torch.nn.CrossEntropyLoss()

        self._smoothed_onehot = SmoothOneHot(
            label_smoothing=label_smoothing, random_off_targets=random_off_targets)

        self._onehot = SmoothOneHot(label_smoothing=0)
        self._dirichlet = Dirichlet(argmax_smoothing=0.5)
        self._topk = topk
        if 'ADF' in str(args.model):
            self._log_bias_c1 = Parameter(-torch.ones(1, 1) * 7.0, requires_grad=True)
            self._log_bias_c2 = Parameter(-torch.ones(1, 1) * 7.0, requires_grad=True)
        else:
            self._log_bias_c1 = Parameter(-torch.ones(1, 1) * 0.0, requires_grad=True)
            self._log_bias_c2 = Parameter(-torch.ones(1, 1) * 0.0, requires_grad=True)
        self._softmax = nn.Softmax(dim=1)

        if label_smoothing < 0:
            raise ValueError("Please specify label_smoothing parameter for DirichletProbOutLoss!")

    def _accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def forward(self, output_dict, target_dict):
        outputs_mean = output_dict["mean1"]
        outputs_variance = output_dict["variance1"]
        target = target_dict["target1"]

        # convert to smoothed onehot labels
        num_classes = outputs_mean.size(-1)

        # onehot = self._onehot(target, num_classes=num_classes)
        smoothed_onehot = Variable(
            self._smoothed_onehot(target.data, num_classes=num_classes), requires_grad=False)

        c1 = tf.softplus(self._log_bias_c1)
        c2 = tf.softplus(self._log_bias_c2)

        # Vc1c2
        mu = self._softmax(outputs_mean)
        # stddev = torch.sum(mu * outputs_variance, dim=1, keepdim=True)
        stddev = torch.sqrt(torch.sum(mu ** 2 * outputs_variance, dim=1, keepdim=True))
        s = 1.0 / (1e-4  + c1 + c2 * stddev)
        alpha = mu * s if self._mult else mu / s

        predictions = alpha / alpha.sum(dim=-1, keepdim=True)
        log_predictions = torch.log(predictions)

        total_loss = - self._dirichlet(alpha, smoothed_onehot).mean()
        loss_dict = {}

        loss_dict["total_loss"] = total_loss
        loss_dict["xe"] = self._nll(log_predictions, target)
        acc_k = self._accuracy(log_predictions, target, topk=self._topk)
        for acc, k in zip(acc_k, self._topk):
            loss_dict["top%i" % k] = acc

        return loss_dict
