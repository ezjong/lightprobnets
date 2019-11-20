from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def __init__(self, args, topk=(1, 2, 3), size_average=True, reduce=True):
        super(ClassificationLoss, self).__init__()
        self._cross_entropy = torch.nn.CrossEntropyLoss(
            size_average=size_average, reduce=reduce)
        self._topk = topk

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
        output = output_dict["output1"]
        target = target_dict["target1"]
        # compute losses
        cross_entropy = self._cross_entropy(output, target)
        # create dictonary for losses
        loss_dict = {
            "xe": cross_entropy,
        }
        acc_k = self._accuracy(output, target, topk=self._topk)
        for acc, k in zip(acc_k, self._topk):
            loss_dict["top%i" % k] = acc
        return loss_dict
