from __future__ import absolute_import
from __future__ import print_function

import numpy as  np
import torch
import torch.nn as nn

from .endpoint_error import downsample2d_as
from .endpoint_error import elementwise_epe


def elementwise_laplacian(input_flow, target_flow, min_variance, log_variance):
    if log_variance:
        predictions_mean, predictions_log_variance = input_flow
        predictions_variance = torch.exp(predictions_log_variance) + min_variance

    else:
        predictions_mean, predictions_variance = input_flow

    const = torch.sum(torch.log(predictions_variance), dim=1, keepdim=True)
    squared_difference = (target_flow - predictions_mean) ** 2

    weighted_epe = torch.sqrt(
        torch.sum(squared_difference / predictions_variance, dim=1, keepdim=True))

    return const + weighted_epe


class MultiScaleLaplacian(nn.Module):
    def __init__(self,
                 args,
                 num_scales=5,
                 num_highres_scales=2,
                 coarsest_resolution_loss_weight=0.32,
                 with_llh=False):

        super(MultiScaleLaplacian, self).__init__()
        self._args = args
        self._with_llh = with_llh
        self._num_scales = num_scales
        self._min_variance = args.model_min_variance
        self._log_variance = args.model_log_variance

        # ---------------------------------------------------------------------
        # start with initial scale
        # for "low-resolution" scales we apply a scale factor of 4
        # for "high-resolution" scales we apply a scale factor of 2
        #
        # e.g. for FlyingChairs  weights=[0.005, 0.01, 0.02, 0.08, 0.32]
        # ---------------------------------------------------------------------
        self._weights = [coarsest_resolution_loss_weight]
        num_lowres_scales = num_scales - num_highres_scales
        for k in range(num_lowres_scales - 1):
            self._weights += [self._weights[-1] / 4]
        for k in range(num_highres_scales):
            self._weights += [self._weights[-1] / 2]
        self._weights.reverse()
        assert (len(self._weights) == num_scales)  # sanity check

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs = [output_dict[key] for key in ["flow2", "flow3", "flow4", "flow5", "flow6"]]

            # div_flow trick
            target = self._args.model_div_flow * target_dict["target1"]

            total_loss = 0
            for i, output_i in enumerate(outputs):
                target_i = downsample2d_as(target, output_i[0])
                epe_i = elementwise_laplacian(
                    output_i, target_i,
                    min_variance=self._min_variance,
                    log_variance=self._log_variance)
                total_loss += self._weights[i] * epe_i.sum()
                loss_dict["epe%i" % (i + 2)] = epe_i.mean()
            loss_dict["total_loss"] = total_loss
        else:
            output = output_dict["flow1"]
            target = target_dict["target1"]
            epe = elementwise_epe(output[0], target)

            lapl = elementwise_laplacian(output, target,
                                         min_variance=self._min_variance,
                                         log_variance=self._log_variance)

            loss_dict["epe"] = epe.mean()
            loss_dict["total_loss"] = lapl.mean()

            if self._with_llh:
                llh = - 0.5 * lapl - np.log(8.0 * np.pi)
                loss_dict["llh"] = llh.mean()

        return loss_dict
