from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf


def elementwise_epe(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p=2, dim=1)


def downsample2d_as(inputs, target_as):
    _, _, h, w = target_as.size()
    return tf.adaptive_avg_pool2d(inputs, [h, w])


def downsample_flow_as(flow, output_as):
    size_inputs = flow.size()[2:4]
    size_targets = output_as.size()[2:4]
    if all([size_inputs == size_targets]):
        return flow
    elif any([size_targets < size_inputs]):
        resized_flow = tf.adaptive_avg_pool2d(flow, size_targets)  # downscaling
    else:
        resized_flow = tf.upsample(flow, size=size_targets, mode="bilinear")  # upsampling
    # correct scaling of flow
    u, v = resized_flow.chunk(2, dim=1)
    u *= float(size_targets[1] / size_inputs[1])
    v *= float(size_targets[0] / size_inputs[0])
    return torch.cat([u, v], dim=1)


class MultiScaleEPE(nn.Module):
    def __init__(self,
                 args,
                 num_scales=5,
                 num_highres_scales=2,
                 coarsest_resolution_loss_weight=0.32,
                 correct_flow_scale=False):

        super(MultiScaleEPE, self).__init__()
        self._args = args
        self._num_scales = num_scales
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

        if correct_flow_scale:
            self._fdownsample_flow = downsample_flow_as
        else:
            self._fdownsample_flow = downsample2d_as

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs = [output_dict[key] for key in ["flow2", "flow3", "flow4", "flow5", "flow6"]]

            # div_flow trick
            target = self._args.model_div_flow * target_dict["target1"]

            total_loss = 0
            for i, output_i in enumerate(outputs):
                target_i = self._fdownsample_flow(target, output_i)
                epe_i = elementwise_epe(output_i, target_i)
                total_loss += self._weights[i] * epe_i.sum()
                loss_dict["epe%i" % (i + 2)] = epe_i.mean()
            loss_dict["total_loss"] = total_loss
        else:
            output = output_dict["flow1"]
            target = target_dict["target1"]
            epe = elementwise_epe(output, target)
            loss_dict["epe"] = epe.mean()

        return loss_dict
