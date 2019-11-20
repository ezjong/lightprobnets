from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .flownet1s import FlowNetS
from .flownet_helpers import upsample2d_as


class FlowNetProbOut(nn.Module):
    def __init__(self, args, div_flow=0.05, min_variance=1e-3, log_variance=True):
        super(FlowNetProbOut, self).__init__()
        self._flownets = FlowNetS(args, num_pred=4)
        self._div_flow = div_flow

    def forward(self, input_dict):
        im1 = input_dict['input1']
        im2 = input_dict['input2']
        inputs = torch.cat((im1, im2), dim=1)

        output_dict = {}
        if self.training:
            flow2, flow3, flow4, flow5, flow6 = self._flownets(inputs)

            output_dict['flow2'] = flow2.chunk(chunks=2, dim=1)
            output_dict['flow3'] = flow3.chunk(chunks=2, dim=1)
            output_dict['flow4'] = flow4.chunk(chunks=2, dim=1)
            output_dict['flow5'] = flow5.chunk(chunks=2, dim=1)
            output_dict['flow6'] = flow6.chunk(chunks=2, dim=1)

        else:
            flow2 = self._flownets(inputs)
            flow2_mean, flow2_log_variance = flow2.chunk(chunks=2, dim=1)

            flow1_mean = (1.0 / self._div_flow) * upsample2d_as(flow2_mean, im1, mode="bilinear")

            z = upsample2d_as(torch.exp(flow2_log_variance) * (1.0/self._div_flow)**2, im1, mode="bilinear")
            flow1_log_variance = torch.log(z)

            output_dict['flow1'] = flow1_mean, flow1_log_variance

        return output_dict
