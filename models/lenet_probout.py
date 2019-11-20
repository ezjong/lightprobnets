from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as tf
import torch


class LeNetProbOut(nn.Module):
    def __init__(self, args):
        super(LeNetProbOut, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 20)

    def forward(self, example_dict):
        x = example_dict["input1"]
        x = tf.max_pool2d(tf.relu(self.conv1(x)), 2)
        x = tf.max_pool2d(tf.relu(self.conv2(x)), 2)
        x = x.view(-1, 1024)
        x = tf.relu(self.fc1(x))
        out = self.fc2(x)
        mean, log_variance = out.chunk(chunks=2, dim=1)

        return {"mean1": mean, "variance1": torch.exp(log_variance) }
