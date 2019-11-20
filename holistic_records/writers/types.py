from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch


def tensor2numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    else:
        if isinstance(tensor, torch.autograd.Variable):
            tensor = tensor.data
        if tensor.dim() == 3:
            return tensor.cpu().numpy().transpose([1,2,0])
        else:
            return tensor.cpu().numpy().transpose([0,2,3,1])


# --------------------------------------------------------
# Base Record Writer
# --------------------------------------------------------
class RecordWriter(object):

    def handle_record(self, record):
        raise ValueError("Not yet implemented!")


# --------------------------------------------------------
# Base Record
# --------------------------------------------------------
class Record(object):
    def __init__(self, example_basename, data, step=None, example_index=None, epoch=None, dataset=None):
        self.data = data
        self.dataset = dataset
        self.epoch = epoch
        self.example_basename = example_basename
        self.example_index = example_index
        self.step = step


def record2list(record):
    if len(record.example_basename) == 1:
        record.example_basename = [record.example_basename]
        record.example_index = [record.example_index]
        record.data = record.data.unsqueeze(dim=0)
    return record


# --------------------------------------------------------
# Dictionary of Scalar records
# --------------------------------------------------------
class ScalarDictRecord(Record):
    def __init__(self, example_basename, data, step=None, example_index=None, epoch=None, dataset=None):
        super(ScalarDictRecord, self).__init__(example_basename=example_basename,
                                               data=data,
                                               step=step,
                                               example_index=example_index,
                                               epoch=epoch,
                                               dataset=dataset)
