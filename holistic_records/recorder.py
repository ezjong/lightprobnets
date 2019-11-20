from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from .writers import types
from .writers import csv_writer


# ----------------------------------------------
# Epoch recorder:
#
#   To be initiated for every epoch.
#
# ----------------------------------------------
class EpochRecorder(object):
    def __init__(self,
                 args,
                 epoch,
                 dataset,
                 csv=True):
        self._root = args.save
        self._epoch = epoch
        self._dataset = dataset
        self._record_writers = []

        if not os.path.exists(self._root):
            os.makedirs(self._root)

        if csv:
            writer = csv_writer.CSVRecordWriter(
                args, root=os.path.join(self._root, "csv"))
            self._record_writers.append(writer)

    @property
    def root(self):
        return self._root

    @property
    def epoch(self):
        return self._epoch

    @property
    def dataset(self):
        return self._dataset

    @property
    def record_writers(self):
        return self._record_writers

    def _handle_record(self, record):
        for writer in self._record_writers:
            writer.handle_record(record)

    def add_scalars(self, example_basename, scalars, step=None, example_index=None):
        record = types.ScalarDictRecord(example_basename,
                                        data=scalars,
                                        step=step,
                                        example_index=example_index,
                                        epoch=self._epoch,
                                        dataset=self._dataset)
        self._handle_record(record)
