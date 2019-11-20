from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import collections

from . import types


class CSVRecordWriter(types.RecordWriter):
    def __init__(self, args, root):
        self._args = args
        self._root = root
        if not os.path.exists(root):
            os.makedirs(root)

        self._sniffer = csv.Sniffer()

    def handle_scalar_dict(self, record):
        filename = "%s/%s_%s.csv" % (self._root, record.dataset, record.example_basename)

        # create sorted dictionary such that epoch/step is on the left
        dict_of_values = collections.OrderedDict()
        dict_of_values["epoch"] = record.epoch
        dict_of_values["step"] = record.step

        for key in sorted(record.data.keys()):
            dict_of_values[key] = record.data[key]

        # figure out if file has header already
        has_header = False

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        # create file with headers
        if not os.path.isfile(filename):

            with open(filename, "w") as file:
                writer = csv.DictWriter(file, fieldnames=dict_of_values.keys())
                writer.writeheader()
                writer.writerow(dict_of_values)

        else:
            with open(filename, "r") as file:
                has_header = self._sniffer.has_header(file.read(1024))

            with open(filename, "a") as file:
                writer = csv.DictWriter(file, fieldnames=dict_of_values.keys())
                if not has_header:
                    writer.writeheader()
                writer.writerow(dict_of_values)

    def handle_record(self, record):
        if isinstance(record, types.ScalarDictRecord):
            return self.handle_scalar_dict(record)
