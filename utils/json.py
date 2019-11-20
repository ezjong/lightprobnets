from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json as jsn
import os
import sys
import unicodedata

from utils import six


def read_json(filename):
    def _convert_from_unicode(data):
        new_data = dict()
        for name, value in six.iteritems(data):
            if isinstance(name, six.string_types):
                name = unicodedata.normalize('NFKD', name).encode(
                    'ascii', 'ignore')
            if isinstance(value, six.string_types):
                value = unicodedata.normalize('NFKD', value).encode(
                    'ascii', 'ignore')
            if isinstance(value, dict):
                value = _convert_from_unicode(value)
            new_data[name] = value
        return new_data

    output_dict = None
    with open(filename, "r") as f:
        lines = f.readlines()
        try:
            output_dict = jsn.loads(''.join(lines), encoding='utf-8')
        except:
            raise ValueError('Could not read %s. %s' % (filename, sys.exc_info()[1]))
        output_dict = _convert_from_unicode(output_dict)
    return output_dict


def _replace_quotes(x):
    return x.replace("\'", "\"")


def _parse_value(value):
    if isinstance(value, tuple):
        value = list(value)

    if value is None:
        return "null"

    if isinstance(value, str):
        if value.lower() == "none":
            return "null"
        if value.lower() == "false":
            return "false"
        if value.lower() == "true":
            return "true"
        value = value.replace("\'", "\"")
        return "\"%s\"" % _replace_quotes(value)

    if isinstance(value, bool):
        return str(value).lower()

    if isinstance(value, list):
        result = "["
        for i, item in enumerate(value):
            result += _parse_value(item)
            if i < len(value) - 1:
                result += ", "
        result += "]"
        return result

    if isinstance(value, dict):
        result = "{"
        item_iterator = six.itersorteditems(value)
        for i, (dict_key, dict_value) in enumerate(item_iterator):
            result += "\"%s\": %s" % (dict_key, _parse_value(dict_value))
            if i < len(value) - 1:
                result += ", "
        result += "}"
        return result

    return "%s" % _replace_quotes(str(value))


# ----------------------------------------------------------------------------
# Writes all pairs to a filename for book keeping
# Either .txt or .json
# ----------------------------------------------------------------------------
def write_dictionary_to_file(input_dict, filename, sortkeys=False):
    # ensure dir
    d = os.path.dirname(filename)
    if not os.path.exists(d):
        os.makedirs(d)

    item_iterator = six.itersorteditems(input_dict) if sortkeys else six.iteritems(input_dict)

    # check for json extension
    ext = os.path.splitext(filename)[1]
    if ext == ".json":
        with open(filename, 'w') as file:
            file.write("{\n")
            for i, (key, value) in enumerate(item_iterator):
                file.write("  \"%s\": %s" % (key, _parse_value(value)))
                if i < len(input_dict) - 1:
                    file.write(',\n')
                else:
                    file.write('\n')
            file.write("}\n")
    else:
        with open(filename, 'w') as file:
            for key, value in item_iterator:
                file.write('%s: %s\n' % (key, value))
