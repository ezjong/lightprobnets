from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import fnmatch
import itertools
import re


def as_unicode(msg):
    return ''.join([i if ord(i) < 128 else ' ' for i in msg])


def search_and_replace(string, regex, replace):
    while True:
        match = re.search(regex, string)
        if match:
            string = string.replace(match.group(0), replace)
        else:
            break
    return string


def filter_list_of_strings(lst, include="*", exclude=()):
    filtered_matches = list(itertools.chain.from_iterable([fnmatch.filter(lst, x) for x in include]))
    filtered_nomatch = list(itertools.chain.from_iterable([fnmatch.filter(lst, x) for x in exclude]))
    matched = list(set(filtered_matches) - set(filtered_nomatch))
    return matched


def as_bool_or_none(v):
    if v.strip().lower() == "none":
        return None
    if v.strip().lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.strip().lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def as_string_or_none(v):
    if v.strip().lower() == "none":
        return None
    return v


def as_dict_or_none(v):
    if v.strip().lower() == "none":
        return None
    return ast.literal_eval(v)


def as_list_or_none(v, astype):
    if v.strip().lower() == "none":
        return None
    return [astype(x.strip()) for x in v.strip()[1:-1].split(',')]


def as_intlist_or_none(v):
    return as_list_or_none(v, int)


def as_stringlist_or_none(v):
    return as_list_or_none(v, str)


def as_booleanlist_or_none(v):
    if v.strip().lower() == "none":
        return None
    return [as_bool_or_none(x.strip()) for x in v.strip()[1:-1].split(',')]


def as_floatlist_or_none(v):
    return as_list_or_none(v, float)
