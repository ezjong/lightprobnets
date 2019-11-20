from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fnmatch
import itertools
import os
import socket
from datetime import datetime

from pytz import timezone


def cd_dotdot(path_or_filename):
    return os.path.abspath(os.path.join(os.path.dirname(path_or_filename), ".."))


def cd_dotdotdot(path_or_filename):
    return os.path.abspath(os.path.join(os.path.dirname(path_or_filename), "../.."))


def cd_dotdotdotdot(path_or_filename):
    return os.path.abspath(os.path.join(os.path.dirname(path_or_filename), "../../.."))


def datestr():
    pacific = timezone('US/Pacific')
    now = datetime.now(pacific)
    return '{}{:02}{:02}_{:02}{:02}'.format(now.year, now.month, now.day, now.hour, now.minute)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def hostname():
    name = socket.gethostname()
    n = name.find('.')
    if n > 0:
        name = name[:n]
    return name


def get_filenames(directory, match='*.*', not_match=()):
    if match is not None:
        if isinstance(match, str):
            match = [match]
    if not_match is not None:
        if isinstance(not_match, str):
            not_match = [not_match]

    result = []
    for dirpath, _, filenames in os.walk(directory):
        filtered_matches = list(itertools.chain.from_iterable(
            [fnmatch.filter(filenames, x) for x in match]))
        filtered_nomatch = list(itertools.chain.from_iterable(
            [fnmatch.filter(filenames, x) for x in not_match]))
        matched = list(set(filtered_matches) - set(filtered_nomatch))
        result += [os.path.join(dirpath, x) for x in matched]
    return result
