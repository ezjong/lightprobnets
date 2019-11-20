from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import logging
import os
import re
import sys
import time

import colorama
import tqdm

from utils import strings


def get_default_logging_format(colorize=False, brackets=False):
    style = colorama.Style.DIM if colorize else ''
    # color = colorama.Fore.CYAN if colorize else ''
    color = colorama.Fore.WHITE if colorize else ''
    reset = colorama.Style.RESET_ALL if colorize else ''
    if brackets:
        result = "{}{}[%(asctime)s]{} %(message)s".format(style, color, reset)
    else:
        result = "{}{}%(asctime)s{} %(message)s".format(style, color, reset)
    return result


def get_default_logging_datefmt():
    return "%Y-%m-%d %H:%M:%S"


def log_module_info(module):
    lines = module.__str__().split("\n")
    for line in lines:
        logging.info(line)


class LogbookFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super(LogbookFormatter, self).__init__(fmt=fmt, datefmt=datefmt)
        self._re = re.compile(r"\033\[[0-9]+m")

    def remove_colors_from_msg(self, msg):
        msg = re.sub(self._re, "", msg)
        return msg

    def format(self, record=None):
        record.msg = self.remove_colors_from_msg(record.msg)
        record.msg = strings.as_unicode(record.msg)
        return super(LogbookFormatter, self).format(record)


class ConsoleFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super(ConsoleFormatter, self).__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record=None):
        indent = sys.modules[__name__].global_indent
        record.msg = " " * indent + record.msg
        return super(ConsoleFormatter, self).format(record)


class SkipLogbookFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.LOGBOOK


# -----------------------------------------------------------------
# Subclass tqdm to achieve two things:
#   1) Output the progress bar into the logbook.
#   2) Remove the comma before {postfix} because it's annoying.
# -----------------------------------------------------------------
class TqdmToLogger(tqdm.tqdm):
    def __init__(self, iterable=None, desc=None, total=None, leave=True,
                 file=None, ncols=None, mininterval=0.1,
                 maxinterval=10.0, miniters=None, ascii=None, disable=False,
                 unit='it', unit_scale=False, dynamic_ncols=False,
                 smoothing=0.3, bar_format=None, initial=0, position=None,
                 postfix=None,
                 logging_on_close=True,
                 logging_on_update=False):

        super(TqdmToLogger, self).__init__(
            iterable=iterable, desc=desc, total=total, leave=leave,
            file=file, ncols=ncols, mininterval=mininterval,
            maxinterval=maxinterval, miniters=miniters, ascii=ascii, disable=disable,
            unit=unit, unit_scale=unit_scale, dynamic_ncols=dynamic_ncols,
            smoothing=smoothing, bar_format=bar_format, initial=initial, position=position,
            postfix=postfix)

        self._logging_on_close = logging_on_close
        self._logging_on_update = logging_on_update
        self._closed = False

    @staticmethod
    def format_meter(n, total, elapsed, ncols=None, prefix='', ascii=False,
                     unit='it', unit_scale=False, rate=None, bar_format=None,
                     postfix=None, unit_divisor=1000):

        meter = tqdm.tqdm.format_meter(
            n=n, total=total, elapsed=elapsed, ncols=ncols, prefix=prefix, ascii=ascii,
            unit=unit, unit_scale=unit_scale, rate=rate, bar_format=bar_format,
            postfix=postfix, unit_divisor=unit_divisor)

        # get rid of that stupid comma before the postfix
        if postfix is not None:
            postfix_with_comma = ", %s" % postfix
            meter = meter.replace(postfix_with_comma, postfix)

        return meter

    def update(self, n=1):
        if self._logging_on_update:
            msg = self.__repr__()
            self._write_msg_to_logbook(msg)
        return super(TqdmToLogger, self).update(n=n)

    def _write_msg_to_logbook(self, msg):
        # Note: This fails sometimes, when the previous writing has not been over yet
        # Hence, we wrap it in a loop waiting for success
        while True:
            try:
                logging.logbook(msg)
            except IOError, e:
                if e.errno != errno.EINTR:
                    raise ValueError("Logbook TQDM IOError")
                else:
                    time.sleep(0.2)
            else:
                break

    def close(self):
        if self._logging_on_close and not self._closed:
            msg = self.__repr__()
            self._write_msg_to_logbook(msg)
            self._closed = True
        return super(TqdmToLogger, self).close()


def tqdm_with_logging(iterable=None, desc=None, total=None, leave=True,
                      ncols=None, mininterval=0.1,
                      maxinterval=10.0, miniters=None, ascii=None, disable=False,
                      unit="it", unit_scale=False, dynamic_ncols=False,
                      smoothing=0.3, bar_format=None, initial=0, position=None,
                      postfix=None,
                      logging_on_close=True,
                      logging_on_update=False):
    return TqdmToLogger(
        iterable=iterable, desc=desc, total=total, leave=leave,
        ncols=ncols, mininterval=mininterval,
        maxinterval=maxinterval, miniters=miniters, ascii=ascii, disable=disable,
        unit=unit, unit_scale=unit_scale, dynamic_ncols=dynamic_ncols,
        smoothing=smoothing, bar_format=bar_format, initial=initial, position=position,
        postfix=postfix,
        logging_on_close=logging_on_close,
        logging_on_update=logging_on_update)


# ----------------------------------------------------------------------------------------
# Comprehensively adds a new logging level to the `logging` module and the
# currently configured logging class.
# e.g. addLoggingLevel('TRACE', logging.DEBUG - 5)
# ----------------------------------------------------------------------------------------
def add_logging_level(level_name, level_num, method_name=None):
    if not method_name:
        method_name = level_name.lower()
    if hasattr(logging, level_name):
        raise AttributeError('{} already defined in logging module'.format(level_name))
    if hasattr(logging, method_name):
        raise AttributeError('{} already defined in logging module'.format(method_name))
    if hasattr(logging.getLoggerClass(), method_name):
        raise AttributeError('{} already defined in logger class'.format(method_name))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(level_num, message, *args, **kwargs)

    logging.addLevelName(level_num, level_name)
    setattr(logging, level_name, level_num)
    setattr(logging.getLoggerClass(), method_name, logForLevel)
    setattr(logging, method_name, logToRoot)


def configure_logging(filename=None):
    # set global indent level
    sys.modules[__name__].global_indent = 0

    # add custom tqdm logger
    add_logging_level("LOGBOOK", 1000)

    # create logger
    root_logger = logging.getLogger("")
    root_logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    fmt = get_default_logging_format(colorize=True, brackets=False)
    datefmt = get_default_logging_datefmt()
    formatter = ConsoleFormatter(fmt=fmt, datefmt=datefmt)
    console.setFormatter(formatter)

    # Skip logging.tqdm requests for console outputs
    skip_logbook_filter = SkipLogbookFilter()
    console.addFilter(skip_logbook_filter)

    # add console to root_logger
    root_logger.addHandler(console)

    # add logbook
    if filename is not None:
        # ensure dir
        d = os.path.dirname(filename)
        if not os.path.exists(d):
            os.makedirs(d)

        # --------------------------------------------------------------------------------------
        # Configure handler that removes color codes from logbook
        # --------------------------------------------------------------------------------------
        logbook = logging.FileHandler(filename=filename, mode="a", encoding="utf-8")
        logbook.setLevel(logging.INFO)
        fmt = get_default_logging_format(colorize=False, brackets=True)
        logbook_formatter = LogbookFormatter(fmt=fmt, datefmt=datefmt)
        logbook.setFormatter(logbook_formatter)
        root_logger.addHandler(logbook)

        # --------------------------------------------------------------------------------------
        # Not necessary
        # --------------------------------------------------------------------------------------
        # logbook_tqdm = logging.FileHandler(filename=filename, mode="a", encoding="utf-8")
        # logbook_tqdm.setLevel(logging.TQDM)
        # fmt = get_default_logging_format(colorize=False, brackets=True)
        # remove_colors_formatter = LogbookFormatter(fmt=fmt, datefmt=datefmt)
        # logbook_tqdm.setFormatter(remove_colors_formatter)
        # root_logger.addHandler(logbook_tqdm)


class LoggingBlock:
    def __init__(self, title, emph=False):
        self._emph = emph
        bright = colorama.Style.BRIGHT
        cyan = colorama.Fore.CYAN
        reset = colorama.Style.RESET_ALL
        if emph:
            logging.info("%s==>%s %s%s%s" % (cyan, reset, bright, title, reset))
        else:
            logging.info(title)

    def __enter__(self):
        sys.modules[__name__].global_indent += 2
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.modules[__name__].global_indent -= 2
