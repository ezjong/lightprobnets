from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class MovingAverage:
    postfix = "avg"

    def __init__(self):
        self._sum = 0.0
        self._count = 0

    def add_value(self, sigma, addcount=1):
        self._sum += sigma
        self._count += addcount

    def add_average(self, avg, addcount):
        self._sum += avg * addcount
        self._count += addcount

    def mean(self):
        return self._sum / self._count


class ExponentialMovingAverage:
    postfix = "ema"

    def __init__(self, alpha=0.7):
        self._weighted_sum = 0.0
        self._weighted_count = 0
        self._alpha = alpha

    def add_value(self, sigma, addcount=1):
        self._weighted_sum = sigma + (1.0 - self._alpha) * self._weighted_sum
        self._weighted_count = 1 + (1.0 - self._alpha) * self._weighted_count

    def add_average(self, avg, addcount):
        self._weighted_sum = avg * addcount + (1.0 - self._alpha) * self._weighted_sum
        self._weighted_count = addcount + (1.0 - self._alpha) * self._weighted_count

    def mean(self):
        return self._weighted_sum / self._weighted_count
