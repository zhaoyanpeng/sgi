import torch
from collections import defaultdict

class Stats(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self._stats = defaultdict(float)

    @property
    def stats(self):
        return self._stats

    def __call__(self, **kwargs):
        for k, v in kwargs.items():
            self._stats[k] += v

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = self.count = 0

    def __call__(self, val, n=1):
        self.count += n
        self.sum += val * n

    @property
    def average(self):
        return self.sum / self.count
