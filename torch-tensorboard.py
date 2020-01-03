# -*- coding: utf-8 -*-
from torch.utils.tensorboard import SummaryWriter
from numpy import genfromtxt
import numpy as np
writer = SummaryWriter()
data = genfromtxt(
    '/Users/suman/quantum/mltransfer/mlencrypt-research/results/analysis/hparams/rawdata.csv', delimiter=',', dtype=None)[1:-1]


def log_hparams(run):
    hparam_dict = {
        'K': int(float(run[1])),
        'N': int(float(run[2])),
        'L': int(float(run[3])),
        'update-rule': run[0].decode("utf-8"),
        'attack': run[4].decode("utf-8"),
    }
    metric_dict = {
        'hparams/training-time': float(run[5]),
        'hparams/Eve-score': float(run[6])
    }
    writer.add_hparams(hparam_dict, metric_dict)


# np.apply_along_axis(log_hparams, axis=1, arr=data)
# don't log hparams because the axis labels are incorrect as of 01-03-2020

data = np.where(data == b'hebbian', -1, data)
data = np.where(data == b'anti_hebbian', 0, data)
data = np.where(data == b'random_walk', 1, data)
data = np.where(data == b'geometric', 1, data)
data = np.where(data == b'none', -1, data)
data = data.astype('float64')
writer.add_embedding(data, tag='hparams_embedding')
writer.close()
