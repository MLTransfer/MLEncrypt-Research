# -*- coding: utf-8 -*-
from torch.utils.tensorboard import SummaryWriter
from numpy import genfromtxt
import numpy as np
writer = SummaryWriter()
data = genfromtxt(
    '/Users/suman/quantum/mltransfer/mlencrypt-research/results/analysis/hparams/averaged/averaged.csv', delimiter=',', dtype=None)[1:-1]


def log_hparams(run):
    hparam_dict = {
        'K': int(float(run[1])),
        'N': int(float(run[2])),
        'L': int(float(run[3])),
        'update-rule': run[0].decode("utf-8"),
    }
    metric_dict = {
        'hparams/training-time': float(run[4]),
        'hparams/Eve-score-none': float(run[5]),
        'hparams/Eve-score-geometric': float(run[6]),
        'hparams/Eve-score-average': float(run[7]),
    }
    writer.add_hparams(hparam_dict, metric_dict)


# np.apply_along_axis(log_hparams, axis=1, arr=data)
# don't log hparams because the axis labels are incorrect as of 01-03-2020


labels = np.apply_along_axis(
# round continuous variables to 3 decimals
    lambda row: f"{row[0]},{row[1]},{row[2]},{row[3]}:{row[4]},{row[5]},{row[6]}", axis=1, arr=data)
data = np.where(data == 'hebbian', -1, data)
data = np.where(data == 'anti_hebbian', 0, data)
data = np.where(data == 'random_walk', 1, data)
data = np.where(data == 'geometric', 1, data)
data = np.where(data == 'none', -1, data)
data = data.astype('float64')
writer.add_embedding(data, metadata=labels.tolist(), tag='hparams_embedding')
writer.close()
