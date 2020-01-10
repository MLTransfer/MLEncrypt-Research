# -*- coding: utf-8 -*-
from torch.utils.tensorboard import SummaryWriter
from numpy import genfromtxt
import numpy as np
writer = SummaryWriter()
# filepath can be a URL, but file will be downloaded to cwd
filepath = '/Users/suman/quantum/mltransfer/mlencrypt-research/results/analysis/hparams/original/rawdata.csv'  # noqa
data = genfromtxt(filepath, delimiter=',', dtype=None, encoding=None)[1:-1]


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
        'hparams/adversary-score': float(run[6])
    }
    writer.add_hparams(hparam_dict, metric_dict)


# np.apply_along_axis(log_hparams, axis=1, arr=data)
# don't log hparams because the axis labels are incorrect as of 01-03-2020


def to_label(row):
    def shorten_update_rule(update_rule):
        if update_rule == 'hebbian':
            return 'h'
        elif update_rule == 'anti_hebbian':
            return 'ah'
        elif update_rule == 'random_walk':
            return 'rw'
        else:
            return 'o'

    def shorten_attack(attack):
        if attack == 'none':
            return 'n'
        elif attack == 'geometric':
            return 'g'
        else:
            return 'o'
    ur = shorten_update_rule(row[0])
    K = int(float(row[1]))
    N = int(float(row[2]))
    L = int(float(row[3]))
    attack = shorten_attack(row[4])
    training_time = round(float(row[5]))
    adversary_score = round(float(row[6]), 3)
    return f"{ur},{K},{N},{L},{attack}:{training_time},{adversary_score}"


labels = np.apply_along_axis(to_label, axis=1, arr=data)

data = np.where(data == 'hebbian', -1, data)
data = np.where(data == 'anti_hebbian', 0, data)
data = np.where(data == 'random_walk', 1, data)
data = np.where(data == 'geometric', 1, data)
data = np.where(data == 'none', -1, data)
data = data.astype('float64')
writer.add_embedding(data, metadata=labels.tolist(), tag='hparams_embedding')
writer.close()
