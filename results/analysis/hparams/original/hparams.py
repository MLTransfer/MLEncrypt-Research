import tensorflow as tf  # 2.0.0 or 2.1.0
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorboard.plugins.hparams import plugin_data_pb2
from tensorboard.backend.event_processing import event_accumulator
import glob
import csv
import pandas as pd


def get_hparams(logfile):
    si = summary_iterator(logfile)

    for event in si:
        for value in event.summary.value:
            proto_bytes = value.metadata.plugin_data.content
            plugin_data = plugin_data_pb2.HParamsPluginData.FromString(
                proto_bytes)
            if plugin_data.HasField("session_start_info"):
                hparams = plugin_data.session_start_info.hparams
                K = hparams['tpm_k'].number_value
                N = hparams['tpm_n'].number_value
                L = hparams['tpm_l'].number_value
                update_rule = hparams['update_rule'].string_value
                attack_raw = hparams['attack'].string_value
                attack = attack_raw if attack_raw else 'none'
                del attack_raw
                return K, N, L, update_rule, attack


def get_metrics(logfile):
    event = event_accumulator.EventAccumulator(
        logfile, size_guidance={event_accumulator.TENSORS: 2})
    event.Reload()
    try:
        training_time = tf.make_ndarray(
            event.Tensors("time_taken")[0].tensor_proto)
        adversary_score = tf.make_ndarray(
            event.Tensors("eve_score")[0].tensor_proto)
    except KeyError:
        training_time = -1
        adversary_score = -1
    return training_time, adversary_score


def get_run_data(logfile):
    return get_hparams(logfile), get_metrics(logfile)


paths = glob.glob("../../../**/**/**/events.out.tfevents.*")
with open('rawdata.csv', 'w', newline='') as csvfile:
    headers = ['update_rule', 'K', 'N', 'L', 'attack',
               'training_time', 'adversary_score']
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()
    for path in paths:
        hparams, metrics = get_run_data(path)
        training_time, adversary_score = metrics
        # print(hparams, metrics)
        if ((hparams is None)
                or (training_time is -1 and adversary_score is -1)):
            continue
        K, N, L, update_rule, attack = hparams
        writer.writerow({'update_rule': update_rule,
                         'K': K,
                         'N': N,
                         'L': L,
                         'attack': attack,
                         'training_time': training_time,
                         'adversary_score': adversary_score
                         })


data = pd.read_csv("rawdata.csv")
processed = data.groupby(['update_rule', 'K', 'N', 'L', 'attack']).mean()
processed.to_csv('../averaged/averaged-with-attack.csv')
