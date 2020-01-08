import tensorflow as tf  # 2.1.0
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorboard.plugins.hparams import plugin_data_pb2
from tensorboard.backend.event_processing import event_accumulator
import glob
import csv


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
        time_taken = tf.make_ndarray(
            event.Tensors("time_taken")[0].tensor_proto)
        eve_score = tf.make_ndarray(event.Tensors("eve_score")[0].tensor_proto)
    except KeyError:
        time_taken = -1
        eve_score = -1
    return time_taken, eve_score


def get_run_data(logfile):
    return get_hparams(logfile), get_metrics(logfile)
    # return update_rule, K, N, L, attack, time_taken, eve_score


paths = glob.glob("../../../**/**/**/events.out.tfevents.*")
with open('rawdata.csv', 'w', newline='') as csvfile:
    headers = ['update_rule', 'K', 'N', 'L', 'attack',
               'time_taken', 'eve_score']
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()
    for path in paths:
        hparams, metrics = get_run_data(path)
        time_taken, eve_score = metrics
        if (hparams is None) or (time_taken is -1 and eve_score is -1):
            continue
        K, N, L, update_rule, attack = hparams
        writer.writerow({'update_rule': update_rule,
                         'K': K,
                         'N': N,
                         'L': L,
                         'attack': attack,
                         'time_taken': time_taken,
                         'eve_score': eve_score
                         })


# def to_decimal(d):
#     return Decimal(d)
#
#
# data = pd.read_csv("rawdata.csv", converters={
#                    'time_taken': to_decimal, 'eve_score': to_decimal})
# groupby = data.groupby(['update_rule', 'K', 'N', 'L', 'attack'])
# processed = groupby.apply(
#     lambda x: x.sum()) / groupby.size()
# del groupby
# processed.to_csv('processed.csv')
