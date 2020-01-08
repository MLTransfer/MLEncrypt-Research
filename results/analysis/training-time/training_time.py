import glob
import csv
import tensorflow as tf  # 2.0.0
from tensorboard.backend.event_processing import event_accumulator


def get_pair(logfile):
    event = event_accumulator.EventAccumulator(
        logfile, size_guidance={
            event_accumulator.SCALARS: 2,
            event_accumulator.TENSORS: 2
        })
    event.Reload()
    try:
        steps = event.Tensors("time_taken")[0][1]
    except KeyError:
        return -1, -1
    training_time = tf.make_ndarray(
        event.Tensors("time_taken")[0].tensor_proto)
    return steps, training_time


paths = glob.glob("../../**/**/**/events.out.tfevents.*")
with open('training_time.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['steps', 'training_time'])
    writer.writeheader()
    for path in paths:
        steps, training_time = get_pair(path)
        if steps is -1 and training_time is -1:
            continue
        writer.writerow({'steps': steps, 'training_time': training_time})
