import glob
import csv
import tensorflow as tf  # 2.0.0
from tensorboard.backend.event_processing import event_accumulator


def get_pair(logfile):
    event = event_accumulator.EventAccumulator(
        logfile, size_guidance={event_accumulator.SCALARS: 2, event_accumulator.TENSORS: 2})
    event.Reload()
    # event.Tags()
    try:
        steps = event.Tensors("time_taken")[0][1]
    except KeyError:
        return -1, -1
    time_taken = tf.make_ndarray(event.Tensors("time_taken")[0].tensor_proto)
    return steps, time_taken


paths = glob.glob("**/**/**/events.out.tfevents.*")
# this glob is incorrect, the repository was reorganized
with open('time_taken.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['step', 'time taken'])
    writer.writeheader()
    for path in paths:
        steps, time_taken = get_pair(path)
        if steps is -1 and time_taken is -1:
            continue
        writer.writerow({'steps': steps, 'time taken': time_taken})
