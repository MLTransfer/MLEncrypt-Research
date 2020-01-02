# DOESN'T WORK, see https://github.com/tensorflow/tensorboard/issues/3091
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
si = summary_iterator(
    "snowy/snowy1/events.out.tfevents.1577067052.snowy.1378.5.v2")
count = 0
for e in si:
    count += 1
    print(str(count) + ': ' + str(e.summary.value))
