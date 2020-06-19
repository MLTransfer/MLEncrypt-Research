import tensorflow as tf


def tb_summary(name, data):
    with tf.name_scope(name):
        # TODO: don't use bitcast? softmax and stddev are NaN
        data_float = tf.bitcast(data, tf.float16)
        with tf.name_scope('summaries'):
            tf.summary.scalar('mean', tf.math.reduce_mean(data))
            tf.summary.scalar('stddev', tf.math.reduce_std(data_float))
            tf.summary.scalar('max', tf.math.reduce_max(data))
            tf.summary.scalar('min', tf.math.reduce_min(data))
            tf.summary.scalar('softmax', tf.math.reduce_logsumexp(data_float))
        tf.summary.histogram('histogram', data)
