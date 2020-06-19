import tensorflow as tf
import seaborn as sns
import numpy as np
from matplotlib import use as matplot_backend
matplot_backend('agg')
import matplotlib.pyplot as plt  # noqa

sns.set()


def create_heatmap(name, data_range, ticks, boundaries, data, xaxis, yaxis):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap(lut=int(data_range.item()))
    sns.heatmap(
        data,
        xticklabels=xaxis,
        yticklabels=yaxis,
        ax=ax,
        cmap=cmap,
        cbar_kws={"ticks": ticks, "boundaries": boundaries}
    )
    ax.set(xlabel="input neuron", ylabel="hidden neuron")

    fig.canvas.draw()
    # w, h = fig.get_size_inches() * fig.get_dpi()
    w, h = fig.canvas.get_width_height()
    pixels = np.frombuffer(fig.canvas.tostring_rgb(),
                           dtype=np.uint8).reshape(h, w, 3)
    plt.close()
    return pixels


def tb_heatmap(name, data, xaxis, yaxis, unique=True, scope=None):
    scope_name = f"{name}/" if (not name.endswith('/') and unique) else name
    with tf.name_scope(scope if scope else scope_name) as scope:
        data_float = tf.cast(data, tf.float32)
        min = tf.math.reduce_min(data_float)
        max = tf.math.reduce_max(data_float)
        data_range = max - min + 1
        ticks = tf.range(min, max + 1)
        boundaries = tf.range(min - .5, max + 1.5)
        inp = [name, data_range, ticks, boundaries, data, xaxis, yaxis]
        # tf.numpy_function: n=50, x-bar=155.974089 s, sigma=36.414627 s
        # tf.py_function:    n=50, x-bar=176.504593 s, sigma=47.747290 s
        # With Welch's t-test, we had a p-value of 0.00880203; we have
        # sufficient evidence to conclude that tf.numpy_function is
        # significantly faster than tf.py_function in this use-case.
        # Note that at the time this script was benchmarked for the above
        # results, image summaries for weights were not logged and the script
        # was not run with XLA.
        pixels = tf.numpy_function(create_heatmap, inp, tf.uint8)
        try:
            tf.summary.image('heatmap', tf.expand_dims(pixels, 0))
        except TypeError:
            # Expected string, got
            # <tf.Tensor 'EmptyTensorList:0' shape=() dtype=variant> of type
            # 'Tensor' instead.
            pass
    return scope
