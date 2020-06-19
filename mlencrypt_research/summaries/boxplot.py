import tensorflow as tf
import seaborn as sns
import numpy as np
from matplotlib import use as matplot_backend
matplot_backend('agg')
import matplotlib.pyplot as plt  # noqa

sns.set()


def create_boxplot(ylabel, data, xaxis):
    fig, ax = plt.subplots()
    # TODO: x-axis is shifted to the right by 1 unit
    ax = sns.boxplot(data=data, ax=ax)
    ax = sns.swarmplot(data=data, size=2, color=".3", linewidth=0, ax=ax)
    ax.xaxis.grid(True)
    ax.set(xlabel="hidden neuron", ylabel=ylabel.decode("utf-8"))
    ax.set_xticks(xaxis)
    # ax = sns.despine(fig=fig, ax=ax, trim=True, left=True)

    fig.canvas.draw()
    # w, h = fig.get_size_inches() * fig.get_dpi()
    w, h = fig.canvas.get_width_height()
    pixels = np.frombuffer(fig.canvas.tostring_rgb(),
                           dtype=np.uint8).reshape(h, w, 3)
    plt.close()
    return pixels


def tb_boxplot(scope_name, data, xaxis, unique=True, scope=None, ylabel=None):
    # if ylabel is None, then use scope_name, else use the given ylabel:
    ylabel = scope_name if not ylabel else ylabel
    scope_name = f'{scope_name}/' if (not scope_name.endswith('/') and unique) \
        else scope_name
    with tf.name_scope(scope if scope else scope_name) as scope:
        inp = [ylabel, tf.transpose(data), xaxis]
        # tf.numpy_function: n=50, x-bar=155.974089 s, sigma=36.414627 s
        # tf.py_function:    n=50, x-bar=176.504593 s, sigma=47.747290 s
        # With Welch's t-test, we had a p-value of 0.00880203; we have
        # sufficient evidence to conclude that tf.numpy_function is
        # significantly faster than tf.py_function in this use-case.
        # Note that at the time this script was benchmarked for the above
        # results, image summaries for weights were not logged and the script
        # was not run with XLA.
        pixels = tf.numpy_function(create_boxplot, inp, tf.uint8)
        try:
            tf.summary.image('boxplot', tf.expand_dims(pixels, 0))
        except TypeError:
            # Expected string, got
            # <tf.Tensor 'EmptyTensorList:0' shape=() dtype=variant> of type
            # 'Tensor' instead.
            pass
    return scope
