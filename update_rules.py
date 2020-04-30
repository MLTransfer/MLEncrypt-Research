# -*- coding: utf-8 -*-
import tensorflow as tf


def theta(t1, t2):
    """
    Args:
        t1 (int): First value to compare.
        t2 (int): Second value to compare.

    Returns:
        int: 1 if t1 and t2 are equal, 0 otherwise.
    """
    # tf.cast(tf.math.equal(t1, t2), tf.int64)
    # tf.where(tf.math.equal(t1, t2), t1, t2)
    #
    return tf.where(tf.math.equal(t1, t2), t1, t2)


def hebbian(W, X, sigma, tau1, tau2, l):
    k, n = W.shape
    W_plus = tf.TensorArray(W.dtype, size=k * n)
    for i in tf.range(k):
        for j in tf.range(n):
            W_plus = W_plus.write(
                i * n + j,
                X[i, j]
                * tau1
                * theta(sigma[i], tau1)
                * theta(tau1, tau2)
            )
    W.assign_add(tf.reshape(W_plus.stack(), W.shape))
    W.assign(tf.clip_by_value(
        W,
        clip_value_min=tf.cast(-l, tf.int64),
        clip_value_max=tf.cast(l, tf.int64),
    ))


def anti_hebbian(W, X, sigma, tau1, tau2, l):
    k, n = W.shape
    W_plus = tf.TensorArray(W.dtype, size=k * n)
    for i in tf.range(k):
        for j in tf.range(n):
            W_plus = W_plus.write(
                i * n + j,
                X[i, j]
                * tau1
                * theta(sigma[i], tau1)
                * theta(tau1, tau2)
            )
    W.assign_sub(tf.reshape(W_plus.stack(), W.shape))
    W.assign(tf.clip_by_value(
        W,
        clip_value_min=tf.cast(-l, tf.int64),
        clip_value_max=tf.cast(l, tf.int64),
    ))


def random_walk(W, X, sigma, tau1, tau2, l):
    k, n = W.shape
    W_plus = tf.TensorArray(W.dtype, size=k * n)
    for i in tf.range(k):
        for j in tf.range(n):
            W_plus = W_plus.write(
                i * n + j,
                X[i, j]
                * theta(sigma[i], tau1)
                * theta(tau1, tau2)
            )
    W.assign_add(tf.reshape(W_plus.stack(), W.shape))
    W.assign(tf.clip_by_value(
        W,
        clip_value_min=tf.cast(-l, tf.int64),
        clip_value_max=tf.cast(l, tf.int64),
    ))
