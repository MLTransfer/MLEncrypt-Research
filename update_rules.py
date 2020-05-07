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
    W_plus_rows = tf.TensorArray(W.dtype, size=k)
    for i in tf.range(k):
        W_plus_cols = tf.TensorArray(W.dtype, size=n)
        for j in tf.range(n):
            W_plus_cols = W_plus_cols.write(
                j,
                X[i, j]
                * tau1
                * theta(sigma[i], tau1)
                * theta(tau1, tau2)
            )
        W_plus_rows = W_plus_rows.write(i, W_plus_cols.stack())
    W.assign_add(W_plus_rows.stack())
    W.assign(tf.clip_by_value(
        W,
        clip_value_min=tf.cast(-l, tf.int64),
        clip_value_max=tf.cast(l, tf.int64),
    ))


def anti_hebbian(W, X, sigma, tau1, tau2, l):
    k, n = W.shape
    W_plus_rows = tf.TensorArray(W.dtype, size=k)
    for i in tf.range(k):
        W_plus_cols = tf.TensorArray(W.dtype, size=n)
        for j in tf.range(n):
            W_plus_cols = W_plus_cols.write(
                j,
                X[i, j]
                * tau1
                * theta(sigma[i], tau1)
                * theta(tau1, tau2)
            )
        W_plus_rows = W_plus_rows.write(i, W_plus_cols.stack())
    W.assign_sub(W_plus_rows.stack())
    W.assign(tf.clip_by_value(
        W,
        clip_value_min=tf.cast(-l, tf.int64),
        clip_value_max=tf.cast(l, tf.int64),
    ))


def random_walk(W, X, sigma, tau1, tau2, l):
    k, n = W.shape
    W_plus_rows = tf.TensorArray(W.dtype, size=k)
    for i in tf.range(k):
        W_plus_cols = tf.TensorArray(W.dtype, size=n)
        for j in tf.range(n):
            W_plus_cols = W_plus_cols.write(
                j,
                X[i, j]
                * theta(sigma[i], tau1)
                * theta(tau1, tau2)
            )
        W_plus_rows = W_plus_rows.write(i, W_plus_cols.stack())
    W.assign_add(W_plus_rows.stack())
    W.assign(tf.clip_by_value(
        W,
        clip_value_min=tf.cast(-l, tf.int64),
        clip_value_max=tf.cast(l, tf.int64),
    ))
