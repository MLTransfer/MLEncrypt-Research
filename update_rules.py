# -*- coding: utf-8 -*-
# Update rules for tree parity machine
import tensorflow as tf


def theta(t1, t2):
    """
    Args:
        t1: First value to compare.
        t2: Second value to compare.

    Returns:
        1 if t1 and t2 are equal, 0 otherwise.
    """
    return tf.cast(tf.math.equal(t1, t2), tf.int64)


def hebbian(W, X, sigma, tau1, tau2, l):
    k, n = W.shape
    for i in tf.range(k):
        for j in tf.range(n):
            W[i, j].assign(tf.clip_by_value(W[i, j] + X[i, j] * tau1
                                            * theta(sigma[i], tau1)
                                            * theta(tau1, tau2),
                                            clip_value_min=tf.cast(-l,
                                                                   tf.int64),
                                            clip_value_max=tf.cast(l, tf.int64)
                                            )
                           )


def anti_hebbian(W, X, sigma, tau1, tau2, l):
    k, n = W.shape
    for i in tf.range(k):
        for j in tf.range(n):
            W[i, j].assign(tf.clip_by_value(W[i, j] - X[i, j] * tau1
                                            * theta(sigma[i], tau1)
                                            * theta(tau1, tau2),
                                            clip_value_min=tf.cast(-l,
                                                                   tf.int64),
                                            clip_value_max=tf.cast(l, tf.int64)
                                            )
                           )


def random_walk(W, X, sigma, tau1, tau2, l):
    k, n = W.shape
    for i in tf.range(k):
        for j in tf.range(n):
            W[i, j].assign(tf.clip_by_value(W[i, j] + X[i, j]
                                            * theta(sigma[i], tau1)
                                            * theta(tau1, tau2),
                                            clip_value_min=tf.cast(-l,
                                                                   tf.int64),
                                            clip_value_max=tf.cast(l, tf.int64)
                                            )
                           )
