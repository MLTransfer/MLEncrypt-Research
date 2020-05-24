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


def indices_from_2d(index_2d, k):
    return tf.math.floormod(index_2d, k), tf.math.floordiv(index_2d, k)


def hebbian(W, X, sigma, tau1, tau2, l):
    k, n = W.shape
    # TODO: benchmark tf.size(W) vs k*n
    indices_2d = tf.range(tf.size(W))

    def update_perceptron(index_2d):
        i, j = indices_from_2d(index_2d, k)
        # assume that anti_hebbian is only called if tau1 equals tau2, so don't
        # multiply by theta(tau1, tau2):
        return X[i, j] * tau1 * theta(sigma[i], tau1)
    W_plus_vectorized = tf.map_fn(update_perceptron, indices_2d)
    W_plus = tf.reshape(W_plus_vectorized, (k, n))
    W.assign_add(W_plus)
    W.assign(tf.clip_by_value(
        W,
        clip_value_min=tf.cast(-l, tf.int32),
        clip_value_max=tf.cast(l, tf.int32),
    ))


def anti_hebbian(W, X, sigma, tau1, tau2, l):
    k, n = W.shape
    # TODO: benchmark tf.size(W) vs k*n
    indices_2d = tf.range(tf.size(W))

    def update_perceptron(index_2d):
        i, j = indices_from_2d(index_2d, k)
        # assume that anti_hebbian is only called if tau1 equals tau2, so don't
        # multiply by theta(tau1, tau2):
        return X[i, j] * tau1 * theta(sigma[i], tau1)
    W_plus_vectorized = tf.map_fn(update_perceptron, indices_2d)
    W_plus = tf.reshape(W_plus_vectorized, (k, n))
    W.assign_sub(W_plus)
    W.assign(tf.clip_by_value(
        W,
        clip_value_min=tf.cast(-l, tf.int32),
        clip_value_max=tf.cast(l, tf.int32),
    ))


def random_walk(W, X, sigma, tau1, tau2, l):
    k, n = W.shape
    # TODO: benchmark tf.size(W) vs k*n
    indices_2d = tf.range(tf.size(W))

    def update_perceptron(index_2d):
        i, j = indices_from_2d(index_2d, k)
        # assume that anti_hebbian is only called if tau1 equals tau2, so don't
        # multiply by theta(tau1, tau2):
        return X[i, j] * theta(sigma[i], tau1)
    W_plus_vectorized = tf.map_fn(update_perceptron, indices_2d)
    W_plus = tf.reshape(W_plus_vectorized, (k, n))
    W.assign_add(W_plus)
    W.assign(tf.clip_by_value(
        W,
        clip_value_min=tf.cast(-l, tf.int32),
        clip_value_max=tf.cast(l, tf.int32),
    ))
