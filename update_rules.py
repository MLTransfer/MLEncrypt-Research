# Update rules for tree parity machine
import tensorflow as tf


def theta(t1, t2):
    return tf.where(tf.math.equal(t1, t2), t1, t2)


def hebbian(W, X, sigma, tau1, tau2, l):
    k, n = W.shape
    for i in tf.range(k):
        for j in tf.range(n):
            W[i, j].assign(tf.clip_by_value(W[i, j] + X[i, j] * tau1
                                            * theta(sigma[i], tau1)
                                            * theta(tau1, tau2),
                                            clip_value_min=tf.cast(-l,
                                                                   tf.int64),
                                            clip_value_max=tf.cast(l,
                                                                   tf.int64)
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
