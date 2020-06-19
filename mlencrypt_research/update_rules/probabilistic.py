import tensorflow as tf
import tensorflow_probability as tfp
from os import getenv


def analytical():
    # https://pdfs.semanticscholar.org/65f8/1f6d9d4ec3eae0d32a3bbb1c1e8593ee8fa7.pdf#page=3

    # maybe just run monte carlo but instead of sampling, run on every
    # permutation? this way it's deterministic
    pass


def mean_field():
    # https://pdfs.semanticscholar.org/65f8/1f6d9d4ec3eae0d32a3bbb1c1e8593ee8fa7.pdf#page=3
    pass


def monte_carlo(W, X, l, tau2, num_samples=10.):
    # https://github.com/tensorflow/probability/issues/976#issuecomment-646689221
    # https://stackoverflow.com/a/25402993/7127932

    # num_samples is number of inner rounds in this paper:
    # https://pdfs.semanticscholar.org/65f8/1f6d9d4ec3eae0d32a3bbb1c1e8593ee8fa7.pdf#page=3
    # https://pdfs.semanticscholar.org/a4d1/66b13f6297438cb95f71c0445bee5743a2f2.pdf#page=55

    # sample from probability distributions
    # set probabilistic weights to a posteriori result of this sampling
    # (like Bayes rule)
    k, n, _ = W.probs.shape
    num_valid_samples = 0.
    posterior_weights = tf.zeros([k, n, 2*l+1], tf.float32)
    for sample in W.sample(sample_shape=num_samples):
        current_sample_rows = tf.TensorArray(tf.int32, size=k)
        for index_k in tf.range(k):
            current_sample_cols = tf.TensorArray(tf.int32, size=n)
            for index_n in tf.range(n):
                current_sample_cols = current_sample_cols.write(
                    index_n,
                    tf.cast(tf.where(sample[index_k][index_n] == 1)[0][0], tf.int32) - l,
                )
            current_sample_rows = current_sample_rows.write(index_k, current_sample_cols.stack())
        current_sample = current_sample_rows.stack()  # sample from weights
        # compute inner activation sigma, [K]
        original = tf.math.sign(tf.math.reduce_sum(
            tf.math.multiply(X, current_sample), axis=1))
        nonzero = tf.where(
            tf.math.equal(original, 0),
            -1,
            original,
            name='sigma-no-zeroes'
        )
        # tau is the output of the TPM, and is a binary scalar
        # tau is float16 for ProbabilisticTPM
        tau = tf.cast(tf.math.sign(tf.math.reduce_prod(nonzero)), tf.int32)
        if tf.math.equal(tau, tau2):
            num_valid_samples += 1.
            posterior_weights += tf.cast(sample, tf.float32)
    posterior_weights /= num_valid_samples
    W = W.copy(probs=posterior_weights)
    # tf.print(posterior_weights)
    # should be equal to tf.fill([k, n], 1.):
    # tf.print(tf.math.reduce_sum(W.probs, axis=2))


def hebbian():
    # https://pdfs.semanticscholar.org/a4d1/66b13f6297438cb95f71c0445bee5743a2f2.pdf#page=56
    pass
