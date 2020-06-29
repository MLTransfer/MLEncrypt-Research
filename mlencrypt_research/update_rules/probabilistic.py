import tensorflow as tf


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
    k, n = W.shape
    num_valid_samples = 0.
    posterior_weights = tf.zeros([k, n, 2*l+1], tf.float32)
    for _ in tf.range(num_samples):
        current_sample = W.sample()
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
            posterior_weights += tf.cast(current_sample, tf.float32)
    posterior_weights /= num_valid_samples


def hebbian():
    # https://pdfs.semanticscholar.org/a4d1/66b13f6297438cb95f71c0445bee5743a2f2.pdf#page=56
    pass
