# -*- coding: utf-8 -*-
import mlencrypt_research.update_rules.basic
import mlencrypt_research.update_rules.probabilistic

import hashlib
from os import environ

import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import use as matplot_backend
matplot_backend('agg')
import matplotlib.pyplot as plt  # noqa

sns.set()


def tb_summary(name, data):
    with tf.name_scope(name):
        data_float = tf.cast(data, tf.float16)
        with tf.name_scope('summaries'):
            tf.summary.scalar('mean', tf.math.reduce_mean(data))
            tf.summary.scalar('stddev', tf.math.reduce_std(data_float))
            tf.summary.scalar('max', tf.math.reduce_max(data))
            tf.summary.scalar('min', tf.math.reduce_min(data))
            tf.summary.scalar('softmax', tf.math.reduce_logsumexp(data_float))
        tf.summary.histogram('histogram', data)


def create_heatmap(name, data_range, ticks, boundaries, data, xaxis, yaxis):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap(lut=int(data_range.item()))
    sns.heatmap(
        pd.DataFrame(data=data, index=yaxis, columns=xaxis),
        ax=ax,
        cmap=cmap,
        cbar_kws={"ticks": ticks, "boundaries": boundaries}
    )
    ax.set(xlabel="input perceptron", ylabel="hidden perceptron")

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
        tf.summary.image('heatmap', tf.expand_dims(pixels, 0))
    return scope


def create_boxplot(name, data, xaxis):
    fig, ax = plt.subplots()
    df = pd.DataFrame(data=data, index=xaxis).transpose()
    sns.boxplot(data=df, ax=ax)
    sns.swarmplot(data=df, size=2, color=".3", linewidth=0)
    ax.xaxis.grid(True)
    ax.set(xlabel="hidden perceptron", ylabel=name)
    sns.despine(fig=fig, ax=ax, trim=True, left=True)

    fig.canvas.draw()
    # w, h = fig.get_size_inches() * fig.get_dpi()
    w, h = fig.canvas.get_width_height()
    pixels = np.frombuffer(fig.canvas.tostring_rgb(),
                           dtype=np.uint8).reshape(h, w, 3)
    plt.close()
    return pixels


def tb_boxplot(name, data, xaxis, unique=True, scope=None):
    scope_name = f"{name}/" if (not name.endswith('/') and unique) else name
    with tf.name_scope(scope if scope else scope_name) as scope:
        inp = [name, data, xaxis]
        # tf.numpy_function: n=50, x-bar=155.974089 s, sigma=36.414627 s
        # tf.py_function:    n=50, x-bar=176.504593 s, sigma=47.747290 s
        # With Welch's t-test, we had a p-value of 0.00880203; we have
        # sufficient evidence to conclude that tf.numpy_function is
        # significantly faster than tf.py_function in this use-case.
        # Note that at the time this script was benchmarked for the above
        # results, image summaries for weights were not logged and the script
        # was not run with XLA.
        pixels = tf.numpy_function(create_boxplot, inp, tf.uint8)
        tf.summary.image('boxplot', tf.expand_dims(pixels, 0))
    return scope


class TPM(tf.Module):
    def __init__(self, name, K, N, L, initial_weights):
        """
        Args:
            K (int): The number of hidden perceptrons.
            N (int): The number of input perceptrons that each hidden
                perceptron has.
            L (int): The synaptic depth of each input perceptron's weights.
        """
        super(TPM, self).__init__(name=name)
        self.type = 'basic'
        self.sigma = tf.Variable(
            tf.zeros([K], dtype=tf.int32),
            trainable=False,
            name='sigma'
        )
        with self.name_scope:
            self.K = tf.guarantee_const(tf.constant(K, name='K'))
            self.N = tf.guarantee_const(tf.constant(N, name='N'))
            self.L = tf.guarantee_const(tf.constant(L, name='L'))
            self.w = initial_weights
            self.key = tf.constant("", name='key')
            self.iv = tf.constant("", name='iv')

    def compute_sigma(self, X):
        """
        Args:
            X: A random vector which is the input for TPM.
        Returns:
            A tuple of the vector of the outputs of each hidden perceptron and
            the vector with all 0s replaced with -1s. For example:

            ([-1, 0, 1, 0, -1, 1], [-1, -1, 1, -1, -1, 1])

            Each vector has dimension [K].
        """
        original = tf.math.sign(tf.math.reduce_sum(
            tf.math.multiply(X, self.w), axis=1))
        id = self.name[0]
        nonzero = tf.where(
            tf.math.equal(original, 0, name=f'{id}-sigma-zero'),
            -1,
            original,
            name='sigma-no-zeroes'
        )
        return original, nonzero

    def get_output(self, X):
        """
        Args:
            X: A random vector which is the input for TPM.

        Returns:
            A binary digit tau for a given random vecor.
        """

        tf.reshape(X, [self.K, self.N])

        # compute inner activation sigma, [K]
        sigma, nonzero = self.compute_sigma(X)
        # compute output of TPM, binary scalar
        tau = tf.cast(tf.math.sign(tf.math.reduce_prod(nonzero)), tf.int32)

        with self.name_scope:
            self.X = X
            self.sigma.assign(sigma)
            self.tau = tau
            if environ['MLENCRYPT_TB'] == 'TRUE':
                tf.summary.scalar('tau', self.tau)

        return tau

    def __call__(self, X):
        return self.get_output(X)

    @tf.function(
        experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
        experimental_relax_shapes=True,
    )
    def update(self, tau2, update_rule):
        """Updates the weights according to the specified update rule.

        Args:
            tau2 (int): Output bit from the other machine, must be -1 or 1.
            update_rule (str): The update rule, must be 'hebbian',
                'anti_hebbian', or 'random_walk'.
        """
        if tf.math.equal(self.tau, tau2):
            if update_rule == "hebbian":
                mlencrypt_research.update_rules.basic.hebbian(
                    self.w,
                    self.X,
                    self.sigma,
                    self.tau,
                    tau2,
                    self.L
                )
            elif update_rule == 'anti_hebbian':
                mlencrypt_research.update_rules.basic.anti_hebbian(
                    self.w,
                    self.X,
                    self.sigma,
                    self.tau,
                    tau2,
                    self.L
                )
            elif update_rule == 'random_walk':
                mlencrypt_research.update_rules.basic.random_walk(
                    self.w,
                    self.X,
                    self.sigma,
                    self.tau,
                    tau2,
                    self.L
                )
            else:
                if isinstance(update_rule, tf.Tensor):
                    # TF AutoGraph is tracing, so don't raise a ValueError
                    pass
                else:
                    raise ValueError(
                        f"'{update_rule}' is an invalid update rule. "
                        "Valid update rules are: "
                        "'hebbian', "
                        "'anti_hebbian' and "
                        "'random_walk'."
                    )
            if environ["MLENCRYPT_TB"] == 'TRUE':
                with tf.name_scope(self.name):
                    try:
                        with tf.experimental.async_scope():
                            tb_summary('sigma', self.sigma)
                            tf.summary.histogram('weights', self.w)

                            hpaxis = tf.range(1, self.K + 1)
                            ipaxis = tf.range(1, self.N + 1)
                            # weights_scope_name is a temporary fix to prevent
                            # the scope becoming 'weights_1'
                            weights_scope_name = f'{self.name}/weights/'
                            weights_scope = tb_heatmap(
                                weights_scope_name,
                                self.w,
                                ipaxis,
                                hpaxis,
                                unique=False
                            )
                            tb_boxplot(
                                weights_scope_name,
                                self.w,
                                hpaxis,
                                unique=False,
                                scope=weights_scope
                            )

                            def log_weights_hperceptron(K, weights):
                                for i in range(K):
                                    # hperceptron weights aren't logged, see
                                    # https://github.com/tensorflow/tensorflow/issues/38772
                                    with tf.name_scope(f'hperceptron{i + 1}'):
                                        tb_summary('weights', weights[i])
                            tf.numpy_function(
                                log_weights_hperceptron,
                                [self.K, self.w],
                                [],
                                name='tb-images-weights'
                            )
                    except tf.errors.OutOfRangeError:
                        tf.experimental.async_clear_error()
            return True
        else:
            return False

    @tf.function(
        experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
        experimental_relax_shapes=True,
    )
    def compute_key(self, key_length, iv_length):
        """Creates a key and IV based on the weights of this TPM.

        Args:
            key_length (int): Length of the key.
                Must be 128, 192, or 256.
            iv_length (int): Length of the independent variable.
                Must be a multiple of 4 between 0 and 256, inclusive.
        Returns:
            The key and IV based on the TPM's weights.
        """
        main_diagonal = tf.guarantee_const(
            tf.range(tf.math.minimum(self.K, self.N))
        )
        iv_indices = tf.guarantee_const(
            tf.stack([main_diagonal, main_diagonal], axis=1)
        )
        iv_weights = tf.strings.format("{}", tf.gather_nd(self.w, iv_indices))
        key_weights = tf.strings.format("{}", self.w)

        def convert_to_hex_dig(input, length):
            return hashlib.sha512(
                input.numpy().decode('utf-8').encode('utf-8')
            ).hexdigest()[0:length]

        # TODO: figure out a way to do this without using py_function
        # py_function is currently needed since we need to get the value from
        # the tf.Tensor
        current_key = tf.py_function(
            convert_to_hex_dig,
            [key_weights, int(key_length / 4)],
            Tout=tf.string
        )
        current_iv = tf.py_function(
            convert_to_hex_dig,
            [iv_weights, int(iv_length / 4)],
            Tout=tf.string
        )
        # if not hasattr(self, 'key'):
        #     self.key = tf.Variable('', trainable=False, name='key')
        #     # try:
        #     #     self.key = tf.Variable('', trainable=False, name='key')
        #     # except ValueError:
        #     #     self.key = tf.constant("", name='key')
        # if not hasattr(self, 'iv'):
        #     self.iv = tf.Variable('', trainable=False, name='iv')
        #     # try:
        #     #     self.iv = tf.Variable('', trainable=False, name='iv')
        #     # except ValueError:
        #     #     self.iv = tf.constant("", name='iv')
        self.key.assign(current_key)
        self.iv.assign(current_iv)
        with self.name_scope:
            tf.summary.text('key', data=current_key)
            tf.summary.text('independent variable', data=current_iv)
        return current_key, current_iv


class ProbabilisticTPM(TPM):
    """
    Attributes:
        W: [K, N, 2L+1] matrix representing the PDF of the weight distribution.
        mu_W: [K, N] matrix of the averages of the weight distributions.
        sigma_W: [K, N] matrix of the standard deviations of the weight
            distributions.
    """

    def __init__(self, name, K, N, L, initial_weights):
        super().__init__(name, K, N, L, initial_weights)
        self.type = 'probabilistic'
        self.w = tf.Variable(
            tf.fill([K, N, 2 * L + 1],
                    tf.constant(1. / (2 * L + 1), tf.float16)),
            trainable=True
        )
        self.mpW = tf.Variable(
            tf.zeros([K, N], dtype=tf.int32),
            # trainable=True
        )
        self.eW = tf.Variable(
            tf.zeros([K, N], dtype=tf.float16),
            # trainable=True
        )
        self.sigma = tf.Variable(
            tf.zeros([K], dtype=tf.float16),
            trainable=False,
            name='sigma'
        )

    def get_expected_weights(self):
        def get_expected_weight(dpdf):
            # tf.size(dpdf) should be 2*L+1
            eW_ij = tf.math.reduce_mean(tf.map_fn(
                # dpdf[x] gives probability of corresponding weight
                # self.index_to_weight(x) gives corresponding weight
                lambda x: dpdf[x] * tf.cast(self.index_to_weight(x),
                                            tf.float16),
                tf.range(tf.size(dpdf)),  # indices
                dtype=tf.float16
            ))
            return eW_ij
        self.eW.assign(tf.reshape(
            tf.map_fn(
                get_expected_weight,
                tf.reshape(self.w, (self.K * self.N, 2 * self.L + 1)),
            ),
            (self.K, self.N)
        ))
        return self.eW

    def compute_sigma(self, X):
        """
        Args:
            X: A random vector which is the input for TPM.
        Returns:
            A tuple of the vector of the outputs of each hidden perceptron and
            the vector with all 0s replaced with -1s. For example:

            ([-1, 0, 1, 0, -1, 1], [-1, -1, 1, -1, -1, 1])

            Each vector has dimension [K].
        """
        original = tf.math.sign(tf.math.reduce_sum(
            tf.math.multiply(tf.cast(X, tf.float16), self.eW), axis=1))
        id = self.name[0]
        nonzero = tf.where(
            tf.math.equal(original, 0., name=f'{id}-sigma-zero'),
            tf.cast(-1., tf.float16, name='negative-1'),
            original,
            name='sigma-no-zeroes'
        )
        return original, nonzero

    def normalize_weights(self, i=-1, j=-1):
        """Normalizes probability distributions.

        Normalizes the probability distribution associated with W[i, j]. If
        negative indeces i, j are provided, the normalization is carried out
        for the entire probability distributions.

        Args:
            i (int): Index of the hidden perceptron distribution to normalize.
            j (int): Index of the input perceptron distribution to normalize.
        """
        # TODO: try tf.vectorized_map
        if (j < 0 and i < 0):
            self.w.assign(tf.map_fn(tf.math.reduce_mean, self.w))
        else:
            self.w[i, j].assign(
                tf.map_fn(tf.math.reduce_mean, self.w[i, j]))

    def index_to_weight(self, index):
        # indices:    0,   1,   2,   3,   4
        # L:        [-2,  -1,   0,   1,   2]
        # pWeights: [.2,  .2,  .2,  .2,  .2]
        return tf.cast(index, tf.int32) - self.L

    def get_most_probable_weight(self):
        """
        Returns:
            [K, N] matrix with each cell representing the weight which has
            the largest probability of existing in the defender's TPM.
        """
        mpW_rows = tf.TensorArray(tf.int32, size=self.K)
        for i in tf.range(self.K):
            mpW_cols = tf.TensorArray(tf.int32, size=self.N)
            for j in tf.range(self.N):
                mpW_cols = mpW_cols.write(
                    j, self.index_to_weight(tf.argmax(self.w[i, j])))
            mpW_rows = mpW_rows.write(i, mpW_cols.stack())
        self.mpW.assign(mpW_rows.stack())
        return self.mpW

    @tf.function(
        experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
        experimental_relax_shapes=True,
    )
    def update(self, tau2, updated_A_B):
        # https://pdfs.semanticscholar.org/a4d1/66b13f6297438cb95f71c0445bee5743a2f2.pdf#page=55
        if updated_A_B:
            mlencrypt_research.update_rules.probabilistic.hebbian()
        mlencrypt_research.update_rules.probabilistic.monte_carlo()

        self.get_expected_weights()
        self.get_most_probable_weight()

    @tf.function(
        experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
        experimental_relax_shapes=True,
    )
    def compute_key(self, key_length, iv_length):
        """Creates a key and IV based on the weights of this TPM.

        Args:
            key_length (int): Length of the key.
                Must be 128, 192, or 256.
            iv_length (int): Length of the independent variable.
                Must be a multiple of 4 between 0 and 256, inclusive.
        Returns:
            The key and IV based on the TPM's weights.
        """

        main_diagonal = tf.guarantee_const(
            tf.range(tf.math.minimum(self.K, self.N))
        )
        iv_indices = tf.guarantee_const(
            tf.stack([main_diagonal, main_diagonal], axis=1)
        )
        iv_weights = tf.strings.format(
            "{}", tf.gather_nd(self.mpW, iv_indices))
        key_weights = tf.strings.format("{}", self.mpW)

        def convert_to_hex_dig(input, length):
            return hashlib.sha512(
                input.numpy().decode('utf-8').encode('utf-8')
            ).hexdigest()[0:length]

        # TODO: figure out a way to do this without using py_function
        # py_function is currently needed since we need to get the value from
        # the tf.Tensor
        current_key = tf.py_function(
            convert_to_hex_dig,
            [key_weights, int(key_length / 4)],
            Tout=tf.string
        )
        current_iv = tf.py_function(
            convert_to_hex_dig,
            [iv_weights, int(iv_length / 4)],
            Tout=tf.string
        )
        # if not hasattr(self, 'key'):
        #     self.key = tf.Variable('', trainable=False, name='key')
        #     # try:
        #     #     self.key = tf.Variable('', trainable=False, name='key')
        #     # except ValueError:
        #     #     self.key = tf.constant("", name='key')
        # if not hasattr(self, 'iv'):
        #     self.iv = tf.Variable('', trainable=False, name='iv')
        #     # try:
        #     #     self.iv = tf.Variable('', trainable=False, name='iv')
        #     # except ValueError:
        #     #     self.iv = tf.constant("", name='iv')
        self.key.assign(current_key)
        self.iv.assign(current_iv)
        with self.name_scope:
            tf.summary.text('key', data=current_key)
            tf.summary.text('independent variable', data=current_iv)
        return current_key, current_iv


class GeometricTPM(TPM):
    # TODO: it might just be autograph but geometric isn't working,
    # running an hparams with 1 trial returns the same results for both attacks
    def __init__(self, name, K, N, L, initial_weights):
        super().__init__(name, K, N, L, initial_weights)
        self.type = 'geometric'

    def update_sigma(self):
        """Updates sigma using the geometric algorithm.

        Negates the sigma value of the hidden perceptron with the lowest
        current state.
        """
        wx = tf.math.reduce_sum(tf.math.multiply(self.X, self.w), axis=1)
        h_i = tf.math.divide(tf.cast(wx, tf.float16),
                             tf.math.sqrt(tf.cast(self.N, tf.float16)))
        min = tf.math.argmin(tf.math.abs(h_i))  # index of min of |h|
        nonzero = tf.where(
            tf.math.equal(self.sigma, 0, name=f'{self.name[0]}-sigma-zero'),
            -1,
            self.sigma,
            name='sigma-no-zeroes'
        )
        self.sigma[min].assign(tf.math.negative(nonzero[min]))
        self.tau = tf.cast(tf.math.sign(
            tf.math.reduce_prod(self.sigma)), tf.int32)

    def update(self, tau2, update_rule):
        """Updates the weights according to the specified update rule.

        Args:
            tau2 (int): Output bit from the other machine, must be -1 or 1.
            update_rule (str): The update rule, must be 'hebbian',
                'anti_hebbian', or 'random_walk'.
        """
        if super().update(tau2, update_rule):
            return True
        else:
            self.update_sigma()
            super().update(tau2, update_rule)
            return False
