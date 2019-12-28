import hashlib
import tensorflow as tf

from update_rules import hebbian, anti_hebbian, random_walk

from os import remove, environ

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set()


def tb_summary(name, data):
    with tf.name_scope(name):
        with tf.name_scope('summaries'):
            tf.summary.scalar(
                'mean', tf.reduce_mean(data))
            tf.summary.scalar(
                'stddev', tf.math.reduce_std(tf.cast(data, tf.float64)))
            tf.summary.scalar(
                'max', tf.reduce_max(data))
            tf.summary.scalar(
                'min', tf.reduce_min(data))
            tf.summary.scalar(
                'softmax', tf.reduce_logsumexp(tf.cast(data, tf.float64)))
            tf.summary.histogram('histogram', data)


def tb_heatmap(name, data, xaxis, yaxis):
    with tf.name_scope(name):
        _, ax = plt.subplots()
        min, max = tf.math.reduce_min(data), tf.math.reduce_max(data)
        ticks = tf.range(min, max + 1).numpy()
        boundaries = tf.range(tf.math.subtract(
            tf.cast(min, tf.float64), .5), tf.math.add(
                tf.cast(max, tf.float64), 1.5)).numpy()
        cmap = plt.get_cmap(lut=max - min + 1)
        sns.heatmap(pd.DataFrame(data=data.numpy(),
                                 index=xaxis,
                                 columns=yaxis).transpose(),
                    ax=ax,
                    cmap=cmap,
                    cbar_kws={"ticks": ticks, "boundaries": boundaries})
        ax.set(xlabel="hidden perceptron", ylabel="input perceptron")
        png_file = f'{name}-heatmap-{tf.summary.experimental.get_step()}.png'
        plt.savefig(png_file)
        plt.close()
        pixels = tf.io.decode_png(tf.io.read_file(png_file))
        remove(png_file)
        tf.summary.image('heatmap', tf.expand_dims(pixels, 0))


def tb_boxplot(name, data, xaxis):
    with tf.name_scope(name):
        _, ax = plt.subplots()
        df = pd.DataFrame(data=data.numpy(), index=xaxis).transpose()
        sns.boxplot(data=df)
        sns.swarmplot(data=df, size=2, color=".3", linewidth=0)
        ax.xaxis.grid(True)
        ax.set(xlabel="hidden perceptron", ylabel=name)
        sns.despine(trim=True, left=True)
        png_file = f'{name}-boxplot-{tf.summary.experimental.get_step()}.png'
        plt.savefig(png_file)
        plt.close()
        pixels = tf.io.decode_png(tf.io.read_file(png_file))
        remove(png_file)
        tf.summary.image('boxplot', tf.expand_dims(pixels, 0))


class TPM:
    """
    A tree parity machine.
    The machine can be described by the following parameters:
        K: The number of hidden neurons
        N: Then number of input neurons connected to each hidden neuron
        L: Defines the range of each weight ({-L, ..., -1, 0, 1, ..., +L })
        W: The weight matrix with dimensions [K, N].
    """

    def __init__(self, name, K=8, N=12, L=4):
        """
        Args:
            K: The number of hidden neurons.
            N: Then number of input neurons connected to each hidden neuron.
            L: Boundaries of each weight ({-L, ..., -1, 0, 1, ..., +L}).
        """
        self.name = name
        with tf.name_scope(name):
            self.K = tf.constant(K)
            self.N = tf.constant(N)
            self.L = tf.constant(L)
            self.W = tf.Variable(tf.random.uniform(
                (K, N), minval=-L, maxval=L + 1, dtype=tf.int64),
                trainable=True)

    def compute_sigma(self, X):
        """
        Args:
            X: A random vector which is the input for TPM.
        Returns:
            A tuple of the vector of the outputs of each hidden perceptron and
            the vector with all 0s replaced with -1s. For example:

            ([-1, -1, 1, 0, -1, 1], [-1, -1, 1, -1, -1, 1])

            Each vector has dimension [K].
        """
        original = tf.math.sign(tf.math.reduce_sum(
            tf.math.multiply(X, self.W), axis=1))
        nonzero = tf.where(tf.math.equal(original, 0), -1, original)
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
        tau = tf.math.reduce_prod(nonzero)

        with tf.name_scope(self.name):
            self.X = X
            self.sigma = tf.Variable(sigma)
            self.tau = tau

        return tau

    def __call__(self, X):
        return self.get_output(X)

    def update(self, tau2, update_rule='hebbian'):
        """
        Updates the weights according to the specified update rule.

        Args:
            tau2: Output bit from the other machine, must be -1 or 1.
            update_rule: The update rule, must be 'hebbian', 'anti_hebbian', or 'random_walk'.
        """
        if tf.math.equal(self.tau, tau2):
            if tf.math.equal(update_rule, 'hebbian'):
                hebbian(self.W, self.X, self.sigma, self.tau, tau2, self.L)
            elif tf.math.equal(update_rule, 'anti_hebbian'):
                anti_hebbian(self.W, self.X, self.sigma,
                             self.tau, tau2, self.L)
            elif tf.math.equal(update_rule, 'random_walk'):
                random_walk(self.W, self.X, self.sigma,
                            self.tau, tau2, self.L)
            else:
                raise Exception("Invalid update rule. Valid update rules are: "
                                + "\'hebbian\', "
                                + "\'anti_hebbian\' and "
                                + "\'random_walk\'.")
            if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
                with tf.name_scope(self.name):
                    xaxis = tf.range(1, self.K + 1)
                    yaxis = tf.range(1, self.N + 1)
                    tb_heatmap('weights', self.W, xaxis, yaxis)
                    tb_boxplot('weights', self.W, xaxis)
                    for i in tf.range(self.K):
                        with tf.name_scope(f'hperceptron{i+1}'):
                            tb_summary('weights', self.W[i])
                            tb_summary('sigma', self.sigma)

    def makeKey(self, key_length, iv_length):
        """
        Args:
            key_length: Length of key, must be 128, 192, or 256.
            iv_length: Length of IV, must be a multiple of 4 between 0 and 256, inclusive.
        Returns:
            The key and IV based on the TPM's weights.
        """
        key = ""
        iv = ""

        for i in tf.range(self.K):
            for j in tf.range(self.N):
                if tf.math.equal(i, j):
                    iv += tf.strings.as_string(self.W[i, j])
                key += tf.strings.as_string(self.W[i, j])

        def convert_to_hex_dig(input, is_iv=True):
            return hashlib.sha512(
                str(input).encode('utf-8')).hexdigest()[0:int(iv_length / 4
                                                              if is_iv else
                                                              key_length / 4)]

        current_key = convert_to_hex_dig(key, is_iv=False)
        current_iv = convert_to_hex_dig(iv, is_iv=True)
        if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
            with tf.name_scope(self.name):
                tf.summary.text('independent variable',
                                data=current_iv)
                tf.summary.text('key',
                                data=current_key)
        return (current_key, current_iv)


class ProbabilisticTPM(TPM):
    """
        W: A [K, N, 2L+1] matrix representing the PDF of the weight distribution.
        mu_W: A [K, N] matrix of the averages of the weight distributions.
        sigma_W: A [K, N] matrix of the standard deviations of the weight distributions.
    """

    def __init__(self, name, K=8, N=12, L=4):
        super().__init__(name, K=8, N=12, L=4)
        self.W = tf.fill([K, N, 2 * L + 1], 1. / (2 * L + 1))

    def normalize_weights(self, i=-1, j=-1):
        """
        Normalizes the probability distribution associated with W[i, j]. If
        negative indeces i, j are provided, the normalization is carried out
        for all the probability distributions.

        Args:
            i: Index of the hidden perceptron distribution to normalize.
            j: Index of the input perceptron distribution to normalize.
        """
        if (j < 0 and i < 0):
            self.W.assign(tf.map_fn(lambda x: tf.math.reduce_mean(x), self.W))
        else:
            self.W[i, j].assign(
                tf.map_fn(lambda x: tf.math.reduce_mean(x), self.W[i, j]))

    def get_most_probable_weight(self):
        """
        Returns:
            A [K, N] matrix with each cell representing the weight which has
            the largest probability of existing in the defender's TPM.
        """
        mPW = tf.Variable(tf.zeros([self.K, self.N], tf.int64))
        # TODO: more efficient way to do this?
        for i in tf.range(self.K):
            for j in tf.range(self.N):
                mPW.assign(
                    tf.map_fn(lambda x: tf.argmax(x) - self.L, self.W[i, j]))
        return mPW

    def update(self, update_rule="hebbian"):
        """
        Args:
            update_rule: Must be "monte_carlo" or "hebbian".
        """
        pass


class GeometricTPM(TPM):
    def update_sigma(self):
        """
        Updates sigma using the geometric algorithm.
        """
        wx = tf.math.reduce_sum(tf.math.multiply(self.X, self.W), axis=1)
        original = tf.math.sign(wx)
        h_i = tf.math.divide(tf.cast(wx, tf.float64),
                             tf.math.sqrt(tf.cast(self.N, tf.float64)))
        min = tf.math.argmin(tf.math.abs(h_i))  # index of min
        self.sigma[min].assign(tf.math.negative(original[min]))

    def update(self, tau2, update_rule='hebbian', geometric=False):
        """
        Updates the weights according to the specified update rule.

        Args:
            tau2: Output bit from the other machine, must be -1 or 1.
            update_rule: The update rule, must be 'hebbian', 'anti_hebbian', or 'random_walk'.
        """
        if geometric:
            self.update_sigma()

        if tf.math.equal(self.tau, tau2):
            if tf.math.equal(update_rule, 'hebbian'):
                hebbian(self.W, self.X, self.sigma, self.tau, tau2, self.L)
            elif tf.math.equal(update_rule, 'anti_hebbian'):
                anti_hebbian(self.W, self.X, self.sigma,
                             self.tau, tau2, self.L)
            elif tf.math.equal(update_rule, 'random_walk'):
                random_walk(self.W, self.X, self.sigma,
                            self.tau, tau2, self.L)
            else:
                raise Exception("Invalid update rule. Valid update rules are: "
                                + "\'hebbian\', "
                                + "\'anti_hebbian\' and "
                                + "\'random_walk\'.")
            if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
                with tf.name_scope(self.name):
                    xaxis = tf.range(1, self.K + 1)
                    yaxis = tf.range(1, self.N + 1)
                    tb_heatmap('weights', self.W, xaxis, yaxis)
                    tb_boxplot('weights', self.W, xaxis)
                    for i in tf.range(self.K):
                        with tf.name_scope(f'hperceptron{i+1}'):
                            tb_summary('weights', self.W[i])
                            tb_summary('sigma', self.sigma)
