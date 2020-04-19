# -*- coding: utf-8 -*-
from update_rules import hebbian, anti_hebbian, random_walk

import hashlib
from os import remove, environ

import tensorflow as tf
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
                                 index=yaxis,
                                 columns=xaxis),
                    ax=ax,
                    cmap=cmap,
                    cbar_kws={"ticks": ticks, "boundaries": boundaries})
        ax.set(xlabel="input perceptron", ylabel="hidden perceptron")
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
            tf.zeros([K], dtype=tf.int64),
            trainable=False,
            name='sigma'
        )
        with self.name_scope:
            self.K = tf.constant(K)
            self.N = tf.constant(N)
            self.L = tf.constant(L)
            self.w = initial_weights

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
            tf.math.multiply(X, tf.cast(self.w, tf.int64)), axis=1))
        nonzero = tf.where(
            tf.math.equal(original, 0),
            tf.cast(-1, tf.int64),
            original
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
        tau = tf.math.reduce_prod(nonzero)

        with self.name_scope:
            self.X = X
            self.sigma.assign(sigma)
            self.tau = tau

        return tau

    def __call__(self, X):
        return self.get_output(X)

    def update(self, tau2, update_rule):
        """Updates the weights according to the specified update rule.

        Args:
            tau2 (int): Output bit from the other machine, must be -1 or 1.
            update_rule (str): The update rule, must be 'hebbian',
                'anti_hebbian', or 'random_walk'.
        """
        if tf.math.equal(self.tau, tau2):
            if update_rule == 'hebbian':
                hebbian(self.w, self.X, self.sigma, self.tau, tau2, self.L)
            elif update_rule == 'anti_hebbian':
                anti_hebbian(self.w, self.X, self.sigma,
                             self.tau, tau2, self.L)
            elif update_rule == 'random_walk':
                random_walk(self.w, self.X, self.sigma,
                            self.tau, tau2, self.L)
            else:
                raise ValueError(
                    f"'{update_rule}' is an invalid update rule. "
                    + "Valid update rules are: "
                    + "\'hebbian\', "
                    + "\'anti_hebbian\' and "
                    + "\'random_walk\'."
                )
            if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
                # TODO: don't refer to variables outside of the method scope,
                # add them as arguments (maybe tf.numpy_function) will help
                with self.name_scope:
                    tb_summary('sigma', self.sigma)
                    tf.summary.histogram('weights', self.w)

                    def log_images():
                        for i in range(self.K):
                            with tf.name_scope(f'hperceptron{i+1}'):
                                tb_summary('weights', self.w[i])

                        # hpaxis, ipaxis = tf.range(
                        #     1, self.K + 1), tf.range(1, self.N + 1)
                        # tb_heatmap('weights', self.w, ipaxis, hpaxis)
                        # tb_boxplot('weights', self.w, hpaxis)
                        # pass
                    tf.py_function(log_images, [], [])

    def makeKey(self, key_length, iv_length):
        """Creates a key and IV based on the weights of this TPM.

        Args:
            key_length (int): Length of the key.
                Must be 128, 192, or 256.
            iv_length (int): Length of the independent variable.
                Must be a multiple of 4 between 0 and 256, inclusive.
        Returns:
            The key and IV based on the TPM's weights.
        """
        key_weights = tf.constant("")
        iv_weights = tf.constant("")

        for i in tf.range(self.K):
            for j in tf.range(self.N):
                if tf.math.equal(i, j):
                    iv_weights += tf.strings.as_string(self.w[i, j])
                key_weights += tf.strings.as_string(self.w[i, j])

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
        if not hasattr(self, 'key'):
            self.key = tf.Variable('', trainable=False, name='key')
        if not hasattr(self, 'iv'):
            self.iv = tf.Variable('', trainable=False, name='iv')
        self.key.assign(current_key)
        self.iv.assign(current_iv)
        # with self.name_scope:
        #     tf.summary.text('key', data=current_key)
        #     tf.summary.text('independent variable', data=current_iv)
        return current_key, current_iv


class ProbabilisticTPM(TPM):
    """
    Attributes:
        W: [K, N, 2L+1] matrix representing the PDF of the weight distribution.
        mu_W: [K, N] matrix of the averages of the weight distributions.
        sigma_W: [K, N] matrix of the standard deviations of the weight
            distributions.
    """

    def __init__(self, name, K=8, N=12, L=4, initial_weights=None):
        super().__init__(name, K=K, N=N, L=L, initial_weights=initial_weights)
        self.type = 'probabilistic'
        self.w = tf.Variable(
            tf.fill([K, N, 2 * L + 1], 1. / (2 * L + 1)), trainable=True)

    def normalize_weights(self, i=-1, j=-1):
        """Normalizes probability distributions.

        Normalizes the probability distribution associated with W[i, j]. If
        negative indeces i, j are provided, the normalization is carried out
        for the entire probability distributions.

        Args:
            i (int): Index of the hidden perceptron distribution to normalize.
            j (int): Index of the input perceptron distribution to normalize.
        """
        if (j < 0 and i < 0):
            self.w.assign(tf.map_fn(lambda x: tf.math.reduce_mean(x), self.w))
        else:
            self.w[i, j].assign(
                tf.map_fn(lambda x: tf.math.reduce_mean(x), self.w[i, j]))

    def get_most_probable_weight(self):
        """
        Returns:
            [K, N] matrix with each cell representing the weight which has
            the largest probability of existing in the defender's TPM.
        """
        mPW = tf.Variable(tf.zeros([self.K, self.N], tf.int64))
        # TODO: more efficient way to do this?
        for i in tf.range(self.K):
            for j in tf.range(self.N):
                mPW.assign(
                    tf.map_fn(lambda x: tf.argmax(x) - self.L, self.w[i, j]))
        return mPW

    def update(self, update_rule):
        """
        Args:
            update_rule (str): Must be "monte_carlo" or "hebbian".
        """
        pass


class GeometricTPM(TPM):
    # TODO: it might just be autograph but geometric isn't working,
    # running an hparams with 1 trial returns the same results for both attacks
    def __init__(self, name, K=8, N=12, L=4, initial_weights=None):
        super().__init__(name, K=K, N=N, L=L, initial_weights=initial_weights)
        self.type = 'geometric'

    def update_sigma(self):
        """Updates sigma using the geometric algorithm.

        Negates the sigma value of the hidden perceptron with the lowest
        current state.
        """
        wx = tf.math.reduce_sum(tf.math.multiply(self.X, self.w), axis=1)
        original = tf.math.sign(wx)
        h_i = tf.math.divide(tf.cast(wx, tf.float64),
                             tf.math.sqrt(tf.cast(self.N, tf.float64)))
        min = tf.math.argmin(tf.math.abs(h_i))  # index of min
        self.sigma[min].assign(tf.math.negative(original[min]))

    def update(self, tau2, update_rule):
        """Updates the weights according to the specified update rule.

        Args:
            tau2 (int): Output bit from the other machine, must be -1 or 1.
            update_rule (str): The update rule, must be 'hebbian',
                'anti_hebbian', or 'random_walk'.
        """
        self.update_sigma()
        TPM.update(self, tau2, update_rule)
