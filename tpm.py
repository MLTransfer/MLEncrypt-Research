import hashlib
import tensorflow as tf

from update_rules import hebbian, anti_hebbian, random_walk

from os import remove

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
        sns.heatmap(pd.DataFrame(data=data.numpy(),
                                 index=xaxis, columns=yaxis).transpose(), ax=ax)
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
    '''Machine
    A tree parity machine. Generates a binary digit(tau) for a given random vector(X).
    The machine can be described by the following parameters:
    k - The number of hidden neurons
    n - Then number of input neurons connected to each hidden neuron
    l - Defines the range of each weight ({-L, ..., -2, -1, 0, 1, 2, ..., +L })
    W - The weight matrix between input and hidden layers. Dimensions : [K, N]
    '''

    def __init__(self, name, K=8, N=12, L=4):
        '''
        Arguments:
        k - The number of hidden neurons
        n - Then number of input neurons connected to each hidden neuron
        l - Defines the range of each weight ({-L, ..., -2, -1, 0, 1, 2, ..., +L })		'''
        self.name = name
        with tf.name_scope(name):
            self.K = tf.constant(K)
            self.N = tf.constant(N)
            self.L = tf.constant(L)
            self.W = tf.Variable(tf.random.uniform(
                (K, N), minval=-L, maxval=L + 1, dtype=tf.int64), trainable=True)

    def get_output(self, X):
        '''
        Returns a binary digit tau for a given random vecor.
        Arguments:
        X - Input random vector
        '''

        W = self.W
        tf.reshape(X, [self.K, self.N])

        # Compute inner activation sigma Dimension:[K]
        sigma = tf.math.sign(tf.math.reduce_sum(
            tf.math.multiply(X, W), axis=1))
        tau = tf.math.reduce_prod(sigma)  # The final output

        with tf.name_scope(self.name):
            self.X = X
            self.sigma = sigma
            self.tau = tau

        return tau

    def __call__(self, X):
        return self.get_output(X)

    def update(self, tau2, update_rule='hebbian'):
        '''
        Updates the weights according to the specified update rule.
        Arguments:
        tau2 - Output bit from the other machine;
        update_rule - The update rule.
        Should be one of ['hebbian', 'anti_hebbian', random_walk']
        '''
        if (self.tau == tau2):
            if update_rule == 'hebbian':
                hebbian(self.W, self.X, self.sigma, self.tau, tau2, self.L)
            elif update_rule == 'anti_hebbian':
                anti_hebbian(self.W, self.X, self.sigma,
                             self.tau, tau2, self.L)
            elif update_rule == 'random_walk':
                random_walk(self.W, self.X, self.sigma,
                            self.tau, tau2, self.L)
            else:
                raise Exception("Invalid update rule. Valid update rules are: "
                                + "\'hebbian\', \'anti_hebbian\' and \'random_walk\'.")
            with tf.name_scope(self.name):
                xaxis = tf.range(1, self.K+1)
                yaxis = tf.range(1, self.N+1)
                tb_heatmap('weights', self.W, xaxis, yaxis)
                tb_boxplot('weights', self.W, xaxis)
                for i in tf.range(self.K):
                    with tf.name_scope(f'hperceptron{i+1}'):
                        tb_summary('weights', self.W[i])
                        tb_summary('sigma', self.sigma)
            return

    def makeKey(self, key_length, iv_length):
        '''
        weight matrix to key and iv : use sha512 on concatenated weights
        '''
        key = ""
        iv = ""

        # generate key
        for i in tf.range(self.K):
            for j in tf.range(self.N):
                if i == j:
                    iv += tf.as_string(self.W[i, j])
                key += tf.as_string(self.W[i, j])

        def convert_to_hex_dig(input, is_iv=True):
            return hashlib.sha512(
                str(input).encode('utf-8')).hexdigest()[0:int(iv_length / 4 if is_iv else key_length / 4)]

        with tf.name_scope(self.name):
            tf.summary.text('independent variable',
                            data=convert_to_hex_dig(iv, is_iv=True))
            tf.summary.text('key',
                            data=convert_to_hex_dig(key, is_iv=False))
        return (convert_to_hex_dig(key, is_iv=False), convert_to_hex_dig(iv, is_iv=True))
