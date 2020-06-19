import tensorflow as tf
from os import getenv
from hashlib import sha512
import mlencrypt_research.update_rules.basic
from .summaries import tb_summary, tb_heatmap, tb_boxplot

autograph_features = tf.autograph.experimental.Feature.all_but(
    tf.autograph.experimental.Feature.NAME_SCOPES)


class TPM(tf.Module):
    def __init__(self, name, K, N, L, initial_weights):
        """
        Args:
            K (int): The number of hidden neurons.
            N (int): The number of input neurons that each hidden
                neuron has.
            L (int): The synaptic depth of each input neuron's weights.
        """
        super(TPM, self).__init__(name=name)
        self.type = 'basic'
        with self.name_scope:
            self.K = tf.guarantee_const(tf.constant(K, name='K'))
            self.N = tf.guarantee_const(tf.constant(N, name='N'))
            self.L = tf.guarantee_const(tf.constant(L, name='L'))
            self.w = initial_weights
            self.sigma = tf.Variable(
                tf.zeros([K], dtype=tf.int32),
                trainable=False,
                name='sigma'
            )
            self.tau = tf.Variable(0, name='tau')
            self.key = tf.Variable("", name='key')
            self.iv = tf.Variable("", name='iv')

    def compute_sigma(self, X):
        """
        Args:
            X: A random vector which is the input for TPM.
        Returns:
            A tuple of the vector of the outputs of each hidden neuron and
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

        # tau is the output of the TPM, and is a binary scalar:
        tau = tf.math.sign(tf.math.reduce_prod(nonzero))
        # TODO: is the if-statement necessary?
        if tau.dtype != tf.int32:
            # tau is float16 for ProbabilisticTPM
            tau = tf.cast(tau, tf.int32)

        with self.name_scope:
            self.X = X
            # self.sigma.assign(sigma)
            self.sigma.assign(nonzero)
            self.tau.assign(tau)
            if getenv('MLENCRYPT_TB', 'FALSE') == 'TRUE':
                tf.summary.scalar('tau', self.tau)

        return tau

    def __call__(self, X):
        return self.get_output(X)

    @tf.function(
        experimental_autograph_options=autograph_features,
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
            if getenv('MLENCRYPT_TB', 'FALSE') == 'TRUE':
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
                                scope=weights_scope,
                                ylabel='weights',
                            )

                            # def log_hneuron(scope_name, value):
                            #     with tf.name_scope(scope_name.decode("utf-8")):
                            #         tb_summary('weights', value)
                            #
                            # for i in range(self.K):
                            #     # hneuron weights aren't logged, see
                            #     # https://github.com/tensorflow/tensorflow/issues/38772
                            #     scope_name = tf.strings.format(
                            #         "hneuron{}",
                            #         i + 1
                            #     )
                            #     value = self.w[i]
                            #     tf.numpy_function(
                            #         log_hneuron,
                            #         [scope_name, value],
                            #         [],
                            #         name='tb-images-weights'
                            #     )
                    except tf.errors.OutOfRangeError:
                        tf.experimental.async_clear_error()
            return True
        else:
            return False

    @tf.function(
        experimental_autograph_options=autograph_features,
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
            return sha512(
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
        self.key.assign(current_key)
        self.iv.assign(current_iv)
        with self.name_scope:
            tf.summary.text('key', data=current_key)
            tf.summary.text('independent variable', data=current_iv)
        return current_key, current_iv
