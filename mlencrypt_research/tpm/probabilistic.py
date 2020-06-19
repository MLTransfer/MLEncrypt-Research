import tensorflow as tf
from os import getenv
from hashlib import sha512
from .basic import TPM
import mlencrypt_research.update_rules.probabilistic

autograph_features = tf.autograph.experimental.Feature.all_but(
    tf.autograph.experimental.Feature.NAME_SCOPES)


class ProbabilisticTPM(TPM):
    """
    Attributes:
        W: [K, N, 2L+1] matrix representing the PDF of the weight distribution.
        mu_W: [K, N] matrix of the averages of the weight distributions.
        sigma_W: [K, N] matrix of the standard deviations of the weight
            distributions.
    """

    def __init__(self, name, K, N, L, initial_weights):
        import tensorflow_probability as tfp
        super().__init__(name, K, N, L, initial_weights)
        self.type = 'probabilistic'
        with self.name_scope:
            # sampling once from a multinomial distribution is the same as
            # sampling from a discrete probability density function
            self.w = tfp.distributions.OneHotCategorical(
                probs=tf.fill(
                    [K, N, 2 * L + 1],
                    tf.guarantee_const(tf.constant(
                        1. / (2 * L + 1), tf.float16)),
                ),
                validate_args=getenv('MLENCRYPT_TB', 'FALSE') == 'TRUE',
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
                tf.reshape(self.w.probs_parameter(),
                           (self.K * self.N, 2 * self.L + 1)),
            ),
            (self.K, self.N)
        ))
        return self.eW

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
            tf.math.multiply(tf.cast(X, tf.float16), self.eW), axis=1))
        id = self.name[0]
        nonzero = tf.where(
            tf.math.equal(original, 0., name=f'{id}-sigma-zero'),
            tf.guarantee_const(tf.cast(-1., tf.float16, name='negative-1')),
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
            i (int): Index of the hidden neuron distribution to normalize.
            j (int): Index of the input neuron distribution to normalize.
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

    def get_most_probable_weights(self):
        """
        Returns:
            [K, N] matrix with each cell representing the weight which has
            the largest probability of existing in the defender's TPM.
        """
        self.mpW.assign(tf.reshape(
            tf.map_fn(
                lambda x: self.index_to_weight(tf.math.argmax(x)),
                tf.reshape(self.w.probs_parameter(),
                           (self.K * self.N, 2 * self.L + 1)),
                dtype=tf.int32,
            ),
            (self.K, self.N),
        ))
        return self.mpW

    @tf.function(
        experimental_autograph_options=autograph_features,
        experimental_relax_shapes=True,
    )
    def update(self, tau2, updated_A_B, num_samples=10.):
        # https://pdfs.semanticscholar.org/a4d1/66b13f6297438cb95f71c0445bee5743a2f2.pdf#page=55
        if updated_A_B:
            num_valid_samples = 0.
            posterior_weights = tf.zeros([self.K, self.N, 2*self.L+1], tf.float32)
            for sample in self.w.sample(sample_shape=num_samples):
                current_sample_rows = tf.TensorArray(tf.int32, size=self.K)
                for index_k in tf.range(self.K):
                    current_sample_cols = tf.TensorArray(tf.int32, size=self.N)
                    for index_n in tf.range(self.N):
                        current_sample_cols = current_sample_cols.write(
                            index_n,
                            tf.cast(tf.where(sample[index_k][index_n] == 1)[0][0], tf.int32) - self.L,
                        )
                    current_sample_rows = current_sample_rows.write(index_k, current_sample_cols.stack())
                current_sample = current_sample_rows.stack()  # sample from weights
                # compute inner activation sigma, [K]
                original = tf.math.sign(tf.math.reduce_sum(
                    tf.math.multiply(self.X, current_sample), axis=1))
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
            # self.w = self.w.copy(probs=posterior_weights)
        mlencrypt_research.update_rules.probabilistic.hebbian()

        self.get_expected_weights()
        self.get_most_probable_weights()

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
        iv_weights = tf.strings.format(
            "{}", tf.gather_nd(self.mpW, iv_indices))
        key_weights = tf.strings.format("{}", self.mpW)

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
