import tensorflow as tf
from hashlib import sha512
from .basic import TPM
import mlencrypt_research.update_rules.probabilistic

autograph_features = tf.autograph.experimental.Feature.all_but(
    tf.autograph.experimental.Feature.NAME_SCOPES)


# class DPDFWeight(tf.Module):
#     def __init__(self, L, initial_probabilities=None):
#         if not initial_probabilities:
#             initial_probabilities = tf.constant(tf.fill(
#                 [2 * L + 1],
#                 tf.guarantee_const(tf.constant(1. / (2 * L + 1),
#                                                tf.float16)),
#             ))
#         self.probs = tf.Variable(initial_probabilities, trainable=True)
#         self.L = L
#
#     def sample(self):
#         # https://github.com/brigan/NeuralCryptography/blob/dec94a21f5de316bd7a87e24f55af23eb146fccd/TPM.h#L944-L973
#         rand = tf.random.uniform([])
#         output = -self.L
#         for prob in self.probs:
#             if rand < prob:
#                 break
#             output += 1
#             rand -= prob
#
#     def assign(self, new_probs):
#         return self.probs.assign(new_probs)


# class DPDFWeight2(tf.Variable):
#     def __init__(self,
#                  L,
#                  initial_value=None,
#                  trainable=None,
#                  validate_shape=True,
#                  caching_device=None,
#                  name=None,
#                  variable_def=None,
#                  dtype=None,
#                  import_scope=None,
#                  constraint=None,
#                  synchronization=tf.VariableSynchronization.AUTO,
#                  aggregation=tf.VariableAggregation.NONE,
#                  shape=None):
#         self.L = L
#         # https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/ops/variables.py#L1339
#         super(DPDFWeight, self)._OverloadAllOperators()
#         super(DPDFWeight, self).__init__(initial_value=initial_value,
#                                          trainable=trainable,
#                                          validate_shape=validate_shape,
#                                          caching_device=caching_device,
#                                          name=name,
#                                          variable_def=variable_def,
#                                          dtype=dtype,
#                                          import_scope=import_scope,
#                                          constraint=constraint,
#                                          synchronization=synchronization,
#                                          aggregation=aggregation,
#                                          shape=shape)
#
#     def sample(self):
#         # https://github.com/brigan/NeuralCryptography/blob/dec94a21f5de316bd7a87e24f55af23eb146fccd/TPM.h#L944-L973
#         rand = tf.random.uniform([])
#         output = -self.L
#         for prob in self.probs:
#             if rand < prob:
#                 break
#             output += 1
#             rand -= prob


class ProbabilisticTPM(TPM):
    """
    Attributes:
        W: [K, N, 2L+1] matrix representing the PDF of the weight distribution.
        mu_W: [K, N] matrix of the averages of the weight distributions.
        sigma_W: [K, N] matrix of the standard deviations of the weight
            distributions.
    """

    def __init__(self, name, K, N, L):
        super().__init__(name, K, N, L, None)
        self.type = 'probabilistic'
        with self.name_scope:
            # sampling once from a multinomial distribution is the same as
            # sampling from a discrete probability density function
            initial_weights = tf.fill(
                [K, N, 2 * L + 1],
                tf.guarantee_const(tf.constant(
                    1. / (2 * L + 1), tf.float16)),
            )
            # initial_weights = tf.random.uniform(
            #     (self.K, self.N, 2*self.L+1),
            #     dtype=tf.float16
            # )
            self.w = tf.Variable(initial_weights, trainable=True)
            # self.normalize_weights()
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
            corresponding_weights = tf.range(-self.L, self.L + 1, delta=1.)
            return tf.math.reduce_mean(dpdf * tf.cast(corresponding_weights, dtype=dpdf.dtype))
        self.eW.assign(tf.reshape(
            tf.map_fn(
                lambda wi: tf.map_fn(
                    # lambda wij: get_expected_weight(wij)
                    get_expected_weight,
                    wi,
                ),
                self.w
            ),
            (self.K, self.N),
        ))
        return self.eW

    def sample(self):
        # https://github.com/brigan/NeuralCryptography/blob/dec94a21f5de316bd7a87e24f55af23eb146fccd/TPM.h#L944-L973
        current_sample_rows = tf.TensorArray(tf.int32, size=self.K)
        for index_k in tf.range(self.K):
            current_sample_cols = tf.TensorArray(tf.int32, size=self.N)
            for index_n in tf.range(self.N):
                # float scalar bound by [0, 1):
                rand = tf.random.uniform([], dtype=self.w.dtype)
                sampled_value = -self.L
                for prob in self.w[index_k][index_n]:
                    if rand < prob:
                        break
                    sampled_value += 1
                    rand -= prob
                current_sample_cols = current_sample_cols.write(
                    index_n,
                    sampled_value,
                )
            current_sample_rows = current_sample_rows.write(
                index_k, current_sample_cols.stack())
        # int32 matrix with shape [K, N] bound by [-L, L]:
        current_sample = current_sample_rows.stack()
        return current_sample

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

    def normalize_weights(self, i=None, j=None):
        """Normalizes probability distributions.

        Normalizes the probability distribution associated with W[i, j]. If
        negative indeces i, j are provided, the normalization is carried out
        for the entire probability distributions.

        Args:
            i (int): Index of the hidden neuron distribution to normalize.
            j (int): Index of the input neuron distribution to normalize.
        """
        # https://github.com/brigan/NeuralCryptography/blob/dec94a21f5de316bd7a87e24f55af23eb146fccd/TPM.h#L581-L619
        if i and j:
            self.w.assign(
                tf.vectorized_map(
                    lambda wi: tf.vectorized_map(
                        lambda wij: wij / tf.math.reduce_sum(wij),
                        wi,
                    ),
                    self.w,
                ),
            )
        else:
            self.w[i, j].assign(
                self.w[i][j] / tf.math.reduce_sum(self.w[i][j])
            )

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
                lambda wi: tf.map_fn(
                    lambda wij: self.index_to_weight(tf.math.argmax(wij)),
                    wi,
                    dtype=tf.int32,
                ),
                self.w,
                dtype=tf.int32,
            ),
            (self.K, self.N),
        ))
        return self.mpW

    @tf.function(
        experimental_autograph_options=autograph_features,
        experimental_relax_shapes=True,
    )
    def update(self, tau2, updated_A_B, num_samples=10):
        # https://pdfs.semanticscholar.org/a4d1/66b13f6297438cb95f71c0445bee5743a2f2.pdf#page=55
        if updated_A_B:
            num_valid_samples = 0
            valid_samples = tf.TensorArray(
                self.w.dtype,
                size=num_samples,  # this is the max size
                infer_shape=False,
                element_shape=self.w.shape,
            )
            for _ in tf.range(num_samples):
                current_sample = self.sample()
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
                tau = tf.cast(tf.math.sign(
                    tf.math.reduce_prod(nonzero)), tf.int32)
                # tf.print(current_sample)
                if tf.math.equal(tau, tau2):
                    # tf.print(tf.one_hot(
                    #     current_sample[1][1] + tf.cast(self.L, current_sample.dtype),
                    #     2 * self.L + 1,
                    #     dtype=self.w.dtype,
                    # ))
                    valid_samples.write(
                        num_valid_samples,
                        tf.vectorized_map(
                            lambda si: tf.vectorized_map(
                                lambda sij: tf.one_hot(
                                    sij + tf.cast(self.L, sij.dtype),
                                    2 * self.L + 1,
                                    dtype=self.w.dtype,
                                ),
                                si,
                            ),
                            current_sample,
                        ),
                    )
                    tf.print('enc', tf.math.count_nonzero(valid_samples.read(num_valid_samples)))
                    num_valid_samples += 1
            valid_samples = valid_samples.stack()
            posterior_weights = tf.math.reduce_mean(valid_samples, axis=0)
            self.w.assign(posterior_weights)
            # self.normalize_weights()  # not necessary?
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
