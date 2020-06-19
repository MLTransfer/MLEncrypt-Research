import tensorflow as tf
from .basic import TPM

autograph_features = tf.autograph.experimental.Feature.all_but(
    tf.autograph.experimental.Feature.NAME_SCOPES)


class GeometricTPM(TPM):
    def __init__(self, name, K, N, L, initial_weights):
        super().__init__(name, K, N, L, initial_weights)
        self.type = 'geometric'

    def update_sigma(self):
        """Updates sigma using the geometric algorithm.

        Negates the sigma value of the hidden neuron with the lowest
        current state.
        """
        # https://arxiv.org/pdf/0711.2411.pdf#page=33
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
        self.tau.assign(tf.bitcast(tf.math.sign(
            tf.math.reduce_prod(self.sigma)), tf.int32))

    def update(self, tau2, update_rule):
        """Updates the weights according to the specified update rule.

        Args:
            tau2 (int): Output bit from the other machine, must be -1 or 1.
            update_rule (str): The update rule, must be 'hebbian',
                'anti_hebbian', or 'random_walk'.
        """
        updated = super().update(tau2, update_rule)
        if updated:
            return True
        else:
            self.update_sigma()
            super().update(tau2, update_rule)
            return False
