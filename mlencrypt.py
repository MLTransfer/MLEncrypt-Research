# -*- coding: utf-8 -*-
from tpm import TPM
from tpm import tb_summary, tb_heatmap, tb_boxplot

from os import environ
from time import perf_counter
from importlib import import_module

import tensorflow as tf
from math import pi


def compute_overlap_matrix(N, L, w1, w2):
    """Computes the overlap matrix for the two vectors provided.

    Args:
        N (int): The number of inputs to the hidden units.
        L (int): The synaptic depth of the perceptrons.
        w1: The weights of a specific input perceptron in the first TPM.
        w2: The weights of a specific input perceptron in the second TPM.

    Returns:
        The overlap matrix of the vectors.
    """
    # shape_f = tf.constant([2 * L + 1, 2 * L + 1])
    f = tf.Variable(tf.constant(
        tf.zeros([2 * L + 1, 2 * L + 1], dtype=tf.float64)))

    one_over_n = tf.constant(tf.math.divide(
        1., tf.cast(N, dtype=tf.float64)), dtype=tf.float64)
    for i in tf.range(N):
        r, c = w1[i] + tf.cast(L, tf.int64), w2[i] + tf.cast(L, tf.int64)
        f[r, c].assign(f[r, c] + one_over_n)

    return f


def compute_overlap_matrix_probabilistic(N, L, w1, w2):
    """Computes the overlap matrix for the two vectors provided.

    Args:
        N (int): The number of inputs to the hidden units.
        L (int): The synaptic depth of the perceptrons.
        w1: The weights of a specific input perceptron in the first TPM.
        w2: The weights of a specific input perceptron in the second TPM.

    Returns:
        The overlap matrix of the vectors.
    """
    f = tf.Variable(tf.zeros([2 * L + 1, 2 * L + 1], tf.float64))
    for i in tf.range(N):
        for j in tf.range(2 * L + 1):
            f[w1[i] + L, j].assign(f[w1[i] + L, j] + w2[i][j])
    return f


def compute_overlap_from_matrix(N, L, f):
    """Computes the overlap of two vectors from their overlap matrix.

    Args:
        N (int): The number of inputs per hidden unit.
        L (int): The depth of the weights.
        f: The overlap matrix.
    """
    R = tf.Variable(0., dtype=tf.float64)
    Q1 = tf.Variable(0., dtype=tf.float64)
    Q2 = tf.Variable(0., dtype=tf.float64)
    for i in tf.range(2 * L + 1):
        for j in tf.range(2 * L + 1):
            Q1.assign_add(tf.cast(i - L, tf.float64)
                          * tf.cast(i - L, tf.float64) * f[i, j])
            Q2.assign_add(tf.cast(j - L, tf.float64)
                          * tf.cast(j - L, tf.float64) * f[i, j])
            R.assign_add(tf.cast(i - L, tf.float64)
                         * tf.cast(j - L, tf.float64) * f[i, j])

    rho = tf.constant(tf.math.divide(
        R, tf.math.sqrt(tf.math.multiply(Q1, Q2))))
    return rho


def sync_score(TPM1, TPM2):
    """
    Args:
        TPM1: Tree Parity Machine 1. Has same parameters as TPM2.
        TPM2: Tree Parity Machine 2. Has same parameters as TPM1.
    Returns:
        The synchronization score between TPM1 and TPM2.
    """
    tpm1_id, tpm2_id = TPM1.name[0], TPM2.name[0]
    # adapted from:
    # https://github.com/tensorflow/tensorflow/blob/f270180a6caa8693f2b2888ac7e6b8e69c4feaa8/tensorflow/python/keras/losses.py#L1073-L1093
    # TODO: am I using experimental_implements correctly?
    # TODO: output is sometimes negative!
    @tf.function(experimental_implements="cosine_similarity")
    def cosine_similarity(weights1, weights2):
        """Computes the cosine similarity between labels and predictions.

        Note that it is a negative quantity between 0 and 1, where 0 indicates
        orthogonality and values closer to 1 indicate greater similarity.

        `loss = sum(y_true * y_pred)`

        Args:
            y_true: Tensor of true targets.
            y_pred: Tensor of predicted targets.
            axis: Axis along which to determine similarity.

        Returns:
            Cosine similarity tensor.
        """
        weights1_flattened = tf.reshape(
            weights1, [-1], name=f'weights-{tpm1_id}-1d')
        weights2_flattened = tf.reshape(
            weights2, [-1], name=f'weights-{tpm2_id}-1d')
        weights1_float = tf.cast(weights1_flattened, tf.float64,
                                 name=f'weights-{tpm1_id}-1d-float')
        weights2_float = tf.cast(weights2_flattened, tf.float64,
                                 name=f'weights-{tpm2_id}-1d-float')
        weights1_norm = tf.math.l2_normalize(weights1_float, axis=-1)
        weights2_norm = tf.math.l2_normalize(weights2_float, axis=-1)
        return tf.math.reduce_sum(weights1_norm * weights2_norm, axis=-1)
    rho = cosine_similarity(TPM1.w, TPM2.w)

    # the generalization error, epsilon, is the probability that a repulsive
    # step occurs if two corresponding hidden units have different sigma
    # (Ruttor, 2006).
    epsilon = tf.math.multiply(
        tf.constant(1. / pi, tf.float32, name='reciprocal-pi'),
        tf.cast(
            tf.math.acos(rho, name=f'angle-{tpm1_id}-{tpm2_id}'),
            tf.float32,
            name=f'angle-{tpm1_id}-{tpm2_id}'
        ),
        name=f'scale-angle-{tpm1_id}-{tpm2_id}-to-0-1'
    )

    if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
        with tf.name_scope(f'{TPM1.name}-{TPM2.name}'):
            tf.summary.scalar('sync', data=rho)
            tf.summary.scalar('generalization-error', data=epsilon)

    return rho


# @tf.function(experimental_compile=True)
@tf.function
def iterate(
    X,
    Alice, Bob, Eve, update_rule,
    nb_updates, nb_eve_updates,
    score, score_eve,
    key_length, iv_length
):
    tf.summary.experimental.set_step(tf.cast(nb_updates, tf.int64))

    no_hparams = tf.math.equal(
        tf.constant(environ["MLENCRYPT_HPARAMS"], name='setting-hparams'),
        tf.constant('FALSE', name='false'),
        name='do-not-use-hparams'
    )

    # TODO: use tf.cond and no_hparams
    if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
        tb_summary('inputs', X)
        # TODO: don't refer to variables outside of the method scope,
        # add them as arguments

        # @tf.autograph.experimental.do_not_convert
        def log_inputs(inputs):
            # TODO: uncomment this:
            K, N = Alice.K, Alice.N
            hpaxis, ipaxis = tf.range(1, K + 1), tf.range(1, N + 1)
            # tb_heatmap('inputs', X, ipaxis, hpaxis)
            # tb_boxplot('inputs', X, hpaxis)
        tf.py_function(log_inputs, [X], [], name='tb-images-inputs')

    # compute outputs of TPMs
    tauA = Alice.get_output(X)
    tauB = Bob.get_output(X)
    tauE = Eve.get_output(X)
    Alice.update(tauB, update_rule)
    Bob.update(tauA, update_rule)
    nb_updates.assign_add(1, name='updates-A-B-increment')
    tf.summary.experimental.set_step(tf.cast(nb_updates, tf.int64))

    # if tauA equals tauB and tauB equals tauE then tauA must equal tauE
    # due to the associative law of boolean multiplication
    tau_A_B_equal = tf.math.equal(tauA, tauB, name='iteration-tau-A-B-equal')
    tau_B_E_equal = tf.math.equal(tauB, tauE, name='iteration-tau-B-E-equal')
    tau_A_B_E_equal = tf.math.logical_and(
        tau_A_B_equal, tau_B_E_equal, name='iteration-tau-A-B-E-equal')
    tau_A_E_not_equal = tf.math.not_equal(
        tauA, tauE, name='iteration-tau-A-E-not-equal')
    update_E_geometric = tf.math.logical_and(
        tf.math.logical_and(tau_A_B_equal, tau_A_E_not_equal,
                            name='tau-A-E-not-equal'),
        tf.math.equal(Eve.type, 'geometric', name='is-E-geometric'),
        name='update-E-geometric'
    )
    should_update_E = tf.math.logical_or(
        tau_A_B_E_equal, update_E_geometric, name='iteration-update-E')

    def update_E():
        Eve.update(tauA, update_rule)
        nb_eve_updates.assign_add(1, name='updates-E-increment')
    tf.cond(
        should_update_E,
        true_fn=update_E,
        false_fn=lambda: None,
        name='iteration-update-E'
    )

    # if tauA equals tauB and tauB does not tauE
    # then tauA does not equal tauE

    def log_updates_E():
        tf.summary.scalar('updates-E', data=nb_eve_updates)
    tf.cond(no_hparams, true_fn=log_updates_E,
            false_fn=lambda: None, name='tb-updates-E')

    def compute_and_log_keys_and_ivs():
        Alice_key, Alice_iv = Alice.makeKey(key_length, iv_length)
        Bob_key, Bob_iv = Bob.makeKey(key_length, iv_length)
        Eve_key, Eve_iv = Eve.makeKey(key_length, iv_length)

    tf.cond(no_hparams, true_fn=compute_and_log_keys_and_ivs,
            false_fn=lambda: None, name='tb-keys-ivs')

    score.assign(tf.cast(100. * sync_score(Alice, Bob),
                         tf.float32), name='calc-sync-A-B')
    score_eve.assign(tf.cast(100. * sync_score(Alice, Eve),
                             tf.float32), name='calc-sync-A-E')

    # def calc_and_log_sync_B_E():
    #     sync_score(Bob, Eve)
    # # log adversary score for Bob's weights
    # tf.cond(no_hparams, true_fn=calc_and_log_sync_B_E,
    #         false_fn=lambda: None, name='calc-sync-B-E')
    if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
        # log adversary score for Bob's weights
        sync_score(Bob, Eve)

    tf.print(
        "\rUpdate rule = ", update_rule, " / "
        "A-B Synchronization = ", score, "% / ",
        "A-E Synchronization = ", score_eve, "% / ",
        nb_updates, " Updates (Alice) / ",
        nb_eve_updates, " Updates (Eve)",
        sep='',
        name='log-iteration'
    )


# @tf.function
def run(
    update_rule, K, N, L,
    attack,
    initial_weights,
    key_length=256, iv_length=128
):
    tf.print(
        "\n\n\n"
        f"Creating machines: K={K}, N={N}, L={L}, "
        f"update-rule={update_rule}, "
        f"attack={attack}",
        "\n",
        name='log-run-initialization'
    )
    Alice = TPM('Alice', K, N, L, initial_weights['Alice'])

    Bob = TPM('Bob', K, N, L, initial_weights['Bob'])

    tpm_mod = import_module('tpm')  # TODO: don't reimport entire file?
    if attack == 'probabilistic':
        Eve_class_name = 'ProbabilisticTPM'
    elif attack == 'geometric':
        Eve_class_name = 'GeometricTPM'
    else:
        Eve_class_name = 'TPM'
    Eve_class = getattr(tpm_mod, Eve_class_name)
    Eve = Eve_class('Eve', K, N, L, initial_weights['Eve'])

    try:
        # synchronization score of Alice and Bob
        score = tf.Variable(0.0, trainable=False, name='score-A-B')
    except ValueError:
        # tf.function-decorated function tried to create variables
        # on non-first call.
        score = 0.
    try:
        # synchronization score of Alice and Eve
        score_eve = tf.Variable(0.0, trainable=False, name='score-A-E')
    except ValueError:
        # tf.function-decorated function tried to create variables
        # on non-first call.
        score_eve = 0.

    # https://www.tensorflow.org/tutorials/customization/performance#zero_iterations
    with Alice.name_scope:
        Alice.key = tf.Variable("", trainable=False, name='key')
        Alice.iv = tf.Variable("", trainable=False, name='iv')
    with Bob.name_scope:
        Bob.key = tf.Variable("", trainable=False, name='key')
        Bob.iv = tf.Variable("", trainable=False, name='iv')
    with Eve.name_scope:
        Eve.key = tf.Variable("", trainable=False, name='key')
        Eve.iv = tf.Variable("", trainable=False, name='iv')

    try:
        nb_updates = tf.Variable(
            0, name='updates-A-B', trainable=False, dtype=tf.uint16)
    except ValueError:
        # tf.function-decorated function tried to create variables
        # on non-first call.
        pass
    try:
        nb_eve_updates = tf.Variable(
            0, name='updates-E', trainable=False, dtype=tf.uint16)
    except ValueError:
        # tf.function-decorated function tried to create variables
        # on non-first call.
        pass
    tf.summary.experimental.set_step(0)
    start_time = perf_counter()

    def train_step():
        # Create random vector, X, with dimensions [K, N] and values {-1, 0, 1}
        X = tf.Variable(
            tf.random.uniform(
                (K, N), minval=-1, maxval=1 + 1, dtype=tf.int64),
            trainable=False,
            name='input'
        )

        update_rules = ['hebbian', 'anti_hebbian', 'random_walk']

        if update_rule == 'random':
            # use tf.random so that the same update rule is used for each
            # iteration across attacks

            # current_ur_index = tf.random.uniform(
            #     [],
            #     maxval=len(update_rules),
            #     dtype=tf.int32,
            #     name='iteration-ur-index'
            # )

            # current_ur_index = tf.random.fixed_unigram_candidate_sampler(
            #     tf.constant(  # true_classes
            #         [range(len(update_rules))],
            #         dtype=tf.int64
            #     ),
            #     len(update_rules),  # num_true
            #     1,  # num_sampled
            #     False,  # unique
            #     len(update_rules),  # range_max
            #     unigrams=[1. / len(update_rules)] * len(update_rules),
            #     name='iteration-ur-index'
            # )[0]

            current_ur_index = tf.random.uniform_candidate_sampler(
                [range(0, len(update_rules))],  # true_classes
                len(update_rules),  # num_true
                1,  # num_sampled,
                False,  # unique
                len(update_rules),  # range_max
                name='iteration-ur-index'
            )[0]
            current_update_rule = update_rules[current_ur_index.numpy().item()]
        else:
            current_update_rule = update_rule
        iterate(
            X,
            Alice, Bob, Eve, current_update_rule,
            nb_updates, nb_eve_updates,
            score, score_eve,
            key_length, iv_length
        )
        tf.summary.experimental.set_step(tf.cast(nb_updates, tf.int64))

    # instead of while, use for until L^4*K*N and break
    weights_A_B_equal = tf.reduce_all(tf.math.equal(
        Alice.w, Bob.w, name='weights-A-B-equal-elementwise'),
        name='weights-A-B-equal')
    # tf.while_loop(
    #     cond=lambda: score < 100. and not weights_A_B_equal,
    #     body=train_step,
    #     loop_vars=[],
    #     # since train_step doesn't return the same value for each call:
    #     parallel_iterations=1,
    #     swap_memory=True,
    #     name='train-step'
    # )
    while score < 100. and not weights_A_B_equal:
        train_step()

    end_time = perf_counter()
    training_time = end_time - start_time
    loss = (tf.math.sigmoid(training_time) + score_eve / 100.) / 2.
    key_length, iv_length = tf.constant(key_length), tf.constant(iv_length)
    if tf.math.equal(environ["MLENCRYPT_HPARAMS"], 'TRUE', name='use-hparams'):
        # creates scatterplot (in scalars) dashboard of metric vs steps
        tf.summary.scalar('training_time', training_time)
        tf.summary.scalar('eve_score', score_eve)
        tf.summary.scalar('loss', loss)

        # do this if hparams is enabled because the keys and IVs haven't been
        # calculated yet
        Alice.makeKey(key_length, iv_length)
        Bob.makeKey(key_length, iv_length)
        Eve.makeKey(key_length, iv_length)

    tf.print(
        "\n\n"
        "Training time = ", training_time, " seconds.\n"
        "Alice's key: ", Alice.key, " iv: ", Alice.iv, "\n",
        "Bob's key: ", Bob.key, " iv: ", Bob.iv, "\n",
        "Eve's key: ", Eve.key, " iv: ", Eve.iv,
        sep='',
        name='log-run-final'
    )
    return training_time, score_eve, loss
