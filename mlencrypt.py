# -*- coding: utf-8 -*-
from tpm import TPM
from tpm import tb_summary  # , tb_heatmap, tb_boxplot

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
    # adapted from:
    # https://github.com/tensorflow/tensorflow/blob/f270180a6caa8693f2b2888ac7e6b8e69c4feaa8/tensorflow/python/keras/losses.py#L1073-L1093
    # TODO: am I using experimental_implements correctly?
    @tf.function(experimental_implements="cosine_similarity")
    def cosine_similarity(y_true, y_pred):
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
        y_true_norm = tf.math.l2_normalize(
            tf.cast(tf.reshape(y_true, [-1]), tf.float64), axis=-1)
        y_pred_norm = tf.math.l2_normalize(
            tf.cast(tf.reshape(y_pred, [-1]), tf.float64), axis=-1)
        return tf.math.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
    rho = cosine_similarity(TPM1.w, TPM2.w)

    tpm1_id, tpm2_id = TPM1.name[0], TPM2.name[0]
    # the generalization error, epsilon, is the probability that a repulsive
    # step occurs if two corresponding hidden units have different sigma
    # (Ruttor, 2006).
    epsilon = tf.math.multiply(
        tf.constant(1. / pi, tf.float32, name='reciprocal-pi'),
        tf.cast(tf.math.acos(rho), tf.float32,
                name=f'angle-{tpm1_id}-{tpm2_id}'),
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

    # TODO: make update_rule an attribute of the TPMs
    # TODO: can Alice and Bob use different update rules?

    no_hparams = tf.math.equal(
        tf.constant(environ["MLENCRYPT_HPARAMS"], name='setting-hparams'),
        tf.constant('FALSE', name='false'),
        name='do-not-use-hparams'
    )

    if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
        tb_summary('inputs', X)
        # TODO: don't refer to variables outside of the method scope,
        # add them as arguments (maybe tf.numpy_function) will help

        def log_inputs(inputs):
            # TODO: uncomment this:
            # K, N = Alice.K, Alice.N
            # hpaxis, ipaxis = tf.range(1, K + 1), tf.range(1, N + 1)
            # tb_heatmap('inputs', X, ipaxis, hpaxis)
            # tb_boxplot('inputs', X, hpaxis)
            pass
        tf.py_function(log_inputs, [X], [], name='tb-images-inputs')

    # compute outputs of TPMs
    with tf.name_scope(Alice.name) as scope:
        tauA = tf.convert_to_tensor(Alice.get_output(X), name=scope)

        def log_tauA():
            tf.summary.scalar('tau', data=tauA)
        tf.cond(no_hparams, true_fn=log_tauA,
                false_fn=lambda: None, name='tb-tau-A')
    with tf.name_scope(Bob.name):
        tauB = tf.convert_to_tensor(Bob.get_output(X), name=scope)

        def log_tauB():
            tf.summary.scalar('tau', data=tauB)
        tf.cond(no_hparams, true_fn=log_tauB,
                false_fn=lambda: None, name='tb-tau-B')
    with tf.name_scope(Eve.name):
        tauE = tf.convert_to_tensor(Eve.get_output(X), name=scope)

        def log_tauE():
            tf.summary.scalar('tau', data=tauE)
        tf.cond(no_hparams, true_fn=log_tauE,
                false_fn=lambda: None, name='tb-tau-E')

    Alice.update(tauB, update_rule)
    Bob.update(tauA, update_rule)
    nb_updates.assign_add(1)
    tf.summary.experimental.set_step(tf.cast(nb_updates, tf.int64))

    # if tauA equals tauB and tauB equals tauE then tauA must equal tauE
    # due to the associative law of boolean multiplication
    tau_A_B_equal = tf.math.equal(tauA, tauB, name='iteration-tau-A-B-equal')
    tau_B_E_equal = tf.math.equal(tauB, tauE, name='iteration-tau-B-E-equal')
    tau_A_E_not_equal = tf.math.not_equal(
        tauA, tauE, name='iteration-tau-A-E-not-equal')

    def update_E():
        Eve.update(tauA, update_rule)
        nb_eve_updates.assign_add(1)
    tf.cond(
        ((tau_A_B_equal and tau_B_E_equal) or (
            tau_A_B_equal and tau_A_E_not_equal and Eve.type == 'geometric')),
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

    # if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
    #     Alice_key, Alice_iv = Alice.makeKey(key_length, iv_length)
    #     Bob_key, Bob_iv = Bob.makeKey(key_length, iv_length)
    #     Eve_key, Eve_iv = Eve.makeKey(key_length, iv_length)

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
        "Synchronization = ", score, "% / ",
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

    initial_weights_eve = initial_weights['Eve']
    tpm_mod = import_module('tpm')
    if attack == 'probabilistic':
        Eve_class_name = 'ProbabilisticTPM'
    elif attack == 'geometric':
        Eve_class_name = 'GeometricTPM'
    else:
        Eve_class_name = 'TPM'
    Eve_class = getattr(tpm_mod, Eve_class_name)
    Eve = Eve_class('Eve', K, N, L, initial_weights_eve)

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
    Alice_key = tf.constant("", name='key-A')
    Alice_iv = tf.constant("", name='IV-A')
    Bob_key = tf.constant("", name='key-B')
    Bob_iv = tf.constant("", name='IV-B')
    Eve_key = tf.constant("", name='key-E')
    Eve_iv = tf.constant("", name='IV-E')

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
            # use tensorflow so that the same update rule is used for each
            # iteration across attacks
            # TODO: use tf.random.categorical?
            current_ur_index = tf.random.uniform(
                [],
                maxval=len(update_rules),
                dtype=tf.int32
            )
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

        # do this if hparams was enabled because the output keys and IVs
        # haven't been defined yet
        Alice_key, Alice_iv = Alice.makeKey(key_length, iv_length)
        Bob_key, Bob_iv = Bob.makeKey(key_length, iv_length)
        Eve_key, Eve_iv = Eve.makeKey(key_length, iv_length)
    else:
        # if hparams is not enabled, the keys and IVs have already been created
        # Alice_key, Alice_iv = Alice.key, Alice.iv
        # Bob_key, Bob_iv = Bob.key, Bob.iv
        # Eve_key, Eve_iv = Eve.key, Eve.iv
        Alice_key, Alice_iv = Alice.makeKey(key_length, iv_length)
        Bob_key, Bob_iv = Bob.makeKey(key_length, iv_length)
        Eve_key, Eve_iv = Eve.makeKey(key_length, iv_length)

    tf.print(
        "\n\n"
        "Training time = ", training_time, " seconds.\n"
        "Alice's key: ", Alice_key, " iv: ", Alice_iv, "\n",
        "Bob's key: ", Bob_key, " iv: ", Bob_iv, "\n",
        "Eve's key: ", Eve_key, " iv: ", Eve_iv,
        sep='',
        name='log-run-final'
    )

    keys_equal = tf.math.equal(Alice_key, Bob_key, name='final-keys-equal')
    ivs_equal = tf.math.equal(Alice_iv, Bob_iv, name='final-IVs-equal')

    def log_run_eve():
        tf.print("Eve's machine is ", score_eve,
                 "% synced with Alice's and Bob's and she did ",
                 nb_eve_updates, " updates.", sep='', name='log-run-eve')
    tf.cond(keys_equal and ivs_equal, true_fn=log_run_eve,
            false_fn=lambda: None, name='log-run-E')

    # TODO: are these returns value valid if the keys don't match?
    # this is here because we have to return something in tf graph mode
    return training_time, score_eve, loss
