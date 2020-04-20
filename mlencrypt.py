# -*- coding: utf-8 -*-
from tpm import TPM, ProbabilisticTPM, GeometricTPM
from tpm import tb_summary, tb_heatmap, tb_boxplot

from os import environ
from time import perf_counter
from random import choice as random_from_array

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
    # rho = tf.Variable(0., dtype=tf.float64)
    #
    # for i in tf.range(TPM1.K):
    #     f = compute_overlap_matrix(TPM1.N, TPM1.L, TPM1.w[i], TPM2.w[i])
    #     rho.assign_add(tf.math.divide(compute_overlap_from_matrix(
    #         TPM1.N, TPM1.L, f), tf.cast(TPM1.K, tf.float64)))

    rho = tf.math.subtract(
        tf.cast(1., tf.float64),
        tf.math.reduce_mean(
            tf.math.divide(
                tf.cast(tf.math.abs(tf.math.subtract(
                    TPM1.w,
                    TPM2.w
                )), tf.float64),
                (2. * tf.cast(TPM1.L, tf.float64))
            )
        )
    )

    epsilon = tf.math.multiply(
        tf.constant(1. / pi, tf.float32),
        tf.cast(tf.math.acos(rho), tf.float32)
    )

    if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
        # TODO: 'Alice + Bob' is not a valid scope name

        # it's not that f-strings are evaluated at runtime, since
        # with tf.name_scope(TPM1.name + ' and ' + TPM2.name):
        # didn't work

        # with tf.name_scope(f'{TPM1.name} and {TPM2.name}'):
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

    if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
        # TODO: don't refer to variables outside of the method scope,
        # add them as arguments (maybe tf.numpy_function) will help
        def log_inputs(inputs):
            tb_summary('inputs', inputs)
            # TODO: uncomment this:
            # K, N = Alice.K, Alice.N
            # hpaxis, ipaxis = tf.range(1, K + 1), tf.range(1, N + 1)
            # tb_heatmap('inputs', X, ipaxis, hpaxis)
            # tb_boxplot('inputs', X, hpaxis)
        tf.py_function(log_inputs, [X], [])

    # compute outputs of TPMs
    with tf.name_scope(Alice.name):
        tauA = Alice.get_output(X)
        if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
            tf.summary.scalar('tau', data=tauA)
    with tf.name_scope(Bob.name):
        tauB = Bob.get_output(X)
        if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
            tf.summary.scalar('tau', data=tauB)
    with tf.name_scope(Eve.name):
        tauE = Eve.get_output(X)
        if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
            tf.summary.scalar('tau', data=tauE)

    Alice.update(tauB, update_rule)
    Bob.update(tauA, update_rule)
    nb_updates.assign_add(1)
    tf.summary.experimental.set_step(tf.cast(nb_updates, tf.int64))

    # if tauA equals tauB and tauB equals tauE then tauA must equal tauE
    # due to the associative law of boolean multiplication
    if tf.math.equal(tauA, tauB) and tf.math.equal(tauB, tauE):
        Eve.update(tauA, update_rule)
        nb_eve_updates.assign_add(1)
    # if tauA equals tauB and tauB does not tauE
    # then tauA does not equal tauE
    elif (tf.math.equal(tauA, tauB) and tf.math.not_equal(tauA, tauE)) \
            and Eve.type == 'geometric':
        Eve.update(tauA, update_rule)
        nb_eve_updates.assign_add(1)
    if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
        with tf.name_scope(Eve.name):
            tf.summary.scalar('updates', data=nb_eve_updates)

    # if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
    #     Alice_key, Alice_iv = Alice.makeKey(key_length, iv_length)
    #     Bob_key, Bob_iv = Bob.makeKey(key_length, iv_length)
    #     Eve_key, Eve_iv = Eve.makeKey(key_length, iv_length)

    score.assign(tf.cast(100. * sync_score(Alice, Bob), tf.float32))
    score_eve.assign(tf.cast(100. * sync_score(Alice, Eve), tf.float32))
    if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
        # log adversary score for Bob's weights
        sync_score(Bob, Eve)

    tf.print(
        "\rUpdate rule = ", update_rule, " / "
        "Synchronization = ", score, "% / ",
        nb_updates, " Updates (Alice) / ",
        nb_eve_updates, " Updates (Eve)",
        sep=''
    )


# @tf.function
def run(
    update_rule, K, N, L,
    attack,
    initial_weights,
    key_length=256, iv_length=128
):
    tf.print("\n\n")
    tf.print(
        f"Creating machines: K={K}, N={N}, L={L}, "
        f"update-rule={update_rule}, "
        f"attack={attack}"
    )
    Alice = TPM('Alice', K, N, L, initial_weights['Alice'])

    Bob = TPM('Bob', K, N, L, initial_weights['Bob'])

    # TODO: load class programatically
    # https://stackoverflow.com/a/4821120/7127932
    initial_weights_eve = initial_weights['Eve']
    Eve = ProbabilisticTPM(
        'Eve', K, N, L, initial_weights_eve
    ) if attack == 'probabilistic' \
        else (GeometricTPM(
            'Eve', K, N, L, initial_weights_eve
        ) if attack == 'geometric'
        else TPM(
            'Eve', K, N, L, initial_weights_eve
        ))

    try:
        # synchronization score of Alice and Bob
        score = tf.Variable(0.0, trainable=False)
    except ValueError:
        # tf.function-decorated function tried to create variables
        # on non-first call.
        score = 0.
    try:
        # synchronization score of Alice and Eve
        score_eve = tf.Variable(0.0, trainable=False)
    except ValueError:
        # tf.function-decorated function tried to create variables
        # on non-first call.
        score_eve = 0.

    # https://www.tensorflow.org/tutorials/customization/performance#zero_iterations
    Alice_key, Alice_iv = tf.constant(""), tf.constant("")
    Bob_key, Bob_iv = tf.constant(""), tf.constant("")
    Eve_key, Eve_iv = tf.constant(""), tf.constant("")

    try:
        nb_updates = tf.Variable(
            0, name='nb_updates', trainable=False, dtype=tf.uint16)
    except ValueError:
        # tf.function-decorated function tried to create variables
        # on non-first call.
        pass
    try:
        nb_eve_updates = tf.Variable(
            0, name='nb_eve_updates', trainable=False, dtype=tf.uint16)
    except ValueError:
        # tf.function-decorated function tried to create variables
        # on non-first call.
        pass
    tf.summary.experimental.set_step(0)
    start_time = perf_counter()

    # instead of while, use for until L^4*K*N and break
    while score < 100. and not tf.reduce_all(tf.math.equal(Alice.w, Bob.w)):
        # Create random vector, X, with dimensions [K, N] and values {-1, 0, 1}
        X = tf.Variable(
            tf.random.uniform(
                (K, N), minval=-1, maxval=1 + 1, dtype=tf.int64),
            trainable=False
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

    end_time = perf_counter()
    training_time = end_time - start_time
    loss = (tf.math.sigmoid(training_time) + score_eve / 100.) / 2.
    key_length, iv_length = tf.constant(key_length), tf.constant(iv_length)
    if environ["MLENCRYPT_HPARAMS"] == 'TRUE':
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

    tf.print()
    tf.print("Training time =", training_time, "seconds.")
    tf.print("Alice's key :", Alice_key, "iv :", Alice_iv)
    tf.print("Bob's key :", Bob_key, "iv :", Bob_iv)
    tf.print("Eve's key :", Eve_key, "iv :", Eve_iv)

    if tf.math.equal(Alice_key, Bob_key) and tf.math.equal(Alice_iv, Bob_iv):
        tf.print("Eve's machine is ", score_eve,
                 "% synced with Alice's and Bob's and she did ",
                 nb_eve_updates, " updates.", sep='')

    # TODO: are these returns value valid if the keys don't match?
    # this is here because we have to return something in tf graph mode
    return training_time, score_eve, loss
