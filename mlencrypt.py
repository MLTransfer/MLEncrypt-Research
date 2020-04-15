# -*- coding: utf-8 -*-
from tpm import TPM, ProbabilisticTPM, GeometricTPM
from tpm import tb_summary, tb_heatmap, tb_boxplot

from os import environ
from time import perf_counter

import tensorflow as tf
from math import pi


tf.config.experimental_run_functions_eagerly(True)


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


@tf.function
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
    #     f = compute_overlap_matrix(TPM1.N, TPM1.L, TPM1.W[i], TPM2.W[i])
    #     rho.assign_add(tf.math.divide(compute_overlap_from_matrix(
    #         TPM1.N, TPM1.L, f), tf.cast(TPM1.K, tf.float64)))

    rho = tf.math.subtract(
        1.,
        tf.math.reduce_mean(
            tf.math.divide(
                tf.cast(tf.math.abs(tf.math.subtract(
                    TPM1.W,
                    TPM2.W
                )), tf.float64),
                (2. * tf.cast(TPM1.L, tf.float64))
            )
        )
    )

    epsilon = tf.math.multiply(
        tf.constant(tf.math.reciprocal(pi), tf.float32),
        tf.cast(tf.math.acos(rho), tf.float32)
    )

    if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
        with tf.name_scope(f'{TPM1.name} + {TPM2.name}'):
            tf.summary.scalar('sync', data=rho)
            tf.summary.scalar('generalization-error', data=epsilon)

    return rho


@tf.function
def run(
    update_rule,
    K,
    N,
    L,
    attack,
    initial_weights=None,
    key_length=256,
    iv_length=128
):
    print(
        f"Creating machines: K={K}, N={N}, L={L}, "
        f"update-rule={update_rule}, "
        f"attack={attack}"
    )
    if initial_weights:
        init_weights_alice = tf.Variable(
            initial_weights['Alice'], trainable=True)
        Alice = TPM('Alice', K, N, L, initial_weights=init_weights_alice)

        init_weights_bob = tf.Variable(initial_weights['Bob'], trainable=True)
        Bob = TPM('Bob', K, N, L, initial_weights=init_weights_bob)

        # TODO: initialize class programatically
        # https://stackoverflow.com/a/4821120/7127932
        init_weights_eve = tf.Variable(initial_weights['Eve'], trainable=True)
        if attack == 'probabilistic':
            Eve = ProbabilisticTPM(
                'Eve', K, N, L, initial_weights=init_weights_eve)
        elif attack == 'geometric':
            Eve = GeometricTPM(
                'Eve', K, N, L, initial_weights=init_weights_eve)
        else:
            Eve = TPM('Eve', K, N, L, initial_weights=init_weights_eve)
    else:
        Alice = TPM('Alice', K, N, L)

        Bob = TPM('Bob', K, N, L)

        # TODO: load class programatically
        # https://stackoverflow.com/a/4821120/7127932
        Eve = ProbabilisticTPM('Eve', K, N, L) if attack == 'probabilistic' \
            else (GeometricTPM('Eve', K, N, L) if attack == 'geometric'
                  else TPM('Eve', K, N, L))

    nb_updates = tf.Variable(0, name='nb_updates',
                             trainable=False, dtype=tf.uint16)
    nb_eve_updates = tf.Variable(0, name='nb_eve_updates',
                                 trainable=False, dtype=tf.uint16)
    tf.summary.experimental.set_step(0)
    start_time = perf_counter()
    score = tf.Variable(0.0)  # synchronisation score of Alice and Bob
    score_eve = tf.Variable(0.0)  # synchronisation score of Alice and Eve

    # instead of while, use for until L^4*K*N
    while score < 100 and not tf.reduce_all(tf.math.equal(Alice.W, Bob.W)):
        # Create random vector [K, N]
        X = tf.Variable(tf.random.uniform(
            (K, N), minval=-1, maxval=1 + 1, dtype=tf.int64))
        if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
            tb_summary('inputs', X)
            hpaxis, ipaxis = tf.range(1, K + 1), tf.range(1, N + 1)
            tb_heatmap('inputs', X, ipaxis, hpaxis)
            tb_boxplot('inputs', X, hpaxis)

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
        nb_updates.assign_add(1, use_locking=True)
        tf.summary.experimental.set_step(tf.cast(nb_updates, tf.int64))

        if tauA == tauB == tauE:
            Eve.update(tauA, update_rule)
            nb_eve_updates.assign_add(1, use_locking=True)
        elif (tauA == tauB != tauE) and Eve.type == 'geometric':
            Eve.update(tauA, update_rule)
            nb_eve_updates.assign_add(1, use_locking=True)
        if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
            with tf.name_scope(Eve.name):
                tf.summary.scalar('updates', data=nb_eve_updates)

        if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
            Alice_key, Alice_iv = Alice.makeKey(key_length, iv_length)
            Bob_key, Bob_iv = Bob.makeKey(key_length, iv_length)
            Eve_key, Eve_iv = Eve.makeKey(key_length, iv_length)

        score.assign(tf.cast(100. * sync_score(Alice, Bob), tf.float32))
        score_eve.assign(tf.cast(100. * sync_score(Alice, Eve), tf.float32))
        if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
            # log adversary score for Bob's weights
            sync_score(Bob, Eve)

        tf.print(
            f"\rSynchronization = {score.numpy()}% / "
            f"{nb_updates.numpy()} Updates (Alice) / "
            f"{nb_eve_updates.numpy()} Updates (Eve)"
        )

    end_time = perf_counter()
    training_time = end_time - start_time
    loss = (tf.math.sigmoid(training_time) + score_eve / 100.) / 2.
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

    tf.print("\nTraining time =", training_time, "seconds.")
    tf.print("Alice's gen key =", Alice_key,
             "key :", Alice_key, "iv :", Alice_iv)
    tf.print("Bob's gen key =", Bob_key, "key :", Bob_key, "iv :", Bob_iv)
    tf.print("Eve's gen key =", Eve_key, "key :", Eve_key, "iv :", Eve_iv)

    if Alice_key == Bob_key and Alice_iv == Bob_iv:
        tf.print("Eve's machine is ", score_eve,
                 "% synced with Alice's and Bob's and she did ",
                 nb_eve_updates, " updates.", sep='')
        return training_time, score_eve, loss

    else:
        print("ERROR: cipher impossible; Alice and Bob have different key/IV")

    print("\n\n")
