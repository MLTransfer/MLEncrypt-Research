# -*- coding: utf-8 -*-
from tpm import TPM, ProbabilisticTPM, GeometricTPM
from datetime import datetime
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import math
from os import environ


tf.config.experimental_run_functions_eagerly(True)


def compute_overlap_matrix(N, L, w1, w2):
    """
    Computes the overlap matrix for the two vectors provided.

    Args:
        N: The number of inputs to the hidden units.
        L: The depth of the weights.
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
    f = tf.Variable(tf.zeros([2 * L + 1, 2 * L + 1], tf.float64))
    for i in tf.range(N):
        for j in tf.range(2 * L + 1):
            f[w1[i] + L, j].assign(f[w1[i] + L, j] + w2[i][j])
    return f


def compute_overlap_from_matrix(N, L, f):
    """
    Computes the overlap of two vectors from their overlap matrix.

    Args:
        N: The number of inputs per hidden unit.
        L: The depth of the weights.
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

    rho = tf.math.subtract(1,
                           tf.math.reduce_mean(
                               tf.math.divide(
                                   tf.cast(tf.math.abs(tf.math.subtract(
                                       TPM1.W, TPM2.W)), tf.float64),
                                   (2 * tf.cast(TPM1.L, tf.float64))
                                   )
                               ))
    tf.print(TPM1.name, TPM1.W)
    tf.print(TPM2.name, TPM2.W)

    epsilon = tf.math.multiply(tf.constant(
        tf.math.reciprocal(math.pi), tf.float32), tf.cast(tf.math.acos(rho),
                                                          tf.float32))

    if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
        with tf.name_scope(f'{TPM1.name} + {TPM2.name}'):
            tf.summary.scalar('sync', data=rho)
            tf.summary.scalar('generalization-error', data=epsilon)

    return rho


@tf.function
def run(update_rule, K, N, L, key_length=256, iv_length=128):
    print(
        f"Creating machines: K={K}, N={N}, L={L}, "
        + f"key-length={key_length}, "
        + f"initialization-vector-length={iv_length}, "
        + f"update-rule={update_rule}, "
        + f"attack={environ['MLENCRYPT_ATTACK']}")
    Alice = TPM('Alice', K, N, L)
    Bob = TPM('Bob', K, N, L)
    Eve = ProbabilisticTPM('Eve', K, N, L) if environ[
        "MLENCRYPT_ATTACK"] == 'PROBABILISTIC' else (
        GeometricTPM('Eve', K, N, L) if environ[
            "MLENCRYPT_ATTACK"] == 'GEOMETRIC' else
        TPM('Eve', K, N, L))

    # Synchronize weights
    nb_updates = tf.Variable(0, name='nb_updates',
                             trainable=False, dtype=tf.int32)
    nb_eve_updates = tf.Variable(0, name='nb_eve_updates',
                                 trainable=False, dtype=tf.int32)
    tf.summary.experimental.set_step(0)
    start_time = tf.timestamp(name='start_time')
    score = tf.Variable(0.0)  # synchronisation score of Alice and Bob
    score_eve = tf.Variable(0.0)  # synchronisation score of Alice and Eve

    while score < 100 and not tf.reduce_all(tf.math.equal(Alice.W, Bob.W)):
        # Create random vector [K, N]
        X = tf.Variable(tf.random.uniform(
            (K, N), minval=-1, maxval=1 + 1, dtype=tf.int64))

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
        elif (tauA == tauB != tauE) and environ[
                "MLENCRYPT_ATTACK"] == 'GEOMETRIC':
            Eve.update(tauA, update_rule, geometric=True)
            nb_eve_updates.assign_add(1, use_locking=True)
        if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
            with tf.name_scope(Eve.name):
                tf.summary.scalar('updates', data=nb_eve_updates)

        Alice_key, Alice_iv = Alice.makeKey(key_length, iv_length)
        Bob_key, Bob_iv = Bob.makeKey(key_length, iv_length)
        Eve_key, Eve_iv = Eve.makeKey(key_length, iv_length)

        score.assign(tf.cast(100 * sync_score(Alice, Bob), tf.float32))
        score_eve.assign(tf.cast(100 * sync_score(Alice, Eve), tf.float32))
        tf.print("\rSynchronization = ", score, "%   /  Updates = ",
                 nb_updates, " / Eve's updates = ", nb_eve_updates, sep='')

    end_time = tf.timestamp(name='end_time')
    time_taken = end_time - start_time
    if environ["MLENCRYPT_HPARAMS"] == 'TRUE':
        # creates scatterplot (in scalars) dashboard of metric vs steps
        tf.summary.scalar('time_taken', time_taken)
        tf.summary.scalar('eve_score', score_eve)

    tf.print("\nTime taken =", time_taken, "seconds.")
    tf.print("Alice's gen key =", Alice_key,
             "key :", Alice_key, "iv :", Alice_iv)
    tf.print("Bob's gen key =", Bob_key,
             "key :", Bob_key, "iv :", Bob_iv)
    tf.print("Eve's gen key =", Eve_key,
             "key :", Eve_key, "iv :", Eve_iv)

    if Alice_key == Bob_key and Alice_iv == Bob_iv:
        if tf.math.greater_equal(tf.cast(score_eve, tf.float64), tf.constant(
                100, tf.float64)):
            print("NOTE: Eve synced her machine with Alice's and Bob's!")
        else:
            tf.print("Eve's machine is ", score_eve,
                     "% synced with Alice's and Bob's and she did ",
                     nb_eve_updates, " updates.", sep='')
        return time_taken, score_eve

    else:
        print("ERROR: cipher impossible; Alice and Bob have different key/IV")

    print("\n\n")


def main():
    # less summaries are logged if MLENCRYPT_HPARAMS is True (for efficiency)
    environ["MLENCRYPT_HPARAMS"] = 'TRUE'

    if environ["MLENCRYPT_HPARAMS"] == 'TRUE':
        HP_K = hp.HParam('tpm_k', hp.IntInterval(4, 24))  # 8
        HP_N = hp.HParam('tpm_n', hp.IntInterval(4, 24))  # 12
        HP_L = hp.HParam('tpm_l', hp.IntInterval(4, 24))  # 4
        HP_UPDATE_RULE = hp.HParam('update_rule', hp.Discrete(
            ['hebbian', 'anti_hebbian', 'random_walk']))  # hebbian
        HP_ATTACK = hp.HParam('attack', hp.Discrete(
            ['none', 'geometric']))  # none

        hparams = [HP_UPDATE_RULE, HP_K, HP_N, HP_L, HP_ATTACK]

        logdir = 'logs/hparams/' + str(datetime.now())
        with tf.summary.create_file_writer(logdir).as_default():
            hp.hparams_config(
                hparams=hparams,
                metrics=[hp.Metric('time_taken', display_name='Time'),
                         hp.Metric('eve_score', display_name='Eve sync')]
            )
        session_num = 0
        for K in range(HP_K.domain.min_value, HP_K.domain.max_value):
            for N in range(HP_N.domain.min_value, HP_N.domain.max_value):
                for L in range(HP_L.domain.min_value, HP_L.domain.max_value):
                    for update_rule in HP_UPDATE_RULE.domain.values:
                        for attack in HP_ATTACK.domain.values:
                            current_hparams = {
                                HP_K: K,
                                HP_N: N,
                                HP_L: L,
                                HP_UPDATE_RULE: update_rule,
                                HP_ATTACK: attack
                            }
                            environ["MLENCRYPT_ATTACK"] = attack.upper()
                            run_name = "run-%d" % session_num
                            with tf.summary.create_file_writer(
                                    logdir + '/' + run_name).as_default():
                                run(update_rule, K, N, L)
                                hp.hparams(current_hparams)

                                session_num += 1
        print(f'Wrote log file to {logdir}')
    else:
        tf.summary.trace_on()
        # tf.python.eager.profiler.start()
        logdir = 'logs/' + str(datetime.now())
        with tf.summary.create_file_writer(logdir).as_default():
            K = 8
            N = 12
            L = 4
            update_rule = 'hebbian'  # or anti_hebbian or random_walk
            key_length = 256
            iv_length = 128
            environ["MLENCRYPT_ATTACK"] = 'NONE'

            run(update_rule, K, N, L, key_length, iv_length)
            tf.summary.trace_export("graph")
            # profiler_result = tf.python.eager.profiler.stop()
            # tf.python.eager.profiler.save(logdir, profiler_result)


if __name__ == "__main__":
    main()
