from tpm import TPM, ProbabilisticTPM, GeometricTPM
from datetime import datetime
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import math
from os import environ


tf.config.experimental_run_functions_eagerly(True)


@tf.function
def sync_score(TPM1, TPM2):
    """
    Args:
        TPM1: Tree Parity Machine 1. Has same parameters as TPM2.
        TPM2: Tree Parity Machine 2. Has same parameters as TPM1.
    Returns:
        The synchronization score between TPM1 and TPM2.
    """

    # Q_tpm1, Q_tpm2, and R are standard order parameters for online learning
    # Q_tpm1 = tf.math.divide(tf.matmul(TPM1.W, TPM1.W, transpose_a=True), N)
    # Q_tpm2 = tf.math.divide(tf.matmul(TPM2.W, TPM2.W, transpose_a=True), N)
    # R = tf.math.divide(tf.matmul(TPM1.W, TPM2.W, transpose_a=True), N)
    # rho = tf.math.divide(R, tf.matrix_square_root(
    #     tf.matmul(Q_tpm1, Q_tpm2)))

    # rho = tf.math.divide(tf.cast(tf.matmul(TPM1.W, TPM2.W, transpose_a=True), tf.float32),
    #                      tf.multiply(
    #     tf.matrix_square_root(
    #         tf.cast(tf.matmul(TPM1.W, TPM1.W, transpose_a=True), tf.float32)),
    #     tf.matrix_square_root(
    #         tf.cast(tf.matmul(TPM2.W, TPM2.W, transpose_a=True), tf.float32)),
    #     ))

    rho = tf.subtract(1,
                      tf.math.reduce_mean(
                          tf.divide(
                              tf.cast(tf.math.abs(tf.subtract(
                                  TPM1.W, TPM2.W)), tf.float64),
                              (2 * tf.cast(TPM1.L, tf.float64))
                          )
                      ))
    epsilon = tf.multiply(tf.constant(
        tf.math.reciprocal(math.pi), tf.float32), tf.cast(tf.acos(rho),
                                                          tf.float32))

    if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
        with tf.name_scope(f'{TPM1.name} + {TPM2.name}'):
            tf.summary.scalar('sync', data=rho)
            tf.summary.scalar('generation-error', data=epsilon)

    return rho


def run(input_file, update_rule, output_file, K, N, L, key_length, iv_length):
    # Create TPM for Alice, Bob and Eve. Eve eavesdrops on Alice and Bob
    print(
        f"Creating machines: K={K}, N={N}, L={L}, key-length={key_length}, "
        + f"initialization-vector-length={iv_length}")
    Alice = TPM('Alice', K, N, L)
    Bob = TPM('Bob', K, N, L)
    Eve = ProbabilisticTPM('Eve', K, N, L) if environ["MLENCRYPT_PROBABILISTIC"
                                                      ] == 'TRUE' else (
                                                      GeometricTPM('Eve', K, N,
                                                                   L) if
                                                                   environ[
                                                          "MLENCRYPT_GEOMETRIC"
                                                          ] == 'TRUE' else TPM(
                                                          'Eve', K, N, L))

    # Synchronize weights
    nb_updates = tf.Variable(0, name='nb_updates',
                             trainable=False, dtype=tf.int32)
    nb_eve_updates = tf.Variable(0, name='nb_eve_updates',
                                 trainable=False, dtype=tf.int32)
    tf.summary.experimental.set_step(0)
    start_time = tf.timestamp(name='start_time')
    sync_history = []
    sync_history_eve = []
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

        # Eve would update only if tauA = tauB = tauE
        if tauA == tauB == tauE:
            Eve.update(tauA, update_rule)
            nb_eve_updates.assign_add(1, use_locking=True)
        if environ["MLENCRYPT_HPARAMS"] == 'FALSE':
            with tf.name_scope(Eve.name):
                tf.summary.scalar('updates', data=nb_eve_updates)

        Alice_key, Alice_iv = Alice.makeKey(key_length, iv_length)
        Bob_key, Bob_iv = Bob.makeKey(key_length, iv_length)
        Eve_key, Eve_iv = Eve.makeKey(key_length, iv_length)

        score.assign(tf.cast(100 * sync_score(Alice, Bob), tf.float32))
        sync_history.append(score)
        score_eve.assign(tf.cast(100 * sync_score(Alice, Eve), tf.float32))
        sync_history_eve.append(score_eve)
        tf.print("\rSynchronization = ", score, "%   /  Updates = ",
                 nb_updates, " / Eve's updates = ", nb_eve_updates, sep='')

    end_time = tf.timestamp(name='end_time')
    time_taken = end_time - start_time
    tf.summary.scalar('time_taken', time_taken)
    tf.summary.scalar('eve_score', score_eve)

    # results
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

    print("\r\n\r\n")


def main():
    # less summaries are logged if MLENCRYPT_HPARAMS is True
    environ["MLENCRYPT_HPARAMS"] = 'FALSE'
    # only one of MLENCRYPT_PROBABILISTIC and MLENCRYPT_GEOMETRIC may be True
    environ["MLENCRYPT_PROBABILISTIC"] = 'FALSE'
    environ["MLENCRYPT_GEOMETRIC"] = 'TRUE'
    input_file = 'test.dcm'  # or test.txt
    output_file = 'out.enc'

    if environ["MLENCRYPT_HPARAMS"] == 'TRUE':
        HP_K = hp.HParam('tpm_k', hp.IntInterval(4, 24))  # 8
        HP_N = hp.HParam('tpm_n', hp.IntInterval(4, 24))  # 12
        HP_L = hp.HParam('tpm_l', hp.IntInterval(4, 24))  # 4
        HP_UPDATE_RULE = hp.HParam('update_rule', hp.Discrete(
            ['hebbian', 'anti_hebbian', 'random_walk']))  # hebbian
        HP_KEY_LENGTH = hp.HParam(
            'key_length', hp.Discrete([128, 192, 256]))  # 256
        HP_IV_LENGTH = hp.HParam('iv_length', hp.Discrete(
            range(0, 256 + 1, 4)))  # 128

        hparams = [HP_UPDATE_RULE, HP_K, HP_N,
                   HP_L, HP_KEY_LENGTH, HP_IV_LENGTH]

        logdir = 'logs/hparams/' + str(datetime.now())
        with tf.summary.create_file_writer(logdir).as_default():
            hp.hparams_config(
                hparams=hparams,
                metrics=[hp.Metric('time_taken', display_name='Time'),
                         hp.Metric('eve_score', display_name='Eve sync')]
            )
        session_num = 0
        for K in (HP_K.domain.min_value, HP_K.domain.max_value):
            for N in (HP_N.domain.min_value, HP_N.domain.max_value):
                for L in (HP_L.domain.min_value, HP_L.domain.max_value):
                    for update_rule in HP_UPDATE_RULE.domain.values:
                        for key_length in HP_KEY_LENGTH.domain.values:
                            for iv_length in HP_IV_LENGTH.domain.values:
                                current_hparams = {
                                    HP_K: K,
                                    HP_N: N,
                                    HP_L: L,
                                    HP_UPDATE_RULE: update_rule,
                                    HP_KEY_LENGTH: key_length,
                                    HP_IV_LENGTH: iv_length
                                }
                                run_name = "run-%d" % session_num
                                with tf.summary.create_file_writer(
                                        logdir + '/' + run_name).as_default():
                                    run(input_file, update_rule, output_file,
                                        K, N, L, key_length, iv_length)
                                    hp.hparams(current_hparams)

                                    session_num += 1
        print(f'Wrote log file to {logdir}')
    else:
        logdir = 'logs/' + str(datetime.now())
        summary_writer = tf.summary.create_file_writer(logdir)
        summary_writer.set_as_default()
        tf.summary.trace_on()

        K = 8
        N = 12
        L = 4
        update_rule = 'hebbian'  # or anti_hebbian or random_walk
        key_length = 256
        iv_length = 128

        run(input_file, update_rule, output_file, K, N, L, key_length,
            iv_length)
        tf.summary.trace_export("graph")


if __name__ == "__main__":
    main()
