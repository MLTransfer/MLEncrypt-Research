from tpm import TPM
from datetime import datetime
import pyAesCrypt
from os import stat, remove, path
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp


tf.config.experimental_run_functions_eagerly(True)


def is_binary(file):
    return bool(open(file, 'rb').read(1024).translate(
        None, bytearray({7, 8, 9, 10, 12, 13, 27} | set(
            range(0x20, 0x100)) - {0x7f})))


@tf.function
def sync_score(TPM1, TPM2, L):
    '''sync_score
    Synchronize the score of 2 tree parity machines
    TPM1 - Tree Parity Machine 1
    TPM2 - Tree Parity Machine 2
    '''
    return tf.subtract(1,
                       tf.math.reduce_mean(
                           tf.math.abs(tf.subtract(TPM1.W, TPM2.W)) / (2 * L)
                          )
                       )


def aes_encrypt_file(is_dicom, input_file, output_file, Alice_key, key_length):
    if is_dicom:
        with open(input_file, "rb") as fIn:
            with open(output_file, "wb") as fOut:
                pyAesCrypt.encryptStream(fIn, fOut, Alice_key, key_length)
    else:
        pyAesCrypt.encryptFile(input_file, output_file, Alice_key, key_length)


def aes_decrypt_file(is_dicom, input_file, output_file, Alice_key, key_length):
    if is_dicom:
        with open(input_file, "rb") as fIn:
            try:
                with open(output_file, "wb") as fOut:
                    # decrypt file stream
                    pyAesCrypt.decryptStream(
                        fIn, fOut, Alice_key, key_length,
                        stat(input_file).st_size)
            except ValueError:
                # remove output file on error
                remove(output_file)
    else:
        pyAesCrypt.decryptFile(input_file, output_file,
                               Alice_key, key_length)


def run(input_file, update_rule, output_file, K, N, L, key_length, iv_length):
    # Create TPM for Alice, Bob and Eve. Eve eavesdrops on Alice and Bob
    print("Creating machines : K=" + str(K) + ", N=" + str(N) + ", L=" + str(L)
          + ", key-length=" + str(key_length)
          + ", initialization-vector-length=" + str(iv_length))
    Alice = TPM('Alice', K, N, L)
    Bob = TPM('Bob', K, N, L)
    Eve = TPM('Eve', K, N, L)

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

    while score < 100:
        # Create random vector [K, N]
        X = tf.Variable(tf.random.uniform(
            (K, N), minval=-L, maxval=L + 1, dtype=tf.int64))

        # compute outputs of TPMs
        with tf.name_scope('Alice'):
            tauA = Alice.get_output(X)
            tf.summary.scalar('tau', data=tauA)
        with tf.name_scope('Bob'):
            tauB = Bob.get_output(X)
            tf.summary.scalar('tau', data=tauB)
        with tf.name_scope('Eve'):
            tauE = Eve.get_output(X)
            tf.summary.scalar('tau', data=tauE)

        Alice.update(tauB, update_rule)
        Bob.update(tauA, update_rule)
        nb_updates.assign_add(1, use_locking=True)
        tf.summary.experimental.set_step(tf.cast(nb_updates, tf.int64))

        # Eve would update only if tauA = tauB = tauE
        if tauA == tauB == tauE:
            Eve.update(tauA, update_rule)
            nb_eve_updates.assign_add(1, use_locking=True)
        with tf.name_scope('Eve'):
            tf.summary.scalar('updates', data=nb_eve_updates)

        Alice_key, Alice_iv = Alice.makeKey(key_length, iv_length)
        Bob_key, Bob_iv = Bob.makeKey(key_length, iv_length)
        Eve_key, Eve_iv = Eve.makeKey(key_length, iv_length)

        with tf.name_scope('sync'):
            score.assign(tf.cast(100 * sync_score(Alice, Bob, L), tf.float32))
            tf.summary.scalar('Alice + Bob', data=score)
            sync_history.append(score)  # plot purpose
            score_eve.assign(
                tf.cast(100 * sync_score(Alice, Eve, L), tf.float32))
            tf.summary.scalar('Alice + Eve', data=score_eve)
            sync_history_eve.append(score_eve)  # plot purpose
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
        is_file_binary = is_binary(input_file)
        decrypt_file = "decipher." + path.splitext(input_file)[1][1:]
        # cipher with AES
        aes_encrypt_file(is_file_binary, input_file, output_file,
                         Alice_key, key_length)
        # decipher with AES
        aes_decrypt_file(is_file_binary, output_file,
                         decrypt_file, Alice_key, key_length)
        tf.print("encryption and decryption with aes",
                 key_length, " done.", sep='')
        if score_eve >= 100:
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
    use_hparams = True
    input_file = 'test.dcm'  # or test.txt
    output_file = 'out.enc'

    if use_hparams:
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
                                with tf.summary.create_file_writer(logdir + '/' + run_name).as_default():
                                    run(input_file, update_rule, output_file,
                                        K, N, L, key_length, iv_length)
                                    hp.hparams(current_hparams)

                                    session_num += 1
        print(f'Wrote log file to {logdir}')
    else:
        logdir = 'logs/' + str(datetime.now())
        summary_writer = tf.summary.create_file_writer(logdir)
        summary_writer.set_as_default()

        K = 8
        N = 12
        L = 4
        update_rule = 'hebbian'  # or anti_hebbian or random_walk
        key_length = 256
        iv_length = 128

        run(input_file, update_rule, output_file, K, N, L, key_length,
            iv_length)


if __name__ == "__main__":
    main()
