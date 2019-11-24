from tpm import TPM
import pydicom
# import matplotlib.pyplot as plt
from datetime import datetime
import pyAesCrypt
from os import stat, remove
import click
import tensorflow as tf


tf.config.experimental_run_functions_eagerly(True)


@tf.function
def sync_score(TPM1, TPM2, L):
    '''sync_score
    Synchronize the score of 2 tree parity machines
    TPM1 - Tree Parity Machine 1
    TPM2 - Tree Parity Machine 2
    '''
    return tf.subtract(1, tf.math.reduce_mean(tf.math.abs(tf.subtract(TPM1.W, TPM2.W)) / (2 * L)))


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


class ChoiceType(click.Choice):
    def __init__(self, typemap):
        super(ChoiceType, self).__init__(typemap.keys())
        self.typemap = typemap

    def convert(self, value, param, ctx):
        value = super(ChoiceType, self).convert(value, param, ctx)
        return self.typemap[value]


@click.command()
@click.option('-i', '--input-file', required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('-r', '--update-rule', default='hebbian', type=click.Choice(['hebbian', 'anti_hebbian', 'random_walk'], case_sensitive=False))
@click.option('-o', '--output-file', default='out.enc', type=click.Path(dir_okay=False, writable=True))
@click.option('-k', default=8, type=int)
@click.option('-n', default=12, type=int)
@click.option('-l', default=4, type=int)
@click.option('-key', '--key-length', default='256', type=ChoiceType({str(x): x for x in [128, 192, 256]}))
@click.option('-iv', '--iv-length', default='128', type=ChoiceType({str(x): x for x in range(0, 256 + 1, 4)}))
def main(input_file, update_rule, output_file, k, n, l, key_length, iv_length):
    logdir = 'logs/' + str(datetime.now())
    summary_writer = tf.summary.create_file_writer(logdir)
    summary_writer.set_as_default()

    # Tree Parity Machine parameters
    K, N, L = k, n, l

    # Create TPM for Alice, Bob and Eve. Eve eavesdrops communication of Alice and Bob
    print("Creating machines : K=" + str(K) + ", N=" + str(N) + ", L="
          + str(L) + ", key-length=" + str(key_length) + ", initialization-vector-length=" + str(iv_length))
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
    sync_history = []  # plot purpose
    sync_history_eve = []  # plot purpose
    score = tf.Variable(0.0)  # synchronisation score of Alice and Bob

    while score < 100:
        # Create random vector [K, N]
        X = tf.Variable(tf.random.uniform(
            (K, N), minval=-l, maxval=l + 1))

        # compute outputs of TPMs
        tauA = Alice.get_output(X)
        tauB = Bob.get_output(X)
        tauE = Eve.get_output(X)

        tf.summary.scalar('Alice\'s tau', data=tauA)
        tf.summary.scalar('Bob\'s tau', data=tauB)
        tf.summary.scalar('Eve\'s tau', data=tauE)

        Alice.update(tauB, update_rule)
        Bob.update(tauA, update_rule)
        nb_updates.assign_add(1, use_locking=True)
        tf.summary.experimental.set_step(tf.cast(nb_updates, tf.int64))

        # Eve would update only if tauA = tauB = tauE
        if tauA == tauB == tauE:
            Eve.update(tauA, update_rule)
            nb_eve_updates.assign_add(1, use_locking=True)
        tf.summary.scalar('Eve\'s updates',
                          data=nb_eve_updates)

        Alice_key, Alice_iv = Alice.makeKey(key_length, iv_length)
        Bob_key, Bob_iv = Bob.makeKey(key_length, iv_length)
        Eve_key, Eve_iv = Eve.makeKey(key_length, iv_length)

        # sync of Alice and Bob
        # Calculate the synchronization of Alice and Bob
        score.assign(tf.cast(100 * sync_score(Alice, Bob, L), tf.float32))
        tf.summary.scalar('sync between Alice and Bob',
                          data=score)
        sync_history.append(score)  # plot purpose
        # sync of Alice and Eve
        # Calculate the synchronization of Alice and Eve
        score_eve = 100 * sync_score(Alice, Eve, L)
        tf.summary.scalar('sync between Alice and Eve', data=score_eve)
        sync_history_eve.append(score_eve)  # plot purpose

        tf.print("\rSynchronization = ", score, "%   /  Updates = ",
                 nb_updates, " / Eve's updates = ", nb_eve_updates, sep='')

    end_time = tf.timestamp(name='end_time')
    time_taken = end_time - start_time

    # results
    tf.print("\nTime taken =", time_taken, "seconds.")
    tf.print("Alice's gen key =", Alice_key,
             "key :", Alice_key, "iv :", Alice_iv)
    tf.print("Bob's gen key =", Bob_key,
             "key :", Bob_key, "iv :", Bob_iv)
    tf.print("Eve's gen key =", Eve_key,
             "key :", Eve_key, "iv :", Eve_iv)

    if Alice_key == Bob_key and Alice_iv == Bob_iv:
        try:
            is_dicom = pydicom.dcmread(input_file) is not None
        except(pydicom.errors.InvalidDicomError):
            is_dicom = False
        decrypt_file = "decipher.dcm" if is_dicom else "decipher.txt"
        # cipher with AES
        aes_encrypt_file(is_dicom, input_file, output_file,
                         Alice_key, key_length)
        # decipher with AES
        aes_decrypt_file(is_dicom, output_file,
                         decrypt_file, Alice_key, key_length)
        tf.print("encryption and decryption with aes",
                 key_length, " done.", sep='')
        eve_score = 100 * int(sync_score(Alice, Eve, L))
        if eve_score > 100:
            print("Oops! Nosy Eve synced her machine with Alice's and Bob's!")
        else:
            tf.print("Eve's machine is only ", eve_score,
                     "% synced with Alice's and Bob's and she did ", nb_eve_updates, " updates.", sep='')

    else:
        print("error, Alice and Bob have different key or iv : cipher impossible")
    print(f'Wrote log file to {logdir}')


if __name__ == "__main__":
    main()
