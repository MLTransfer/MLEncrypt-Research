from tpm import TPM
import numpy as np
import time
import sys
import pydicom
import matplotlib.pyplot as plt
import pyAesCrypt
from os import stat, remove
import click


def random(K, N, L):
    '''random
    return a random vector input for TPM
    '''
    return np.random.randint(-L, L + 1, [K, N])


def sync_score(TPM1, TPM2, L):
    '''sync_score
    Synchronize the score of 2 tree parity machines
    TPM1 - Tree Parity Machine 1
    TPM2 - Tree Parity Machine 2
    '''
    return 1.0 - np.average(1.0 * np.abs(TPM1.W - TPM2.W) / (2 * L))


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
    # Tree Parity Machine parameters
    K, N, L = k, n, l

    # Create TPM for Alice, Bob and Eve. Eve eavesdrops communication of Alice and Bob
    print("Creating machines : K=" + str(K) + ", N=" + str(N) + ", L="
          + str(L) + ", k=" + str(key_length) + ", v=" + str(iv_length))
    Alice = TPM(K, N, L)
    Bob = TPM(K, N, L)
    Eve = TPM(K, N, L)

    # Synchronize weights
    nb_updates = 0
    nb_eve_updates = 0
    start_time = time.time()  # Start time
    sync_history = []  # plot purpose
    sync_history_eve = []  # plot purpose
    score = 0  # synchronisation score of Alice and Bob

    while score < 100:

        X = random(K, N, L)  # Create random vector [K, N]

        # compute outputs of TPMs
        tauA = Alice.get_output(X)
        tauB = Bob.get_output(X)
        tauE = Eve.get_output(X)

        Alice.update(tauB, update_rule)
        Bob.update(tauA, update_rule)

        # Eve would update only if tauA = tauB = tauE
        if tauA == tauB == tauE:
            Eve.update(tauA, update_rule)
            nb_eve_updates += 1

        nb_updates += 1
        # sync of Alice and Bob
        # Calculate the synchronization of Alice and Bob
        score = 100 * sync_score(Alice, Bob, L)
        sync_history.append(score)  # plot purpose
        # sync of Alice and Eve
        # Calculate the synchronization of Alice and Eve
        score_eve = 100 * sync_score(Alice, Eve, L)
        sync_history_eve.append(score_eve)  # plot purpose

        sys.stdout.write("\r" + "Synchronization = " + str(int(score)) + "%   /  Updates = "
                         + str(nb_updates) + " / Eve's updates = " + str(nb_eve_updates))

    end_time = time.time()
    time_taken = end_time - start_time

    # results
    print("\nTime taken = " + str(time_taken) + " seconds.")
    Alice_key, Alice_iv = Alice.makeKey(key_length, iv_length)
    Bob_key, Bob_iv = Bob.makeKey(key_length, iv_length)
    Eve_key, Eve_iv = Eve.makeKey(key_length, iv_length)
    print("Alice's gen key = " + str(len(Alice_key))
          + " key : " + Alice_key + " iv : " + Alice_iv)
    print("Bob's gen key = " + str(len(Bob_key))
          + " key : " + Bob_key + " iv : " + Bob_iv)
    print("Eve's gen key = " + str(len(Eve_key))
          + " key : " + Eve_key + " iv : " + Eve_iv)

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
        print("encryption and decryption with aes"
              + str(key_length) + " done.")
    else:
        print("error, Alice and Bob have different key or iv : cipher impossible")

    # Plot graph
    plt.figure(1)
    plt.title('Synchronisation')
    plt.ylabel('sync %')
    plt.xlabel('nb iterations')
    sync_AB, = plt.plot(sync_history)
    sync_Eve, = plt.plot(sync_history_eve)
    plt.legend([sync_AB, sync_Eve], ["sync Alice Bob", "sync Alice Eve"])
    plt.show()


if __name__ == "__main__":
    main()
