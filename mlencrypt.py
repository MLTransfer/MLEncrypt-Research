# -*- coding: utf-8 -*-
from tpm import TPM
from tpm import tb_summary, tb_heatmap, tb_boxplot

from os import environ
from time import perf_counter
from importlib import import_module

import tensorflow as tf
from math import pi


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
    # https://github.com/tensorflow/tensorflow/blob/e6da7ff3b082dfff2188b242847b620f1fe79426/tensorflow/python/keras/losses.py#L1674-L1706
    # TODO: am I using experimental_implements correctly?
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
        weights1_float = tf.cast(weights1_flattened, tf.float32,
                                 name=f'weights-{tpm1_id}-1d-float')
        weights2_float = tf.cast(weights2_flattened, tf.float32,
                                 name=f'weights-{tpm2_id}-1d-float')
        weights1_norm = tf.math.l2_normalize(weights1_float, axis=-1)
        weights2_norm = tf.math.l2_normalize(weights2_float, axis=-1)
        # cos_sim can be from -1 to 1, inclusive
        cos_sim = -tf.math.reduce_sum(weights1_norm * weights2_norm, axis=-1)
        return -cos_sim / 2. + .5  # bound cos_sim to 0 to 1, inclusive
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

    if environ["MLENCRYPT_TB"] == 'TRUE':
        with tf.name_scope(f'{TPM1.name}-{TPM2.name}'):
            tf.summary.scalar('sync', data=rho)
            tf.summary.scalar('generalization-error', data=epsilon)

    return rho


def select_random_from_list(input_list, op_name=None):
    # see this gist for how to select a random value from a list:
    # https://gist.github.com/sumanthratna/b9b57134bb76c9fc62b73553728ca896
    index = tf.random.uniform(
        [],
        maxval=len(input_list),
        dtype=tf.int32,
        name=op_name
    )
    return input_list[index]


# @tf.function(experimental_compile=True)
@tf.function(
    experimental_relax_shapes=True,
    experimental_autograph_options=(
        # tf.autograph.experimental.Feature.AUTO_CONTROL_DEPS,
        # tf.autograph.experimental.Feature.ASSERT_STATEMENTS,
        tf.autograph.experimental.Feature.BUILTIN_FUNCTIONS,
        # tf.autograph.experimental.Feature.EQUALITY_OPERATORS,
        # tf.autograph.experimental.Feature.LISTS
    )
)
def iterate(
    X,
    Alice, Bob, Eve,
    update_rule_A, update_rule_B, update_rule_E,
    nb_updates, nb_eve_updates,
    score, score_eve,
    key_length, iv_length
):
    tf.summary.experimental.set_step(tf.cast(nb_updates, tf.int64))

    log_tb = tf.math.equal(
        tf.constant(environ["MLENCRYPT_TB"], name='setting-hparams'),
        tf.constant('TRUE', name='true'),
        name='do-not-use-hparams'
    )

    # TODO: use tf.cond
    if environ["MLENCRYPT_TB"] == 'TRUE':
        tb_summary('inputs', X)
        K, N = Alice.K, Alice.N
        hpaxis, ipaxis = tf.range(1, K + 1), tf.range(1, N + 1)
        tb_heatmap('inputs', X, ipaxis, hpaxis)
        tb_boxplot('inputs', X, hpaxis)

    # compute outputs of TPMs
    tauA = Alice.get_output(X)
    tauB = Bob.get_output(X)
    tauE = Eve.get_output(X)
    updated = Alice.update(tauB, update_rule_A) \
        and Bob.update(tauA, update_rule_B)
    if updated:
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
                            name='iteration-tau-E-not-equal'),
        tf.math.equal(Eve.type, 'geometric', name='is-E-geometric'),
        name='iteration-update-E-geometric'
    )
    should_update_E = tf.math.logical_or(
        tau_A_B_E_equal,
        update_E_geometric,
        name='iteration-update-E'
    )

    def update_E():
        Eve.update(tauA, update_rule_E)
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
    tf.cond(log_tb, true_fn=log_updates_E,
            false_fn=lambda: None, name='tb-updates-E')

    def compute_and_log_keys_and_ivs():
        Alice_key, Alice_iv = Alice.compute_key(key_length, iv_length)
        Bob_key, Bob_iv = Bob.compute_key(key_length, iv_length)
        Eve_key, Eve_iv = Eve.compute_key(key_length, iv_length)

    tf.cond(log_tb, true_fn=compute_and_log_keys_and_ivs,
            false_fn=lambda: None, name='tb-keys-ivs')

    score.assign(100. * sync_score(Alice, Bob), name='calc-sync-A-B')
    score_eve.assign(100. * sync_score(Alice, Eve), name='calc-sync-A-E')

    # def calc_and_log_sync_B_E():
    #     sync_score(Bob, Eve)
    # # log adversary score for Bob's weights
    # tf.cond(log_tb, true_fn=calc_and_log_sync_B_E,
    #         false_fn=lambda: None, name='calc-sync-B-E')
    if environ["MLENCRYPT_TB"] == 'TRUE':
        # log adversary score for Bob's weights
        sync_score(Bob, Eve)

    tf.print(
        "\rUpdate rule = ", (update_rule_A, update_rule_B,
                             update_rule_E), " / "
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
        "\n\n\n",
        "Creating machines: K=", K, ", N=", N, ", L=", L, ", ",
        "update-rule=", update_rule, ", ",
        "attack=", attack,
        "\n",
        sep='',
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
        score = tf.Variable(0.0, trainable=False, name='score-A-B', dtype=tf.float32)

        # synchronization score of Alice and Eve
        score_eve = tf.Variable(0.0, trainable=False, name='score-A-E', dtype=tf.float32)
    except ValueError:
        # tf.function-decorated function tried to create variables
        # on non-first call.
        score = 0.
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
        nb_eve_updates = tf.Variable(
            0, name='updates-E', trainable=False, dtype=tf.uint16)
    except ValueError:
        # tf.function-decorated function tried to create variables
        # on non-first call.
        pass
    tf.summary.experimental.set_step(0)

    def train_step():
        # Create random vector, X, with dimensions [K, N] and values {-1, 0, 1}
        X = tf.Variable(
            tf.random.uniform(
                (K, N), minval=-1, maxval=1 + 1, dtype=tf.int64),
            trainable=False,
            name='input'
        )

        update_rules = tf.constant(['hebbian', 'anti_hebbian', 'random_walk'])

        if update_rule == 'random-same':
            # use tf.random so that the same update rule is used for each
            # iteration across attacks

            current_update_rule = tf.constant(select_random_from_list(
                update_rules,
                op_name='iteration-ur-A-B-E'
            ))
            update_rule_A = current_update_rule
            update_rule_B = current_update_rule
            update_rule_E = current_update_rule
        elif update_rule == 'random-different':
            update_rule_A = tf.constant(select_random_from_list(
                update_rules,
                op_name='iteration-ur-A'
            ))
            update_rule_B = tf.constant(select_random_from_list(
                update_rules,
                op_name='iteration-ur-B'
            ))
            # update_rule_E = tf.constant(select_random_from_list(
            #     update_rules,
            #     op_name='iteration-ur-E'
            # ))
            update_rule_E = update_rule_A
        elif update_rule in update_rules:
            current_update_rule = tf.constant(update_rule)
            update_rule_A = current_update_rule
            update_rule_B = current_update_rule
            update_rule_E = current_update_rule
        else:
            # TODO: raise an appropriate error/exception
            pass
        iterate(
            X,
            Alice, Bob, Eve,
            update_rule_A, update_rule_B, update_rule_E,
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
    start_time = perf_counter()
    while score < 100. and not weights_A_B_equal:
        train_step()

    end_time = perf_counter()
    training_time = end_time - start_time
    # loss = (tf.math.sigmoid(training_time) + score_eve / 100.) / 2.
    loss = (tf.math.log(training_time) + score_eve / 100.) / 2.
    key_length, iv_length = tf.constant(key_length), tf.constant(iv_length)
    if tf.math.equal(environ["MLENCRYPT_TB"], 'TRUE', name='log-tb'):
        # creates scatterplot (in scalars) dashboard of metric vs steps
        tf.summary.scalar('training_time', training_time)
        tf.summary.scalar('eve_score', score_eve)
        tf.summary.scalar('loss', loss)

        # do this if hparams is enabled because the keys and IVs haven't been
        # calculated yet
        Alice.compute_key(key_length, iv_length)
        Bob.compute_key(key_length, iv_length)
        Eve.compute_key(key_length, iv_length)
        tf.print(
            "\n\n",
            "Training time = ", training_time, " seconds.\n",
            "Alice's key: ", Alice.key, " iv: ", Alice.iv, "\n",
            "Bob's key: ", Bob.key, " iv: ", Bob.iv, "\n",
            "Eve's key: ", Eve.key, " iv: ", Eve.iv,
            sep='',
            name='log-run-final'
        )
    else:
        tf.print(
            "\n\n",
            "Training time = ", training_time, " seconds.\n",
            sep='',
            name='log-run-final'
        )
    return training_time, score_eve, loss
