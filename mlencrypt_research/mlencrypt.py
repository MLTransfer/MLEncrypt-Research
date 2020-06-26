# -*- coding: utf-8 -*-
from mlencrypt_research.tpms import TPM
from mlencrypt_research.summaries import tb_summary, tb_heatmap, tb_boxplot

from os import getenv
from time import perf_counter
from importlib import import_module

import tensorflow as tf
from numpy import pi


@tf.function(
    experimental_autograph_options=tf.autograph.experimental.Feature.all_but(
        tf.autograph.experimental.Feature.NAME_SCOPES),
    experimental_relax_shapes=True,
)
def sync_score(tpm1_w, tpm2_w, tpm1_name, tpm2_name):
    """
    Args:
        TPM1: Tree Parity Machine 1. Has same parameters as TPM2.
        TPM2: Tree Parity Machine 2. Has same parameters as TPM1.
    Returns:
        The synchronization score between TPM1 and TPM2.
    """
    tpm1_id, tpm2_id = tpm1_name[0], tpm2_name[0]

    # adapted from:
    # https://github.com/tensorflow/tensorflow/blob/r2.2/tensorflow/python/keras/losses.py#L1672-L1716
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
        weights1_float = tf.cast(weights1, tf.float32,
                                 name=f'weights-{tpm1_id}-float')
        weights2_float = tf.cast(weights2, tf.float32,
                                 name=f'weights-{tpm2_id}-float')
        weights1_norm = tf.math.l2_normalize(weights1_float)
        weights2_norm = tf.math.l2_normalize(weights2_float)

        # this doesn't work well; cos_sim is occasionally greater than 1.
        # this is also marginally slower:
        # cos_sim = tf.tensordot(weights1_norm, weights2_norm,
        #                        [[0, 1], [0, 1]])
        # return cos_sim

        # cos_sim is bound by [-1, 1] (for the most part); note that with
        # cos_sim is still occasionally greater than 1:
        cos_sim = -tf.math.reduce_sum(weights1_norm * weights2_norm)
        # we change cos_sim's range to [0, 1] according to:
        # https://arxiv.org/pdf/0711.2411.pdf#page=62
        return -cos_sim / 2. + .5
    rho = cosine_similarity(tpm1_w, tpm2_w)

    if getenv('MLENCRYPT_TB', 'FALSE') == 'TRUE':
        # the generalization error, epsilon, is the probability that a
        # repulsive step occurs if two corresponding hidden units have
        # different sigma (Ruttor, 2006).
        epsilon = tf.math.multiply(
            tf.guarantee_const(tf.constant(1. / pi), name='reciprocal-pi'),
            tf.bitcast(
                tf.math.acos(rho, name=f'angle-{tpm1_id}-{tpm2_id}'),
                tf.float32,
                name=f'angle-{tpm1_id}-{tpm2_id}'
            ),
            name=f'scale-angle-{tpm1_id}-{tpm2_id}-to-0-1'
        )

        with tf.name_scope(f'{tpm1_name}-{tpm2_name}'):
            tf.summary.scalar('sync', data=rho)
            tf.summary.scalar('generalization-error', data=epsilon)

    return rho


@tf.function(
    experimental_autograph_options=(
        tf.autograph.experimental.Feature.AUTO_CONTROL_DEPS,
        tf.autograph.experimental.Feature.ASSERT_STATEMENTS,
        tf.autograph.experimental.Feature.BUILTIN_FUNCTIONS,
        tf.autograph.experimental.Feature.EQUALITY_OPERATORS,
    ),
    experimental_relax_shapes=True,
)
def select_random_from_list(input_list, op_name=None):
    # see this gist for how to select a random value from a list:
    # https://gist.github.com/sumanthratna/b9b57134bb76c9fc62b73553728ca896
    index = tf.random.uniform(
        [],
        maxval=len(input_list),
        dtype=tf.int32,
        name=op_name
    )
    return tf.constant(input_list)[index]


def iterate(
    X,
    Alice, Bob, Eve,
    update_rule_A, update_rule_B, update_rule_E,
    nb_updates, nb_eve_updates,
    score, score_eve,
    key_length, iv_length
):
    tf.summary.experimental.set_step(nb_updates)

    if getenv('MLENCRYPT_TB', 'FALSE') == 'TRUE':
        tb_summary('inputs', X)
        K, N = Alice.K, Alice.N
        hpaxis, ipaxis = tf.range(1, K + 1), tf.range(1, N + 1)
        tb_heatmap('inputs', X, ipaxis, hpaxis)
        tb_boxplot('inputs', X, hpaxis)

    # compute outputs of TPMs
    tauA = Alice.get_output(X)
    tauB = Bob.get_output(X)
    tauE = Eve.get_output(X)
    updated_A_B = Alice.update(tauB, update_rule_A) \
        and Bob.update(tauA, update_rule_B)
    if updated_A_B:
        nb_updates.assign_add(1, name='updates-A-B-increment')
    tf.summary.experimental.set_step(nb_updates)

    def update_E():
        Eve.update(tauA, update_rule_E)
        nb_eve_updates.assign_add(1, name='updates-E-increment')

    if Eve.type == 'basic' and (updated_A_B and tauA == tauE):
        # (updated_A_B and tauA == tauE) is the same as
        # (tauA == tauB and tauB == tauE); if tauA equals tauB and tauB equals
        # tauE then tauA must equal tauE due to the associative law of boolean
        # multiplication
        update_E()
    elif Eve.type == 'geometric' and updated_A_B:
        # https://www.ki.tu-berlin.de/fileadmin/fg135/publikationen/Ruttor_2007_DNC.pdf#page=2
        # https://www.ccs.neu.edu/home/riccardo/courses/cs6750-fa09/talks/Lowell-neural-crypto.pdf#page=12
        update_E()
    elif Eve.type == 'probabilistic':
        Eve.update(tauA, updated_A_B)
        nb_eve_updates.assign_add(1, name='updates-E-increment')

    if getenv('MLENCRYPT_TB', 'FALSE') == 'TRUE':
        tf.summary.scalar('updates-E', data=nb_eve_updates)

    if getenv('MLENCRYPT_TB', 'FALSE') == 'TRUE':
        Alice_key, Alice_iv = Alice.compute_key(key_length, iv_length)
        Bob_key, Bob_iv = Bob.compute_key(key_length, iv_length)
        Eve_key, Eve_iv = Eve.compute_key(key_length, iv_length)

    if getenv('MLENCRYPT_BARE', 'FALSE') != 'TRUE':
        score.assign(
            100. * sync_score(
                Alice.w, Bob.w,
                Alice.name, Bob.name
            ),
            name='calc-sync-A-B'
        )
        if Eve.type == 'probabilistic':
            eve_w = Eve.mpW
        else:
            eve_w = Eve.w
        score_eve.assign(
            100. * sync_score(
                Alice.w, eve_w,
                Alice.name, Eve.name
            ),
            name='calc-sync-A-E'
        )
        if getenv('MLENCRYPT_TB', 'FALSE') == 'TRUE':
            sync_score(Bob.w, eve_w, Bob.name, Eve.name)

    current_update_rules = (update_rule_A, update_rule_B, update_rule_E)
    if getenv('MLENCRYPT_BARE', 'FALSE') == 'TRUE':
        tf.print(
            "Update rules = ", current_update_rules, " / ",
            nb_updates, " Updates (Alice) / ",
            nb_eve_updates, " Updates (Eve)",
            sep='',
            name='log-iteration'
        )
    else:
        tf.print(
            "Update rules = ", current_update_rules, " / "
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
    with tf.experimental.async_scope():

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

        # TODO: don't reimport entire file:
        tpm_mod = import_module('mlencrypt_research.tpms')
        if attack == 'probabilistic':
            Eve_class_name = 'ProbabilisticTPM'
        elif attack == 'geometric':
            Eve_class_name = 'GeometricTPM'
        elif attack == 'none':
            Eve_class_name = 'TPM'
        else:
            # TODO: better message for ValueError
            raise ValueError
        Eve_class = getattr(tpm_mod, Eve_class_name)
        Eve = Eve_class('Eve', K, N, L, initial_weights['Eve'])

        # try:
        # synchronization score of Alice and Bob
        score = tf.Variable(0., trainable=False,
                            name='score-A-B', dtype=tf.float32)

        # synchronization score of Alice and Eve
        score_eve = tf.Variable(0., trainable=False,
                                name='score-A-E', dtype=tf.float32)
        # except ValueError:
        #     # tf.function-decorated function tried to create variables
        #     # on non-first call.
        #     score = 0.
        #     score_eve = 0.

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

        # try:
        nb_updates = tf.Variable(
            0, name='updates-A-B', trainable=False, dtype=tf.int64)
        nb_eve_updates = tf.Variable(
            0, name='updates-E', trainable=False, dtype=tf.int64)
        # except ValueError:
        #     # tf.function-decorated function tried to create variables
        #     # on non-first call.
        #     pass

        autograph_features = tf.autograph.experimental.Feature.all_but([
            tf.autograph.experimental.Feature.AUTO_CONTROL_DEPS,
            tf.autograph.experimental.Feature.NAME_SCOPES,
        ])

        @tf.function(
            experimental_autograph_options=autograph_features,
            experimental_relax_shapes=True,
        )
        def train_step():
            # Create random vector, X, with dimensions [K, N] and values
            # bound by [-1, 1]
            X = tf.random.uniform(
                (K, N), minval=-1, maxval=1 + 1, dtype=tf.int32)

            tpm_update_rules = [
                'hebbian',
                'anti_hebbian',
                'random_walk'
            ]

            if update_rule == 'random-same':
                # use tf.random so that the same update rule is used for
                # each iteration across attacks

                current_update_rule = select_random_from_list(
                    tpm_update_rules,
                    op_name='iteration-ur-A-B-E'
                )
                update_rule_A = current_update_rule
                update_rule_B = current_update_rule
                update_rule_E = current_update_rule
            elif update_rule.startswith('random-different'):
                update_rule_A = select_random_from_list(
                    tpm_update_rules,
                    op_name='iteration-ur-A'
                )
                update_rule_B = select_random_from_list(
                    tpm_update_rules,
                    op_name='iteration-ur-B'
                )
                if update_rule == 'random-different-A-B-E':
                    update_rule_E = select_random_from_list(
                        tpm_update_rules,
                        op_name='iteration-ur-E'
                    )
                elif update_rule == 'random-different-A-B':
                    update_rule_E = update_rule_A
                else:
                    raise ValueError(
                        f"'{update_rule}' is an invalid update rule. "
                        "Valid update rules are: "
                        "'hebbian', "
                        "'anti_hebbian', "
                        "'random_walk', "
                        "'random-same', "
                        "'random-different-A-B', and "
                        "'random-different-A-B-E'"
                    )
            # elif tf.reduce_any(tf.math.equal(update_rule, tpm_update_rules)):
            elif not isinstance(tpm_update_rules, tf.Tensor) and \
                    update_rule in tpm_update_rules:
                current_update_rule = tf.guarantee_const(update_rule)
                update_rule_A = current_update_rule
                update_rule_B = current_update_rule
                update_rule_E = current_update_rule
            else:
                raise ValueError(
                    f"'{update_rule}' is an invalid update rule. "
                    "Valid update rules are: "
                    "'hebbian', "
                    "'anti_hebbian', "
                    "'random_walk', "
                    "'random-same', "
                    "'random-different-A-B', and "
                    "'random-different-A-B-E'"
                )
            iterate(
                X,
                Alice, Bob, Eve,
                update_rule_A, update_rule_B, update_rule_E,
                nb_updates, nb_eve_updates,
                score, score_eve,
                key_length, iv_length
            )
            tf.summary.experimental.set_step(nb_updates)

        # In the long run, perf_counter doesn't help because different CPUs
        # have different performance. For the sake of prosperity, we use the
        # number of training steps, although training time would be more useful
        # in a practical setting.
        # start_time = perf_counter()
        while not tf.reduce_all(tf.math.equal(Alice.w, Bob.w)):
            # TODO: instead of while, use for until L^4*K*N and break
            train_step()

        # end_time = perf_counter()
        # training_time = end_time - start_time
        # loss = (tf.math.sigmoid(training_time) + score_eve / 100.) / 2.
        # for reference, log(120) = 2.08 and log(43000) = 4.63
        loss = tf.math.log(nb_updates) * score_eve / 100.
        key_length = tf.guarantee_const(tf.constant(key_length))
        iv_length = tf.guarantee_const(tf.constant(iv_length))
        if getenv('MLENCRYPT_BARE', 'FALSE') == 'TRUE':
            if attack == 'probabilistic':
                eve_w = Eve.mpW
            else:
                eve_w = Eve.w
            score_eve.assign(
                100. * sync_score(Alice.w, eve_w, Alice.name, Eve.name),
                name='calc-sync-A-E'
            )

            tf.print(
                "\n\n",
                "Training time = ", training_time, " seconds.\n",
                sep='',
                name='log-run-final'
            )
        else:
            if getenv('MLENCRYPT_TB', 'FALSE') == 'TRUE':
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
        if tf.math.equal(getenv('MLENCRYPT_TB', 'FALSE'), 'TRUE', name='log-tb'):
            # create scatterplots (in scalars dashboard) of metric vs steps
            tf.summary.scalar('training_time', training_time)
            tf.summary.scalar('eve_score', score_eve)
            tf.summary.scalar('loss', loss)

    return training_time, score_eve, loss
