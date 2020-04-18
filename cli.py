import click
from mlencrypt import run

from os.path import join
from os import environ
from datetime import datetime


def get_initial_weights(K, N, L):
    from tensorflow import (
        random as tfrandom,
        int64 as tfint64
    )
    return {
        'Alice': tfrandom.uniform(
            (K, N),
            minval=-L,
            maxval=L + 1,
            dtype=tfint64
        ),
        'Bob': tfrandom.uniform(
            (K, N),
            minval=-L,
            maxval=L + 1,
            dtype=tfint64
        ),
        # TODO: doesn't work for probabilistic:
        'Eve': tfrandom.uniform(
            (K, N),
            minval=-L,
            maxval=L + 1,
            dtype=tfint64
        )
    }


def weights_tensor_to_variable(weights):
    from tensorflow import Variable as tfVariable
    return tfVariable(weights, trainable=True)


@click.group()
def cli():
    from tensorflow import config as tfconfig
    # tfconfig.experimental_run_functions_eagerly(True)
    tfconfig.optimizer.set_experimental_options({
        'layout_optimizer': True,
        'constant_folding': True,
        'shape_optimization': True,
        'remapping': True,
        'arithmetic_optimization': True,
        'dependency_optimization': True,
        'loop_optimization': True,
        # 'function_optimization': True,
        # debug_stripper results in:
        # W tensorflow/core/common_runtime/process_function_library_runtime.cc:697] Ignoring multi-device function optimization failure: Not found: No attr named 'T' in NodeDef:
        # [[{{node PrintV2}}]]
        # [[PrintV2]]
        # 'debug_stripper': True,
        'disable_model_pruning': False,
        'scoped_allocator_optimization': True,
        'pin_to_host_optimization': True,
        'implementation_selector': True,
        'auto_mixed_precision': False,  # hasn't been implemented in our code
        'disable_meta_optimizer': False
    })


@cli.command(name='single')
@click.option(
    '-ur', '--update_rule', default='hebbian', show_default=True, type=str
)
@click.option(
    '--K', default=8, show_default=True, type=int
)
@click.option(
    '--N', default=12, show_default=True, type=int
)
@click.option(
    '--L', default=4, show_default=True, type=int
)
@click.option(
    '--attack', default='none', show_default=True, type=str
)
@click.option(
    '-kl', '--key_length', default=256, show_default=True, type=int
)
@click.option(
    '-ivl', '--iv_length', default=128, show_default=True, type=int
)
def single(update_rule, k, n, l, attack, key_length, iv_length):
    import tensorflow.summary

    environ["MLENCRYPT_HPARAMS"] = 'FALSE'

    initial_weights = {tpm: weights_tensor_to_variable(
        weights) for tpm, weights in get_initial_weights(k, n, l).items()}

    logdir = join(
        'logs/',
        str(datetime.now()),
        f"ur={update_rule},K={k},N={n},L={l},attack={attack}"
    )

    tensorflow.summary.trace_on()
    with tensorflow.summary.create_file_writer(logdir).as_default():
        run(
            update_rule, k, n, l,
            attack,
            initial_weights,
            key_length=key_length, iv_length=iv_length
        )
        tensorflow.summary.trace_export("graph")


@cli.command(name='hparams')
def hparams():
    from glob import glob

    import tensorflow.summary
    from tensorflow import random as tfrandom, int64 as tfint64
    # from ray import init as init_ray
    from ray import tune
    from ray.tune.schedulers import AsyncHyperBandScheduler
    from ray.tune.suggest.hyperopt import HyperOptSearch
    from hyperopt import hp as hyperopt
    from hyperopt.pyll.base import scope

    # less summaries are logged if MLENCRYPT_HPARAMS is TRUE (for efficiency)
    environ["MLENCRYPT_HPARAMS"] = 'TRUE'

    logdir = f'logs/hparams/{datetime.now()}'

    def get_session_num(logdir):
        current_runs = glob(join(logdir, "run-*"))
        if current_runs:
            last_run_path = current_runs[-1]
            last_run_session_num = int(last_run_path.split('-')[-1])
            return last_run_session_num + 1
        else:  # there are no runs yet, start at 0
            return 0

    def trainable(config):
        """
        Args:
            config (dict): Parameters provided from the search algorithm
                or variant generation.
        """
        run_name = f"run-{get_session_num(logdir)}"
        run_logdir = join(logdir, run_name)
        # for each attack, the TPMs should start with the same weights
        initial_weights_tensors = get_initial_weights(
            config['K'],
            config['N'],
            config['L']
        )
        run_training_times = {}
        run_eve_scores = {}
        run_losses = {}
        # for each attack, the TPMs should use the same inputs
        seed = tfrandom.uniform(
            [1], minval=0, maxval=tfint64.max, dtype=tfint64).numpy()[0]
        for attack in ['none', 'geometric']:
            attack_logdir = join(run_logdir, attack)
            initial_weights = {tpm: weights_tensor_to_variable(
                weights) for tpm, weights in initial_weights_tensors.items()}
            with tensorflow.summary.create_file_writer(attack_logdir).as_default():
                tfrandom.set_seed(seed)
                run_training_times[attack], \
                    run_eve_scores[attack], \
                    run_losses[attack] = \
                    run(
                        config['update_rule'],
                        config['K'],
                        config['N'],
                        config['L'],
                        attack,
                        initial_weights
                )
        avg_training_time = tensorflow.math.reduce_mean(
            list(run_training_times.values()))
        avg_eve_score = tensorflow.math.reduce_mean(
            list(run_eve_scores.values()))
        avg_loss = tensorflow.math.reduce_mean(list(run_losses.values()))
        tune.track.log(
            avg_training_time=avg_training_time.numpy(),
            avg_eve_score=avg_eve_score.numpy(),
            avg_loss=avg_loss.numpy()
        )

    space = {
        'update_rule': hyperopt.choice(
            'update_rule', ['hebbian', 'anti_hebbian', 'random_walk'],
        ),
        'K': scope.int(hyperopt.quniform('K', 4, 8, q=1)),
        'N': scope.int(hyperopt.quniform('N', 4, 8, q=1)),
        'L': scope.int(hyperopt.quniform('L', 4, 8, q=1))
    }
    # TODO: is atpe.suggest better?
    # best = fmin(objective, space=space, algo=tpe.suggest, max_evals=100)
    # print(best)

    # init_ray(local_mode=True)
    analysis = tune.run(
        trainable,
        search_alg=HyperOptSearch(
            space,
            metric='avg_loss',
            mode='min'
        ),
        scheduler=AsyncHyperBandScheduler(
            metric="avg_loss",
            mode="min"
        ),
    )

    print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))


if __name__ == '__main__':
    cli()
