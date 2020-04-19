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
@click.argument(
    'method',
    type=click.Choice(
        [
            'hyperopt',
            'bayesopt',
            'nevergrad',
            'skopt',
        ],
        case_sensitive=False
    )
)
def hparams(method):
    from glob import glob

    import tensorflow.summary
    from tensorflow import random as tfrandom, int64 as tfint64
    from ray import init as init_ray
    from ray import tune
    from ray.tune.schedulers import AsyncHyperBandScheduler

    # less summaries are logged if MLENCRYPT_HPARAMS is TRUE (for efficiency)
    environ["MLENCRYPT_HPARAMS"] = 'TRUE'

    logdir = f'logs/hparams/{datetime.now()}'

    update_rules = ['hebbian', 'anti_hebbian', 'random_walk']

    def get_session_num(logdir):
        current_runs = glob(join(logdir, "run-*"))
        if current_runs:
            last_run_path = current_runs[-1]
            last_run_session_num = int(last_run_path.split('-')[-1])
            return last_run_session_num + 1
        else:  # there are no runs yet, start at 0
            return 0

    def trainable(config, reporter):
        """
        Args:
            config (dict): Parameters provided from the search algorithm
                or variant generation.
        """
        if not isinstance(config['update_rule'], str):
            update_rule = update_rules[int(config['update_rule'])]
        else:
            update_rule = config['update_rule']
        K, N, L = int(config['K']), int(config['N']), int(config['L'])
        run_name = f"run-{get_session_num(logdir)}"
        run_logdir = join(logdir, run_name)
        # for each attack, the TPMs should start with the same weights
        initial_weights_tensors = get_initial_weights(K, N, L)
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
                        update_rule, K, N, L,
                        attack,
                        initial_weights
                )
        avg_training_time = tensorflow.math.reduce_mean(
            list(run_training_times.values()))
        avg_eve_score = tensorflow.math.reduce_mean(
            list(run_eve_scores.values()))
        avg_loss = tensorflow.math.reduce_mean(list(run_losses.values()))
        reporter(
            avg_training_time=avg_training_time.numpy(),
            avg_eve_score=avg_eve_score.numpy(),
            avg_loss=avg_loss.numpy()
        )

    if method == 'hyperopt':
        from hyperopt import hp as hyperopt
        from hyperopt.pyll.base import scope
        from ray.tune.suggest.hyperopt import HyperOptSearch
        space = {
            'update_rule': hyperopt.choice(
                'update_rule', ['hebbian', 'anti_hebbian', 'random_walk'],
            ),
            'K': scope.int(hyperopt.quniform('K', 4, 8, q=1)),
            'N': scope.int(hyperopt.quniform('N', 4, 8, q=1)),
            'L': scope.int(hyperopt.quniform('L', 4, 8, q=1)),
            # 'K': scope.int(hyperopt.quniform('K', 4, 32, q=1)),
            # 'N': scope.int(hyperopt.quniform('N', 4, 32, q=1)),
            # 'L': scope.int(hyperopt.quniform('L', 4, 128, q=1))
        }
        algo = HyperOptSearch(
            space,
            metric='avg_loss',
            mode='min'
        )
    elif method == 'bayesopt':
        from ray.tune.suggest.bayesopt import BayesOptSearch
        space = {
            'update_rule': (0, len(update_rules) - 1),
            'K': (4, 8),
            'N': (4, 8),
            'L': (4, 8),
        }
        algo = BayesOptSearch(
            space,
            metric="avg_loss",
            mode="min",
            utility_kwargs={
                "kind": "ucb",
                "kappa": 2.5,
                "xi": 0.0
            }
        )
    elif method == 'nevergrad':
        from ray.tune.suggest.nevergrad import NevergradSearch
        from nevergrad import optimizers
        from nevergrad import p as ngp
        space = {
            'update_rule': (0, len(update_rules) - 1),
            'K': (4, 8),
            'N': (4, 8),
            'L': (4, 8),
        }
        algo = NevergradSearch(
            optimizers.TwoPointsDE(ngp.Instrumentation(
                update_rule=ngp.Choice([
                    'hebbian',
                    'anti_hebbian',
                    'random_walk'
                ]),
                K=ngp.Scalar(lower=4, upper=8).set_integer_casting(),
                N=ngp.Scalar(lower=4, upper=8).set_integer_casting(),
                L=ngp.Scalar(lower=4, upper=8).set_integer_casting(),
            )),
            None,
            metric="avg_loss",
            mode="min"
        )
    elif method == 'skopt':
        from skopt import Optimizer
        from ray.tune.suggest.skopt import SkOptSearch

        optimizer = Optimizer([update_rules, (4, 8), (4, 8), (4, 8)])
        algo = SkOptSearch(
            optimizer,
            ["update_rule", "K", "N", "L"],
            metric="avg_loss",
            mode="min"
        )

    init_ray()
    analysis = tune.run(
        trainable,
        search_alg=algo,
        scheduler=AsyncHyperBandScheduler(
            metric="avg_loss",
            mode="min"
        ),
        # num_samples=100,
        num_samples=10,
    )
    print("Best config: ", analysis.get_best_config(metric="avg_loss"))


if __name__ == '__main__':
    cli()
