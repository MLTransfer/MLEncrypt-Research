import click
from mlencrypt import run

from os.path import join
from os import environ
from datetime import datetime

from tensorflow import Variable as tfVariable


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


def weights_tensor_to_variable(weights, name):
    return tfVariable(weights, trainable=True, name=name)


@click.group()
def cli():
    from tensorflow import config as tfconfig

    # TODO: use experimental_jit_scope?
    try:
        use_xla = "--tf_xla_auto_jit=2" in environ["TF_XLA_FLAGS"]
    except KeyError:
        # KeyError: 'TF_XLA_FLAGS'
        use_xla = False

    tfconfig.optimizer.set_jit(use_xla)
    tfconfig.experimental.set_synchronous_execution(True)
    tfconfig.optimizer.set_experimental_options({
        'layout_optimizer': True,
        'constant_folding': True,
        'shape_optimization': True,
        'remapping': True,
        'arithmetic_optimization': True,
        'dependency_optimization': True,
        'loop_optimization': True,
        'function_optimization': True,
        'debug_stripper': False,
        'disable_model_pruning': False,
        'scoped_allocator_optimization': True,
        'pin_to_host_optimization': True,
        'implementation_selector': True,
        # auto_mixed_precision is dangerous, see
        # https://github.com/tensorflow/tensorflow/issues/38724
        'auto_mixed_precision': False,
        'disable_meta_optimizer': False
    })


@cli.command(name='single')
@click.option(
    '-ur', '--update_rule',
    type=click.Choice([
        'random-same',
        'random-different-A-B-E',
        'random-different-A-B',
        'hebbian',
        'anti_hebbian',
        'random_walk',
    ]),
    default='hebbian',
    show_default=True
)
@click.option(
    '-K', '--K', default=8, show_default=True, type=int
)
@click.option(
    '-N', '--N', default=12, show_default=True, type=int
)
@click.option(
    '-L', '--L', default=4, show_default=True, type=int
)
@click.option(
    '-a', '--attack',
    type=click.Choice([
        'none',
        'geometric',
        'probabilistic',
    ]),
    default='none',
    show_default=True
)
@click.option(
    '-kl', '--key_length', default=256, show_default=True, type=int
)
@click.option(
    '-ivl', '--iv_length', default=128, show_default=True, type=int
)
@click.option(
    '-tb', '--tensorboard', is_flag=True
)
def single(update_rule, k, n, l, attack, key_length, iv_length, tensorboard):
    environ["MLENCRYPT_TB"] = str(tensorboard).upper()

    initial_weights = {
        tpm: weights_tensor_to_variable(weights, tpm)
        for tpm, weights in get_initial_weights(k, n, l).items()
    }

    logdir = join(
        'logs/',
        str(datetime.now()),
        f"ur={update_rule},K={k},N={n},L={l},attack={attack}"
    )

    if tensorboard:
        import tensorflow.summary
        # import tensorflow.profiler

        tensorflow.summary.trace_on()
        # TODO: don't profile for more than 10 steps at a time
        tensorflow.profiler.experimental.start(logdir)
        with tensorflow.summary.create_file_writer(logdir).as_default():
            run(
                update_rule, k, n, l,
                attack,
                initial_weights,
                key_length=key_length, iv_length=iv_length
            )
            tensorflow.summary.trace_export("graph")
            tensorflow.profiler.experimental.stop()
    else:
        run(
            update_rule, k, n, l,
            attack,
            initial_weights,
            key_length=key_length, iv_length=iv_length
        )


@cli.command(name='multiple')
@click.argument('count', type=int)
@click.option(
    '-o', '--output_file',
    type=click.Path(dir_okay=False, writable=True, resolve_path=True),
    default='./output.csv',
    show_default=True
)
@click.option(
    '-ur', '--update_rule',
    type=click.Choice([
        'random-same',
        'random-different-A-B-E',
        'random-different-A-B',
        'hebbian',
        'anti_hebbian',
        'random_walk',
    ]),
    default='hebbian',
    show_default=True
)
@click.option(
    '-K', '--K', default=8, show_default=True, type=int
)
@click.option(
    '-N', '--N', default=12, show_default=True, type=int
)
@click.option(
    '-L', '--L', default=4, show_default=True, type=int
)
@click.option(
    '-a', '--attack',
    type=click.Choice([
        'none',
        'geometric',
        'probabilistic',
    ]),
    default='none',
    show_default=True
)
@click.option(
    '-kl', '--key_length', default=256, show_default=True, type=int
)
@click.option(
    '-ivl', '--iv_length', default=128, show_default=True, type=int
)
@click.option(
    '-tb', '--tensorboard', is_flag=True
)
def multiple(
    count, output_file,
    update_rule, k, n, l,
    attack,
    key_length, iv_length,
    tensorboard
):
    environ["MLENCRYPT_TB"] = str(tensorboard).upper()
    with open(output_file, 'w') as losses_writer:
        losses_writer.write(
            "training time (s),synchronization score (%),loss\n"
        )
        losses_writer.flush()
        for _ in range(count):
            initial_weights = {
                tpm: weights_tensor_to_variable(weights, tpm)
                for tpm, weights in get_initial_weights(k, n, l).items()
            }

            if tensorboard:
                import tensorflow.summary
                # import tensorflow.profiler

                logdir = join(
                    'logs/',
                    str(datetime.now()),
                    f"ur={update_rule},K={k},N={n},L={l},attack={attack}"
                )

                tensorflow.summary.trace_on()
                # TODO: don't profile for more than 10 steps at a time
                tensorflow.profiler.experimental.start(logdir)
                with tensorflow.summary.create_file_writer(
                    logdir
                ).as_default():
                    training_time, sync_score, loss = run(
                        update_rule, k, n, l,
                        attack,
                        initial_weights,
                        key_length=key_length, iv_length=iv_length
                    )
                    tensorflow.summary.trace_export("graph")
                    tensorflow.profiler.experimental.stop()
            else:
                training_time, sync_score, loss = run(
                    update_rule, k, n, l,
                    attack,
                    initial_weights,
                    key_length=key_length, iv_length=iv_length
                )
            data = (
                f"{training_time},{sync_score.numpy()},{loss.numpy()}\n"
            )
            losses_writer.write(data)
            losses_writer.flush()


@cli.command(name='hparams')
@click.argument(
    'algorithm',
    type=click.Choice(
        [
            'hyperopt',
            'bayesopt',
            'nevergrad',
            'skopt',
            'dragonfly',
            'bohb',
            'zoopt',
        ],
        case_sensitive=False
    )
)
@click.argument(
    'scheduler',
    type=click.Choice(
        [
            'pbt',
            'ahb',
            'hb',
            'bohb',
            'smr',
        ],
        case_sensitive=False
    )
)
@click.option(
    '-tb', '--tensorboard', is_flag=True
)
def hparams(algorithm, scheduler, tensorboard):
    from glob import glob

    import tensorflow.summary
    from tensorflow import random as tfrandom, int64 as tfint64
    from ray import init as init_ray
    from ray import tune
    from wandb.ray import WandbLogger

    # less summaries are logged if MLENCRYPT_TB is TRUE (for efficiency)
    # TODO: use tf.summary.record_if?
    environ["MLENCRYPT_TB"] = str(tensorboard).upper()

    logdir = f'logs/hparams/{datetime.now()}'

    # "These results show that K = 3 is the optimal choice for the
    # cryptographic application of neural synchronization. K = 1 and K = 2 are
    # too insecure in regard to the geometric attack. And for K > 3 the effort
    # of A and B grows exponentially with increasing L, while the simple attack
    # is quite successful in the limit K -> infinity. Consequently, one should
    # only use Tree Parity Machines with three hidden units for the neural
    # key-exchange protocol." (Ruttor, 2006)

    update_rules = [
        'random-same',
        # 'random-different-A-B-E', 'random-different-A-B',
        'hebbian', 'anti_hebbian', 'random_walk'
    ]
    K_bounds = {'min': 4, 'max': 8}
    N_bounds = {'min': 4, 'max': 8}
    L_bounds = {'min': 4, 'max': 8}
    # TODO: don't use *_bounds.values() since .values doesn't preserve order

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
        run_training_times, run_eve_scores, run_losses = {}, {}, {}
        # for each attack, the TPMs should use the same inputs
        seed = tfrandom.uniform(
            [], minval=0, maxval=tfint64.max, dtype=tfint64).numpy()
        for attack in ['none', 'geometric', 'probabilistic']:
            initial_weights = {
                tpm: weights_tensor_to_variable(weights, tpm)
                for tpm, weights in initial_weights_tensors.items()
            }
            tfrandom.set_seed(seed)

            if tensorboard:
                attack_logdir = join(run_logdir, attack)
                attack_writer = tensorflow.summary.create_file_writer(
                    attack_logdir)
                with attack_writer.as_default():
                    training_times, sync_scores, losses = run(
                        update_rule, K, N, L,
                        attack,
                        initial_weights
                    )
            else:
                training_times, sync_scores, losses = run(
                    update_rule, K, N, L,
                    attack,
                    initial_weights
                )
            run_training_times[attack] = training_times
            run_eve_scores[attack] = sync_scores
            run_losses[attack] = losses
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

    if algorithm == 'hyperopt':
        from hyperopt import hp as hyperopt
        from hyperopt.pyll.base import scope
        from ray.tune.suggest.hyperopt import HyperOptSearch

        space = {
            'update_rule': hyperopt.choice(
                'update_rule', update_rules,
            ),
            'K': scope.int(hyperopt.quniform('K', *K_bounds.values(), q=1)),
            'N': scope.int(hyperopt.quniform('N', *N_bounds.values(), q=1)),
            'L': scope.int(hyperopt.quniform('L', *L_bounds.values(), q=1)),
        }
        algo = HyperOptSearch(
            space,
            metric='avg_loss',
            mode='min'
        )
    elif algorithm == 'bayesopt':
        from ray.tune.suggest.bayesopt import BayesOptSearch

        space = {
            'update_rule': (0, len(update_rules)),
            'K': tuple(K_bounds.values()),
            'N': tuple(N_bounds.values()),
            'L': tuple(L_bounds.values()),
        }
        algo = BayesOptSearch(
            space,
            metric="avg_loss",
            mode="min",
            # TODO: what is utility_kwargs for and why is it needed?
            utility_kwargs={
                "kind": "ucb",
                "kappa": 2.5,
                "xi": 0.0
            }
        )
    elif algorithm == 'nevergrad':
        from ray.tune.suggest.nevergrad import NevergradSearch
        from nevergrad import optimizers
        from nevergrad import p as ngp

        algo = NevergradSearch(
            optimizers.TwoPointsDE(ngp.Instrumentation(
                update_rule=ngp.Choice(update_rules),
                K=ngp.Scalar(
                    lower=K_bounds['min'],
                    upper=K_bounds['max']
                ).set_integer_casting(),
                N=ngp.Scalar(
                    lower=N_bounds['min'],
                    upper=N_bounds['max']
                ).set_integer_casting(),
                L=ngp.Scalar(
                    lower=L_bounds['min'],
                    upper=L_bounds['max']
                ).set_integer_casting(),
            )),
            None,
            metric="avg_loss",
            mode="min"
        )
    elif algorithm == 'skopt':
        from skopt import Optimizer
        from ray.tune.suggest.skopt import SkOptSearch

        optimizer = Optimizer([
            update_rules,
            tuple(K_bounds.values()),
            tuple(N_bounds.values()),
            tuple(L_bounds.values())
        ])
        algo = SkOptSearch(
            optimizer,
            ["update_rule", "K", "N", "L"],
            metric="avg_loss",
            mode="min"
        )
    elif algorithm == 'dragonfly':
        # TODO: doesn't work
        from ray.tune.suggest.dragonfly import DragonflySearch
        from dragonfly.exd.experiment_caller import EuclideanFunctionCaller
        from dragonfly.opt.gp_bandit import EuclideanGPBandit
        # from dragonfly.exd.experiment_caller import CPFunctionCaller
        # from dragonfly.opt.gp_bandit import CPGPBandit
        from dragonfly import load_config

        domain_config = load_config({
            "domain": [
                {
                    "name": "update_rule",
                    "type": "discrete",
                    "dim": 1,
                    "items": update_rules
                },
                {
                    "name": "K",
                    "type": "int",
                    "min": K_bounds['min'],
                    "max": K_bounds['max'],
                    # "dim": 1
                },
                {
                    "name": "N",
                    "type": "int",
                    "min": N_bounds['min'],
                    "max": N_bounds['max'],
                    # "dim": 1
                },
                {
                    "name": "L",
                    "type": "int",
                    "min": L_bounds['min'],
                    "max": L_bounds['max'],
                    # "dim": 1
                }
            ]
        })
        func_caller = EuclideanFunctionCaller(
            None, domain_config.domain.list_of_domains[0])
        optimizer = EuclideanGPBandit(func_caller, ask_tell_mode=True)
        algo = DragonflySearch(optimizer, metric="avg_loss", mode="min")
    elif algorithm == 'bohb':
        from ConfigSpace import ConfigurationSpace
        from ConfigSpace import hyperparameters as CSH
        from ray.tune.suggest.bohb import TuneBOHB

        config_space = ConfigurationSpace()
        config_space.add_hyperparameter(CSH.CategoricalHyperparameter(
            "update_rule", choices=update_rules))
        config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter(
            name='K', lower=K_bounds['min'], upper=K_bounds['max']))
        config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter(
            name='N', lower=N_bounds['min'], upper=N_bounds['max']))
        config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter(
            name='L', lower=L_bounds['min'], upper=L_bounds['max']))
        algo = TuneBOHB(config_space, metric="avg_loss", mode="min")
    elif algorithm == 'zoopt':
        from ray.tune.suggest.zoopt import ZOOptSearch
        from zoopt import ValueType

        space = {
            "update_rule": (
                ValueType.DISCRETE,
                range(0, len(update_rules)),
                False
            ),
            "K": (
                ValueType.DISCRETE,
                range(K_bounds['min'], K_bounds['max'] + 1),
                True
            ),
            "N": (
                ValueType.DISCRETE,
                range(N_bounds['min'], N_bounds['max'] + 1),
                True
            ),
            "L": (
                ValueType.DISCRETE,
                range(L_bounds['min'], L_bounds['max'] + 1),
                True
            ),
        }
        # TODO: change budget to a large value
        algo = ZOOptSearch(
            budget=10,
            dim_dict=space,
            metric="avg_loss",
            mode="min"
        )

    # TODO: use more appropriate arguments for schedulers:
    # https://docs.ray.io/en/master/tune/api_docs/schedulers.html
    if scheduler == 'pbt':
        from ray.tune.schedulers import PopulationBasedTraining
        from random import randint
        sched = PopulationBasedTraining(
            metric="avg_loss",
            mode="min",
            hyperparam_mutations={
                "update_rule": update_rules,
                "K": lambda: randint(K_bounds['min'], K_bounds['max']),
                "N": lambda: randint(N_bounds['min'], N_bounds['max']),
                "L": lambda: randint(L_bounds['min'], L_bounds['max']),
            }
        )
    elif scheduler == 'ahb':
        from ray.tune.schedulers import AsyncHyperBandScheduler
        sched = AsyncHyperBandScheduler(metric="avg_loss", mode="min")
    elif scheduler == 'hb':
        from ray.tune.schedulers import HyperBandScheduler
        sched = HyperBandScheduler(metric="avg_loss", mode="min")
    elif algorithm == 'bohb' or scheduler == 'bohb':
        from ray.tune.schedulers import HyperBandForBOHB
        sched = HyperBandForBOHB(metric="avg_loss", mode="min")
    elif scheduler == 'msr':
        from ray.tune.schedulers import MedianStoppingRule
        sched = MedianStoppingRule(metric="avg_loss", mode="min")
    init_ray()
    analysis = tune.run(
        trainable,
        search_alg=algo,
        scheduler=sched,
        # num_samples=100,
        num_samples=10,
        loggers=[
            tune.logger.JsonLogger,
            tune.logger.CSVLogger,
            tune.logger.TBXLogger,
            WandbLogger
        ],
        config={
            "monitor": True,
            "env_config": {
                "wandb": {
                    "project": "mlencrypt-research",
                    "sync_tensorboard": True,
                },
            },
        },
    )
    print("Best config: ", analysis.get_best_config(metric="avg_loss"))


if __name__ == '__main__':
    cli()
