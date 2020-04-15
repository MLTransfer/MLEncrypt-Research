import click
from mlencrypt import run

from os.path import join
from os import environ
from datetime import datetime

import tensorflow.summary
from tensorflow import random as tfrandom
from tensorflow import int64 as tfint64, function as tffunction
from tensorboard.plugins.hparams import api as hp
from hyperopt import hp as hyperopt, fmin, tpe
from hyperopt.pyll.base import scope


@click.group()
def cli():
    pass


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
    environ["MLENCRYPT_HPARAMS"] = 'FALSE'

    tensorflow.summary.trace_on()

    initial_weights = None

    logdir = join(
        'logs/',
        str(datetime.now()),
        f"ur={update_rule},K={k},N={n},L={l},attack={attack}"
    )

    with tensorflow.summary.create_file_writer(logdir).as_default():
        run(
            update_rule,
            k,
            n,
            l,
            attack,
            initial_weights=initial_weights,
            key_length=key_length,
            iv_length=iv_length
        )
        tensorflow.summary.trace_export("graph")


@cli.command(name='hparams')
def hparams():
    # less summaries are logged if MLENCRYPT_HPARAMS is TRUE (for efficiency)
    environ["MLENCRYPT_HPARAMS"] = 'TRUE'

    logdir = f'logs/hparams/{datetime.now()}'
    writer = tensorflow.summary.create_file_writer(logdir)

    HP_UPDATE_RULE = hp.HParam(
        'update_rule',
        domain=hp.Discrete(['hebbian', 'anti_hebbian', 'random_walk']),
        display_name='update_rule'
    )
    HP_K = hp.HParam(
        'tpm_k',
        domain=hp.IntInterval(4, 32),
        display_name='K'
    )
    HP_N = hp.HParam(
        'tpm_n',
        domain=hp.IntInterval(4, 32),
        display_name='N'
    )
    HP_L = hp.HParam(
        'tpm_l',
        domain=hp.IntInterval(4, 128),
        display_name='L'
    )
    hparams = [HP_UPDATE_RULE, HP_K, HP_N, HP_L]

    with writer.as_default():
        hp.hparams_config(
            hparams=hparams,
            metrics=[
                hp.Metric(
                    'training_time',
                    display_name='Average Training Time (s)'
                ),
                hp.Metric(
                    'eve_score',
                    display_name='Average Eve Sync (%)'
                ),
                hp.Metric(
                    'avg_loss',
                    display_name='Average Loss',
                    description='Average of S(Training Time) and Eve Sync for each type of TPM.'
                )
            ]
        )

    session_num = 0

    @tffunction
    def objective(args):
        run_name = f"run-{session_num}"
        K, N, L = args[1], args[2], args[3]
        run_logdir = join(logdir, run_name)
        # for each attack, the TPMs should start with the same weights
        initial_weights = {
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
        with tensorflow.summary.create_file_writer(run_logdir).as_default():
            hp.hparams({
                HP_UPDATE_RULE: args[0],
                HP_K: K,
                HP_N: N,
                HP_L: L
            })
            run_training_times = {}
            run_eve_scores = {}
            run_losses = {}
            # for each attack, the TPMs should use the same inputs
            seed = tfrandom.uniform(
                [1], minval=0, maxval=tfint64.max, dtype=tfint64).numpy()[0]
            for attack in ['none', 'geometric']:
                attack_logdir = join(run_logdir, attack)
                with tensorflow.summary.create_file_writer(attack_logdir).as_default():
                    tfrandom.set_seed(seed)
                    run_training_times[attack], \
                        run_eve_scores[attack], \
                        run_losses[attack] = \
                        run(*args, attack, initial_weights=initial_weights)
            avg_training_time = tensorflow.math.reduce_mean(
                list(run_training_times.values()))
            avg_eve_score = tensorflow.math.reduce_mean(
                list(run_eve_scores.values()))
            avg_loss = tensorflow.math.reduce_mean(list(run_losses.values()))
            tensorflow.summary.scalar('training_time', avg_training_time)
            tensorflow.summary.scalar('eve_score', avg_eve_score)
            tensorflow.summary.scalar('avg_loss', avg_loss)
            session_num += 1

        return avg_loss.numpy().item()

    space = [
        hyperopt.choice(
            'update_rule', ['hebbian', 'anti_hebbian', 'random_walk'],
        ),
        scope.int(hyperopt.quniform('tpm_k', 4, 32, q=1)),
        scope.int(hyperopt.quniform('tpm_n', 4, 32, q=1)),
        scope.int(hyperopt.quniform('tpm_l', 4, 128, q=1))
    ]
    # TODO: is atpe.suggest better?
    best = fmin(objective, space=space, algo=tpe.suggest, max_evals=100)
    print(best)


if __name__ == '__main__':
    cli()
