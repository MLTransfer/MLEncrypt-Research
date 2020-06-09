from click.testing import CliRunner
from mlencrypt_research import cli
from os import mkdir


def test_hparams():
    runner = CliRunner(env={'WANDB_MODE': 'dryrun'})
    with runner.isolated_filesystem():
        mkdir('./wandb')
        with open('./wandb/settings', mode='w') as wandb_settings_file:
            wandb_settings_file.write('[default]\n\n')

        result = runner.invoke(
            cli.cli,
            ['hparams', 'nevergrad', '-n', '2'],
            catch_exceptions=False
        )
        assert result.exit_code == 0
        assert not result.exception
        assert result.exit_code == 0
