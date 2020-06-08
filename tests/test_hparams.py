from click.testing import CliRunner
from mlencrypt_research import cli


def test_hparams():
    runner = CliRunner(env={'WANDB_MODE': 'dryrun'})
    # with runner.isolated_filesystem():
    result = runner.invoke(
        cli.cli,
        ['hparams', 'nevergrad', '-n', '2'],
        catch_exceptions=False
    )
    assert result.exit_code == 0
    assert not result.exception
    assert result.exit_code == 0
