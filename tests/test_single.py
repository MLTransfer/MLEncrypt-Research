import pytest
from click.testing import CliRunner
from mlencrypt_research import cli


@pytest.mark.skip(reason="takes too long")
def test_single():
    runner = CliRunner()
    for ur in [
        'random-same',
        # 'random-different-A-B-E',
        # 'random-different-A-B',
        'hebbian',
        # 'anti_hebbian',
        # 'random_walk',
    ]:
        for attack in ['none', 'geometric', 'probabilistic']:
            result = runner.invoke(
                cli.cli,
                ['single', '-ur', ur, '-a', attack],
                catch_exceptions=False
            )
            assert result.exit_code == 0
            assert not result.exception
            assert result.exit_code == 0

            result_b = runner.invoke(
                cli.cli,
                ['single', '-ur', ur, '-a', attack, '-b'],
                catch_exceptions=False
            )
            assert result_b.exit_code == 0
            assert not result_b.exception
            assert result_b.exit_code == 0

            with runner.isolated_filesystem():
                result_tb = runner.invoke(
                    cli.cli,
                    ['single', '-ur', ur, '-a', attack, '-tb'],
                    catch_exceptions=False
                )
                assert result_tb.exit_code == 0
                assert not result_tb.exception
                assert result_tb.exit_code == 0
