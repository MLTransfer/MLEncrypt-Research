from click.testing import CliRunner
from mlencrypt_research import cli


def test_multiple():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli.cli,
            ['multiple', '2'],
            catch_exceptions=False
        )
        assert result.exit_code == 0
        assert not result.exception
        assert result.exit_code == 0
