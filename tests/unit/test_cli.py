"""The command line loads and its commands run (a broken entry point once slipped past every other test)."""

from typer.testing import CliRunner

from facegroup.entrypoints import cli
from tests.conftest import make_photo


def test_help_lists_every_command():
    result = CliRunner().invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for command in ("scan", "status", "people", "retry", "ui", "download-models"):
        assert command in result.output


def test_status_runs_on_an_empty_library():
    result = CliRunner().invoke(cli.app, ["status"])
    assert result.exit_code == 0 and "Items:" in result.output


def test_download_failure_is_a_short_message(monkeypatch, capsys):
    from facegroup.adapters.models.download import ModelDownloadError

    def boom():
        raise ModelDownloadError("Could not download x. Check the internet connection and try again.")

    monkeypatch.setattr(cli, "app", boom)
    try:
        cli.main()
    except SystemExit as exc:
        assert exc.code == 2
    assert "Check the internet connection" in capsys.readouterr().err
