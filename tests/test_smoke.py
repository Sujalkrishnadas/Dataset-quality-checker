"""Smoke tests: the package imports and the CLI runs."""

from dataset_quality_checker import __version__
from dataset_quality_checker.cli import main


def test_version_is_a_non_empty_string() -> None:
    assert isinstance(__version__, str) and __version__


def test_cli_without_arguments_prints_help(capsys) -> None:
    assert main([]) == 0
    assert "dqc" in capsys.readouterr().out
