"""Tests for main garmin CLI (garmin init, etc.)."""

import pytest
import sys


def test_garmin_init_subcommand(capsys):
    from garmin_analysis.cli_main import main
    sys.argv = ["garmin", "init"]
    assert main() == 0
    out, _ = capsys.readouterr()
    assert "Bootstrap" in out or "Garmin" in out
    assert "Folders" in out or "Databases" in out


def test_garmin_no_args_prints_help(capsys):
    from garmin_analysis.cli_main import main
    sys.argv = ["garmin"]
    assert main() == 0
    out, _ = capsys.readouterr()
    assert "init" in out.lower() or "usage" in out.lower() or "command" in out.lower()


def test_garmin_init_verbose(capsys):
    from garmin_analysis.cli_main import main
    sys.argv = ["garmin", "init", "-v"]
    assert main() == 0
    out, _ = capsys.readouterr()
    assert "Folders" in out
