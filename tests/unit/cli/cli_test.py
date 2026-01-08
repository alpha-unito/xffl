import logging
from argparse import Namespace
from types import ModuleType
from unittest.mock import MagicMock

import pytest

import xffl.cli.cli as cli

# ============================================================
# Available xFFL commands
# ============================================================

ALL_SUBCOMMANDS = list(cli._COMMANDS.items())
ALL_COMMAND_NAMES = list(cli._COMMANDS.keys())


def test_commands_map_not_empty():
    assert cli._COMMANDS, "No CLI subcommands defined"


# ============================================================
# Helpers
# ============================================================


@pytest.fixture
def fake_args():
    return Namespace(command="run", loglevel=logging.INFO, version=False)


@pytest.fixture
def mock_parser(monkeypatch):
    parser = MagicMock()
    parser.print_help = MagicMock()

    monkeypatch.setattr(cli.cli_parser, "parser", parser, raising=True)
    return parser


@pytest.fixture
def mock_logging(monkeypatch):
    setup = MagicMock()
    monkeypatch.setattr(cli, "setup_logging", setup)
    return setup


# ============================================================
# _dispatch_command
# ============================================================


def test_dispatch_command_unknown_command(mock_parser):
    exit_code = cli._dispatch_command("unknown", Namespace())

    mock_parser.print_help.assert_called_once()
    assert exit_code == 1


@pytest.mark.parametrize("command, module_path", ALL_SUBCOMMANDS)
def test_dispatch_command_all_subcommands(monkeypatch, command, module_path):
    fake_module = ModuleType(module_path)
    fake_module.main = MagicMock(return_value=0)  # type: ignore

    import_module = MagicMock(return_value=fake_module)
    monkeypatch.setattr(cli, "import_module", import_module)

    args = Namespace()

    exit_code = cli._dispatch_command(command, args)

    import_module.assert_called_once_with(module_path)
    fake_module.main.assert_called_once_with(args)
    assert exit_code == 0


@pytest.mark.parametrize("command", ALL_COMMAND_NAMES)
def test_dispatch_command_exception_all_subcommands(monkeypatch, command):
    fake_module = ModuleType(f"xffl.cli.{command}")

    def boom(_):
        raise RuntimeError("crash")

    fake_module.main = boom  # type: ignore

    import_module = MagicMock(return_value=fake_module)
    logger = MagicMock()

    monkeypatch.setattr(cli, "import_module", import_module)
    monkeypatch.setattr(cli, "logger", logger)

    exit_code = cli._dispatch_command(command, Namespace())

    logger.exception.assert_called_once()
    assert exit_code == 1


# ============================================================
# main()
# ============================================================


@pytest.mark.parametrize("command", ALL_COMMAND_NAMES)
def test_main_dispatches_all_subcommands(
    monkeypatch, mock_parser, mock_logging, command
):
    args = Namespace(command=command, loglevel=logging.INFO, version=False)

    mock_parser.parse_args.return_value = args

    dispatch = MagicMock(return_value=0)
    monkeypatch.setattr(cli, "_dispatch_command", dispatch)

    exit_code = cli.main([command])

    mock_logging.assert_called_once_with(logging.INFO)
    dispatch.assert_called_once_with(command, args)
    assert exit_code == 0


def test_main_version_flag(monkeypatch, mock_parser):
    args = Namespace(command=None, loglevel=logging.INFO, version=True)

    dispatch = MagicMock()
    monkeypatch.setattr(cli, "_dispatch_command", dispatch)

    mock_parser.parse_args.return_value = args

    monkeypatch.setattr(
        "xffl.utils.constants.VERSION",
        "1.2.3",
        raising=False,
    )

    logger = MagicMock()
    monkeypatch.setattr(cli, "logger", logger)

    exit_code = cli.main(["--version"])

    logger.info.assert_called_once_with("xFFL version: %s", "1.2.3")
    dispatch.assert_not_called()
    assert exit_code == 0


# ============================================================
# run()
# ============================================================


def test_run_exits_with_main_return_code(monkeypatch):
    monkeypatch.setattr(cli, "main", MagicMock(return_value=2))

    with pytest.raises(SystemExit) as excinfo:
        cli.run()

    assert excinfo.value.code == 2
