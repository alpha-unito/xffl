"""Command line interface (CLI) for xFFL.

This is the main entrypoint of the xFFL CLI.
Argument parsing takes place here and the various xFFL subcommands are dispatched.
"""

import sys
from argparse import Namespace
from importlib import import_module
from logging import Logger, getLogger
from types import ModuleType
from typing import Dict, List

import xffl.cli.parser as cli_parser
from xffl.utils.logging import setup_logging

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""

# Mappa subcomandi → modulo main
_COMMANDS: Dict[str, str] = {
    "config": "xffl.cli.config",
    "run": "xffl.cli.run",
    "exec": "xffl.cli.exec",
}


def _dispatch_command(command: str, args: Namespace) -> int:
    """Dispatch a subcommand safely, logging errors.

    :param command: Subcommand name
    :param args: Parsed Namespace
    :return: Exit code
    """
    module_path = _COMMANDS.get(command)
    if not module_path:
        cli_parser.parser.print_help()
        return 1

    try:
        module: ModuleType = import_module(module_path)
        return module.main(args)
    except Exception as e:
        logger.exception("Unhandled exception in %s: %s", command, e)
        return 1


def main(arguments: List[str]) -> int:
    """xFFL command line interface.

    :param arguments: Command line arguments
    :type arguments: List[str]
    :return: Exit code
    """

    args: Namespace = cli_parser.parser.parse_args(arguments)

    # Setup logging
    setup_logging(args.loglevel)
    logger.debug(f"Input arguments: {args}")

    # Handle version
    if getattr(args, "version", False):
        from xffl.utils.constants import VERSION

        logger.info("xFFL version: %s", VERSION)
        return 0

    # Dispatch to subcommand
    return _dispatch_command(args.command, args)


def run() -> None:
    """Main CLI entrypoint"""
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":
    run()
