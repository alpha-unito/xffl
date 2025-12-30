"""Command line interface (CLI) for xFFL.

This is the main entrypoint of the xFFL CLI.
Argument parsing takes place here and the various xFFL subcommands are dispatched.
"""

import argparse
import sys
from logging import Logger, getLogger
from typing import Callable, Dict, List

import xffl.cli.parser as cli_parser
from xffl.utils.logging import setup_logging

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""

# Mappa subcomandi → modulo main
COMMANDS: Dict[str, str] = {
    "config": "xffl.cli.config",
    "run": "xffl.cli.run",
    "exec": "xffl.cli.exec",
}


def dispatch_command(command: str, args: argparse.Namespace) -> int:
    """Dispatch a subcommand safely, logging errors.

    :param command: Subcommand name
    :param args: Parsed argparse.Namespace
    :return: Exit code
    """
    module_path = COMMANDS.get(command)
    if not module_path:
        cli_parser.parser.print_help()
        return 1

    try:
        module: Callable = __import__(module_path, fromlist=["main"])
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

    args: argparse.Namespace = cli_parser.parser.parse_args(arguments)

    # Setup logging
    setup_logging(args.loglevel)
    logger.debug(f"Input arguments: {args}")

    # Handle version
    if getattr(args, "version", False):
        from xffl.utils.constants import VERSION

        logger.info("xFFL version: %s", VERSION)
        return 0

    # Dispatch to subcommand
    return dispatch_command(args.command, args)


def run() -> None:
    """Main CLI entrypoint"""
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":
    run()
