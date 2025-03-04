"""Command line interface (CLI) for xFFL

This is the main entrypoint of the xFFL CLI
Here argument parsing takes place, anche the various xFFL subcommands are interpreted
"""

import sys
from logging import Logger, getLogger
from typing import List

from xffl.cli.config import main as xffl_config
from xffl.cli.parser import config_parser, parser, run_parser, simulate_parser
from xffl.cli.run import main as xffl_run
from xffl.cli.simulate import main as xffl_simulate
from xffl.utils.constants import VERSION
from xffl.utils.logging import setup_logging

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def main(arguments: List[str]) -> int:
    """xFFL command line interface (CLI)

    :param arguments: Command line arguments
    :type arguments: argparse.Namespace
    :return: Exit code
    :rtype: int
    """

    # Logging facilities setup
    setup_logging()

    # Check if there are arguments to be passed to the runner script
    args_index = len(arguments)
    if "-args" in arguments:
        args_index = arguments.index("-args")
    elif "--arguments" in arguments:
        args_index = arguments.index("--arguments")

    try:
        # Parse CLI arguments and isolate script arguments
        args = parser.parse_args(arguments[:args_index])
        args.arguments = arguments[args_index + 1 :]

        # Logging facilities setup
        setup_logging(args.loglevel)

        # xFFL subcommands handling
        if args.command == "config":
            if args.help:
                logger.info(f"\n{config_parser.format_help()}")
                return 0
            else:
                return xffl_config(args)
        elif args.command == "run":
            if args.help:
                logger.info(f"\n{run_parser.format_help()}")
                return 0
            else:
                return xffl_run(args)
        elif args.command == "simulate":
            if args.help:
                logger.info(f"\n{simulate_parser.format_help()}")
                return 0
            else:
                return xffl_simulate(args=args)

        # xFFL arguments handling
        if args.help:
            logger.info(f"\n{parser.format_help()}")
            return 0
        elif args.version:
            logger.info(f"xFFL version: {VERSION}")
            return 0
        else:
            logger.critical(f"\n{parser.format_help()}")
            return 1
    except Exception as e:
        logger.exception(e)
        return 1


def run() -> int:
    """Main CLI entrypoint

    :return: Exit code
    :rtype: int
    """
    return main(sys.argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:])
