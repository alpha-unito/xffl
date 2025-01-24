"""Command line interface (CLI) for xFFL
"""

import argparse
import sys
from logging import Logger, getLogger

from xffl.cli.config import main as create_config
from xffl.cli.parser import parser
from xffl.cli.run import main as xffl_run
from xffl.utils.constants import VERSION
from xffl.utils.logging import setup_logging
from xffl.workflow.config import main as xffl_config

logger: Logger = getLogger(__name__)
"""Deafult xFFL logger"""


def main(arguments: argparse.Namespace) -> int:
    """xFFL command line interface (CLI)

    :param arguments: Command line arguments
    :type arguments: argparse.Namespace
    :return: Exit code
    :rtype: int
    """

    try:
        args = parser.parse_args(arguments)
        setup_logging(args.loglevel)

        if args.help:
            logger.info(f"\n{parser.format_help()}")
            return 0
        elif args.version:
            logger.info(f"xFFL version: {VERSION}")
            return 0

        if args.command == "config":
            return xffl_config(args)
        elif args.command == "run":
            return xffl_run(args)
        else:
            logger.critical(f"\n{parser.format_help()}")
            return 1
    except KeyboardInterrupt:
        logger.critical("Unexpected keyboard interrupt")
        return 1


def run() -> int:
    """Main CLI entrypoint

    :return: Exit code
    :rtype: int
    """
    return main(sys.argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:])
