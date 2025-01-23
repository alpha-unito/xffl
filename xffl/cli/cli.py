"""Command line interface (CLI) for xFFL
"""

import argparse
import sys

from xffl.cli.config import main as create_config
from xffl.cli.parser import parser
from xffl.utils.constants import VERSION
from xffl.utils.logging import get_logger, set_log_level

logger = get_logger("CLI")


def main(arguments: argparse.Namespace) -> int:
    """xFFL command line interface (CLI)

    :param arguments: Command line arguments
    :type arguments: argparse.Namespace
    :return: Exit code
    :rtype: int
    """

    try:
        args = parser.parse_args(arguments)
        set_log_level(args.loglevel)

        if args.help:
            logger.info(f"\n{parser.format_help()}")
            return 0
        elif args.version:
            logger.info(f"xFFL version: {VERSION}")
            return 0

        if args.command == "config":
            create_config(args)
            return 0
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
