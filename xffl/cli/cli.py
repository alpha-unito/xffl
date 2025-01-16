"""Command line interface (CLI) for xFFL
"""

import argparse
import sys

from xffl.cli.parser import parser
from xffl.config import main as create_config
from xffl.version import VERSION


def main(arguments: argparse.Namespace) -> int:
    """xFFL command line interface (CLI)

    :param arguments: Command line arguments
    :type arguments: argparse.Namespace
    :return: Exit code
    :rtype: int
    """
    try:
        args = parser.parse_args(arguments)
        if args.command == "config":
            create_config(args)
            return 0
        elif args.command == "version":
            print(f"xFFL version: {VERSION}")
            return 0
        else:
            parser.print_help(file=sys.stderr)
            return 1
    except KeyboardInterrupt:
        print("Unexpected keyboard interrupt")
        return 1


def run() -> int:
    """Main CLI entrypoint

    :return: Exit code
    :rtype: int
    """
    return main(sys.argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:])
