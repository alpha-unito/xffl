"""xFFL-handled experiment launching

This script wraps StreamFlow with a simple Python CLI, offering a homogeneous interface with xFFL
"""

import argparse
import asyncio
from logging import Logger, getLogger

from xffl.cli.parser import run_parser
from xffl.cli.utils import check_cli_arguments

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def run(args: argparse.Namespace) -> int:
    """Runs an xFFL project

    :param args: Command line arguments
    :type args: argparse.Namespace
    """
    # Check the CLI arguments
    check_cli_arguments(args=args, parser=run_parser)

    # import here for performance concerns
    from xffl.workflow.streamflow import run_streamflow

    asyncio.run(run_streamflow(args=args))

    return 0


def main(args: argparse.Namespace) -> int:
    """xFFL project run entrypoint

    :param args: Command line arguments
    :type args: argparse.Namespace
    :return: Exit code
    :rtype: int
    """
    logger.info("*** Cross-Facility Federated Learning (xFFL) - Project run ***")
    result: int = 0
    try:
        result = run(args=args)
    except Exception as exception:  # TODO check which exception SF raises
        logger.exception(exception)
        raise exception

    logger.info("*** Cross-Facility Federated Learning (xFFL) - Project run ***")
    return result


if __name__ == "__main__":

    try:
        main(run_parser.parse_args())
    except KeyboardInterrupt as e:
        logger.exception(e)
