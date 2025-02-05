"""xFFL-handled experiment launching

This script wraps StreamFlow with a simple Python CLI, offering a homogeneous interface with xFFL
"""

import argparse
import asyncio
from logging import Logger, getLogger
from subprocess import run

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def run(args: argparse.Namespace) -> None:
    """Runs an xFFL project

    :param args: Command line arguments
    :type args: argparse.Namespace
    """

    # import here for performance concerns
    from xffl.workflow.streamflow import run_streamflow

    asyncio.run(run_streamflow(args=args))


def main(args: argparse.Namespace) -> int:
    """xFFL project run entrypoint

    :param args: Command line arguments
    :type args: argparse.Namespace
    :return: Exit code
    :rtype: int
    """
    logger.info("*** Cross-Facility Federated Learning (xFFL) - Project run ***")
    try:
        run(args)
    except Exception as e:  # TODO check which exception SF raises
        logger.exception(e)
        raise
    finally:
        logger.info("\*** Cross-Facility Federated Learning (xFFL) - Project run ***")


if __name__ == "__main__":
    from xffl.cli.parser import run_parser

    try:
        main(run_parser.parse_args())
    except KeyboardInterrupt as e:
        logger.exception(e)
