"""xFFL-handled experiment launching.

This script wraps StreamFlow with a simple Python CLI,
offering a homogeneous interface with xFFL.
"""

import argparse
import asyncio
from logging import Logger, getLogger

from xffl.cli.parser import subparsers

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def run(args: argparse.Namespace) -> int:
    """Run an xFFL project.

    :param args: Command line arguments.
    :type args: argparse.Namespace
    :return: Exit code (0 if success).
    :rtype: int
    """

    # Import deferred for performance and to avoid heavy dependencies at CLI init
    from xffl.workflow.streamflow import run_streamflow

    asyncio.run(run_streamflow(args=args))
    return 0


def main(args: argparse.Namespace) -> int:
    """xFFL project run entrypoint.

    :param args: Command line arguments.
    :type args: argparse.Namespace
    :return: Exit code (0 if success, 1 if error).
    :rtype: int
    """
    logger.info("*** Cross-Facility Federated Learning (xFFL) - Run starting ***")
    try:
        return run(args=args)
    except Exception as exception:  # TODO: narrow down to StreamFlow-specific errors
        logger.exception("Run failed: %s", exception)
        raise exception
    finally:
        logger.info("*** Cross-Facility Federated Learning (xFFL) - Run finished ***")


if __name__ == "__main__":
    main(subparsers.choices["run"].parse_args())
