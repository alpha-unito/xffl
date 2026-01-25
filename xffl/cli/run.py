"""xFFL-handled experiment launching.

This script wraps StreamFlow with a simple Python CLI,
offering a homogeneous interface with xFFL.
"""

import asyncio
from argparse import Namespace
from logging import Logger, getLogger

import xffl.cli.parser as cli_parser

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


# --------------------------------------------------------------------------- #
#                                   Entrypoint                                #
# --------------------------------------------------------------------------- #


def run(args: Namespace) -> int:
    """Run an xFFL project.

    :param args: Command line arguments.
    :type args: Namespace
    :return: Exit code (0 if success).
    :rtype: int
    """

    # Import deferred for performance and to avoid heavy dependencies at CLI init
    from xffl.workflow.streamflow import run_streamflow

    asyncio.run(run_streamflow(args=args))
    return 0


def main(args: Namespace) -> int:
    """xFFL project run entrypoint.

    :param args: Command line arguments.
    :type args: Namespace
    :return: Exit code (0 if success, 1 if error).
    :rtype: int
    """
    logger.info("*** Cross-Facility Federated Learning (xFFL) - Run starting ***")
    try:
        return run(args=args)
    except Exception as exception:  # TODO: narrow down to StreamFlow-specific errors
        logger.exception("Run failed: %s", exception)
        return 1
    finally:
        logger.info("*** Cross-Facility Federated Learning (xFFL) - Run finished ***")


if __name__ == "__main__":
    main(cli_parser.subparsers.choices["run"].parse_args())
