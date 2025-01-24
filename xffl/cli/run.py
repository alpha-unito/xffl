"""xFFL-handled experiment launching

This script wraps StreamFlow with a simple Python CLI, offering a homogeneous interface with xFFL
"""

import argparse
import asyncio
from logging import Logger, getLogger

from xffl.workflow.streamflow import run_streamflow

logger: Logger = getLogger(__name__)
"""Deafult xFFL logger"""


def run_project(args: argparse.Namespace) -> None:
    asyncio.run(run_streamflow(args=args))


def main(args: argparse.Namespace) -> int:
    logger.info("*** Cross-Facility Federated Learning (xFFL) - Project run ***")
    run_project(args)
    logger.info("\*** Cross-Facility Federated Learning (xFFL) - Project run ***")

    return 0


if __name__ == "__main__":
    from xffl.cli.parser import config_parser

    try:
        main(config_parser.parse_args())
    except KeyboardInterrupt:
        logger.exception("Unexpected keyboard interrupt")
