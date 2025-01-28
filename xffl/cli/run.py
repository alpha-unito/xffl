"""xFFL-handled experiment launching

This script wraps StreamFlow with a simple Python CLI, offering a homogeneous interface with xFFL
"""

import argparse
import asyncio
import inspect
import logging
import os
from logging import Logger, getLogger
from subprocess import PIPE, STDOUT, Popen, run

import xffl
import xffl.workflow

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def local_run(args: argparse.Namespace) -> int:
    xffl_env = os.environ.copy()

    if args.workdir == os.getcwd():
        logger.warning(
            f"The specified working directory corresponds with the current directory, which is the default value."
        )
    xffl_env["CODE_FOLDER"] = args.workdir

    if args.model == os.getcwd():
        logger.warning(
            f"The specified model directory corresponds with the current directory, which is the default value."
        )
    xffl_env["MODEL_FOLDER"] = args.model

    if args.dataset == os.getcwd():
        logger.warning(
            f"The specified dataset directory corresponds with the current directory, which is the default value."
        )
    xffl_env["DATASET_FOLDER"] = args.dataset

    if args.image == os.getcwd():
        logger.warning(
            f"The specified image directory corresponds with the current directory, which is the default value."
        )
    xffl_env["IMAGE"] = args.image

    if args.project == "project":
        logger.warning(
            f"The specified project executable file path corresponds with the default value."
        )

    if args.venv:
        logger.debug(f"Using virtual environment: {args.venv}")
        xffl_env["VENV"] = args.venv
    elif args.image:
        if args.image == os.getcwd():
            logger.warning(
                f"The specified image directory corresponds with the current directory, which is the default value."
            )
        logger.debug(f"Using container image: {args.venv}")
        xffl_env["IMAGE"] = args.image
    else:
        logger.error(f"No execution environment specified [container/virtual env]")

    xffl_env["FACILITY"] = "local"
    xffl_env["XFFL_WORLD_SIZE"] = str(args.world_size)
    logger.debug(f"Created xFFL environment: {xffl_env}")

    facilitator = os.path.join(
        os.path.dirname(inspect.getfile(xffl.workflow)), "scripts", "facilitator.sh"
    )
    command = [
        facilitator,
        args.project,
        "-dbg" if args.loglevel == logging.DEBUG else "",
        "-m",
        args.model,
        "-d",
        args.dataset,
        # "2>&1",
    ]
    logger.debug(f"Running local execution: {' '.join(command)}")
    with run(command, env=xffl_env) as process:
        for line in process.stdout:
            logger.info(" ".join(line.decode("utf-8").rstrip().split()))
        process.wait()
        status = process.poll()
    return status


def streamflow_run(args: argparse.Namespace) -> None:
    """Runs an xFFL project

    :param args: Command line arguments
    :type args: argparse.Namespace
    """
    from xffl.workflow.streamflow import (  # import here for performance concerns
        run_streamflow,
    )

    asyncio.run(run_streamflow(args=args))


def main(args: argparse.Namespace) -> int:
    """xFFL project run entrypoing

    :param args: Command line arguments
    :type args: argparse.Namespace
    :return: Exit code
    :rtype: int
    """
    logger.info("*** Cross-Facility Federated Learning (xFFL) - Project run ***")
    exit_code = 0
    try:
        if args.local:
            logger.debug(f"Starting local run with arguments: {args}")
            local_run(args)
            logger.debug(f"Concluding local run with arguments: {args}")
        else:
            logger.debug(f"Starting StreamFlow run with arguments: {args}")
            streamflow_run(args)
    except Exception as e:  # TODO check which exception SF raises
        logger.exception(e.strerror)
        exit_code = 1
    finally:
        logger.info("\*** Cross-Facility Federated Learning (xFFL) - Project run ***")
        return exit_code


if __name__ == "__main__":
    from xffl.cli.parser import run_parser

    try:
        main(run_parser.parse_args())
    except KeyboardInterrupt:
        logger.exception("Unexpected keyboard interrupt")
