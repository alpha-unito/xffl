"""StreamFlow wrappers

This file contains all the scripts and wrapper code related to StreamFlow and its execution
"""

import argparse
import asyncio
import logging
import os
from logging import Logger, getLogger

from streamflow.config.config import WorkflowConfig
from streamflow.config.validator import SfValidator
from streamflow.cwl.main import main as cwl_main
from streamflow.ext.utils import load_extensions
from streamflow.log_handler import logger as sf_logger
from streamflow.main import build_context

from xffl.utils.logging import set_external_loggers

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


async def run_streamflow(args: argparse.Namespace) -> None:
    """Run a StreamFlow workflow

    :param args: Command line arguments
    :type args: argparse.Namespace
    """

    # Mapping between xFFL and StreamFlow CLI arguments
    args.name = args.project
    args.outdir = args.outdir or os.path.join(args.workdir, args.project)
    streamflow_file = os.path.join(args.workdir, args.project, "streamflow.yml")

    # Logger
    set_external_loggers()
    if args.loglevel == logging.WARNING:
        sf_logger.setLevel(logging.WARNING)
    elif args.loglevel == logging.DEBUG:
        sf_logger.setLevel(logging.DEBUG)

    # StreamFlow run
    load_extensions()  # Load 2FA extension
    streamflow_config = SfValidator().validate_file(str(streamflow_file))
    streamflow_config["path"] = streamflow_file
    context = build_context(streamflow_config)
    try:
        workflow_tasks = []
        for workflow in streamflow_config.get("workflows", {}):
            workflow_config = WorkflowConfig(workflow, streamflow_config)
            if workflow_config.type == "cwl":
                workflow_tasks.append(
                    asyncio.create_task(cwl_main(workflow_config, context, args))
                )
        await asyncio.gather(*workflow_tasks)
    finally:
        await context.close()
