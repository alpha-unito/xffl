"""StreamFlow wrappers

This file contains all the scripts and wrapper code related to StreamFlow and its execution
"""

import argparse
import asyncio
import logging
import os
from logging import Logger, getLogger
from pathlib import Path

import yaml
from streamflow.config.config import WorkflowConfig
from streamflow.config.validator import SfValidator
from streamflow.cwl.main import main as cwl_main
from streamflow.ext.utils import load_extensions
from streamflow.log_handler import logger as sf_logger
from streamflow.main import build_context

from xffl.utils.constants import XFFL_CACHE_DIR
from xffl.workflow.templates.cwl import MainWorkflow

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def create_federation_main(num_clients: int) -> None:
    main = MainWorkflow(num_clients=num_clients)
    main.dumps()


async def run_streamflow(args: argparse.Namespace) -> None:
    """Run a StreamFlow workflow

    :param args: Command line arguments
    :type args: argparse.Namespace
    """

    # Mapping between xFFL and StreamFlow CLI arguments
    clients = args.clients.split(",")
    create_federation_main(len(clients))

    args.name = args.project
    args.outdir = args.outdir or os.path.join(args.workdir, args.project)
    streamflow_file = str(os.path.join(XFFL_CACHE_DIR, "streamflow.yml"))
    with open(streamflow_file, "w") as fd:
        data = {
            "version": "v1.0",
            "workflows": {
                "xffl": {
                    "type": "cwl",
                    "config": {
                        "file": str(
                            Path(
                                XFFL_CACHE_DIR,
                                "workflow",
                                f"main_{len(clients)}_clients.cwl",
                            )
                        )
                    },
                }
            },
        }
        yaml.dump(data, fd, default_flow_style=False, sort_keys=False)

    # Logger
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
        # shutil.rmtree(XFFL_CACHE_DIR)
