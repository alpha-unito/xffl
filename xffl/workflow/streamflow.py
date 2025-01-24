import argparse
import asyncio
import os

from streamflow.config.config import WorkflowConfig
from streamflow.config.validator import SfValidator
from streamflow.cwl.main import main as cwl_main
from streamflow.ext.utils import load_extensions
from streamflow.main import build_context


async def run_streamflow(args: argparse.Namespace) -> None:
    args.name = args.name if args.name else os.path.basename(args.project)
    streamflow_file = os.path.join(args.workdir, args.project, "streamflow.yml")
    load_extensions()  # Load 2FA extension
    streamflow_config = SfValidator().validate_file(streamflow_file)
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
    finally:
        await context.close()
