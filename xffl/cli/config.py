"""Guided configuration files creation for xFFL.

This script guides the user in the creation of the StreamFlow and CWL
configuration files necessary to run xFFL workloads across different HPCs.
"""

import argparse
import json
import os.path
import re
import shutil
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, MutableMapping, Tuple

import yaml

import xffl.cli.parser as cli_parser
from xffl.cli.utils import check_input
from xffl.utils.constants import SUPPORTED_QUEUE_MANAGERS, DEFAULT_xFFL_DIR
from xffl.workflow.templates.cwl import (
    AggregateStep,
    CWLConfig,
    MainWorkflow,
    RoundWorkflow,
    TrainingStep,
)
from xffl.workflow.templates.streamflow import StreamFlowFile

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #


def _dump_cwl(path: Path, data: MutableMapping[str, Any]) -> None:
    with path.open("w") as fd:
        fd.write("#!/usr/bin/env cwl-runner\n")
        yaml.dump(data, fd, default_flow_style=False, sort_keys=False)


def _get_training_info() -> Tuple[str, str]:
    executable_path: str = check_input(
        text="Training script path: ",
        warning_msg="File {} does not exist.",
        control=lambda path: Path(path).is_file(),
        is_local_path=True,
    )
    config_file: str = check_input(
        text="Dataclass file path: ",
        warning_msg="File {} does not exist.",
        control=lambda path: Path(path).is_file(),
        is_local_path=True,
    )
    return executable_path, config_file


def _get_model_info() -> Tuple[str, str]:
    """Prompt user for model path and new model name."""
    model_path: str = check_input(
        text="Python model path (file or directory): ",
        warning_msg="{} does not exist.",
        control=lambda path: Path(path).exists(),
        is_local_path=True,
    )
    new_model_name: str = check_input(
        text="Name of the new model: ",
        warning_msg="Invalid name (letters, numbers, dashes, underscores, must start with alphanum).",
        control=lambda x: re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", x),
    )
    return model_path, new_model_name


def _configure_facility(
    facility_name: str,
    streamflow_config: StreamFlowFile,
    main_cwl: MainWorkflow,
    round_cwl: RoundWorkflow,
    cwl_config: CWLConfig,
) -> None:
    """Configure a single facility interactively."""

    # SSH info
    address: str = input(f"{facility_name}'s frontend node address [IP:port]: ")
    username: str = input(f"{facility_name}'s username: ")
    ssh_key: str = check_input(
        text=f"Path to {facility_name}'s SSH key file: ",
        warning_msg="{} does not exist.",
        control=lambda p: Path(p).is_file(),
        is_local_path=True,
    )

    available_queue_manager = ["none", *SUPPORTED_QUEUE_MANAGERS]
    queue_manager = check_input(
        text=f"Queue manager in the facility (choices {available_queue_manager}): ",
        warning_msg="{} did not in the available queue managers.",
        control=lambda p: p in available_queue_manager,
        is_local_path=False,
    )

    # template
    template: str = check_input(
        text=f"Path to {facility_name}'s template (optional): ",
        warning_msg="{} does not exist.",
        control=lambda p: Path(p).is_file(),
        is_local_path=True,
        optional=True,
    )

    # Remote paths (must be resolved as remote)
    step_workdir: str = check_input(
        text="Facility working directory: ",
        warning_msg="Invalid path.",
        is_local_path=False,
    )
    image_path: str = check_input(
        text="Facility image file path: ",
        warning_msg="Invalid path.",
        is_local_path=False,
    )
    dataset_path: str = check_input(
        text="Facility dataset directory path: ",
        warning_msg="Invalid path.",
        is_local_path=False,
    )

    # Populate CWL config for this facility
    main_cwl.add_inputs(facility_name=facility_name)
    round_cwl.add_inputs(facility_name=facility_name)
    cwl_config.add_inputs(
        inputs={
            f"facility_{facility_name}": facility_name,
            f"image_{facility_name}": {
                "class": "File",
                "path": os.path.basename(image_path),
            },
            f"dataset_{facility_name}": {
                "class": "Directory",
                "path": os.path.basename(dataset_path),
            },
        }
    )

    # Populate StreamFlow config for this facility
    streamflow_config.add_deployment(
        facility_name=facility_name,
        address=address,
        username=username,
        ssh_key=ssh_key,
        step_workdir=step_workdir,
        queue_manager=queue_manager if queue_manager != "none" else None,
        template=template if template else None,
    )

    streamflow_config.add_training_step(
        facility_name=facility_name,
        mapping={
            f"dataset_{facility_name}": os.path.dirname(dataset_path),
            f"image_{facility_name}": os.path.dirname(image_path),
        },
    )
    streamflow_config.add_inputs(facility_name=facility_name)

    logger.debug(
        "\n".join(
            [
                f"Inserted the following record for {facility_name} in the StreamFlow file:",
                json.dumps(streamflow_config.step_bindings[facility_name], indent=2),
                json.dumps(streamflow_config.deployments[facility_name], indent=2),
            ]
        )
    )


def _write_output_files(
    workdir: Path,
    streamflow_config: StreamFlowFile,
    cwl_config: CWLConfig,
    main_cwl: MainWorkflow,
    round_cwl: RoundWorkflow,
    aggregate_cwl: AggregateStep,
    training_cwl: TrainingStep,
) -> None:
    """Write StreamFlow, CWL and config files to disk."""

    # StreamFlow file
    with (workdir / "streamflow.yml").open("w") as f:
        yaml.dump(
            streamflow_config.save(), f, default_flow_style=False, sort_keys=False
        )

    # CWL files
    cwl_dir = workdir / "cwl"
    clt_dir = cwl_dir / "clt"
    cwl_dir.mkdir(parents=True, exist_ok=True)
    clt_dir.mkdir(parents=True, exist_ok=True)

    _dump_cwl(cwl_dir / "main.cwl", main_cwl.save())
    _dump_cwl(cwl_dir / "round.cwl", round_cwl.save())
    _dump_cwl(clt_dir / "aggregate.cwl", aggregate_cwl.save())
    _dump_cwl(clt_dir / "training.cwl", training_cwl.save())

    with (cwl_dir / "config.yml").open("w") as f:
        yaml.dump(cwl_config.save(), f, default_flow_style=False, sort_keys=False)

    # Scripts and py_scripts
    shutil.copytree(
        Path(DEFAULT_xFFL_DIR) / "workflow" / "scripts", cwl_dir / "scripts"
    )
    (cwl_dir / "py_scripts").mkdir(exist_ok=True)
    shutil.copy(
        Path(DEFAULT_xFFL_DIR)
        / "workflow"
        / "templates"
        / "aggregation_application.py",
        cwl_dir / "py_scripts",
    )


def _check_and_create_dir(dir_path: Path, folder_name: str) -> Path:
    """Check the base directory and create a subfolder.

    :param dir_path: Base directory path.
    :type dir_path: FolderLike
    :param folder_name: Name of the subfolder to create.
    :type folder_name: str
    :raises FileNotFoundError: If the base directory path does not exist.
    :raises FileExistsError: If the target directory already exists and overwrite is denied.
    :return: Absolute path to the created (or existing) folder.
    :rtype: Path
    """
    if not (dir_path).exists():
        logger.error(f"The provided working directory path {dir_path} does not exist.")
        raise FileNotFoundError(dir_path)

    target_dir: Path = dir_path / folder_name
    logger.debug(f"Attempting to create directory {target_dir}")

    try:
        target_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        answer = check_input(
            f"Directory {target_dir} already exists. Overwrite it? [y/n]: ",
            "Answer not accepted.",
            lambda reply: reply.lower() in ("y", "yes", "n", "no"),
        )
        if answer.lower() in ("n", "no"):
            raise
    return target_dir.resolve()


# --------------------------------------------------------------------------- #
#                                   Entrypoint                                #
# --------------------------------------------------------------------------- #


def config(args: argparse.Namespace) -> int:
    """Run guided configuration for xFFL project.

    The user parser is imported twice on purpose:
      * once before facility loop to populate general training inputs (arg_to_bidding),
      * once per facility inside _configure_facility to re-import/re-evaluate arguments
        (this mirrors the original behaviour you requested).
    """

    # Prepare workdir
    try:
        workdir: Path = _check_and_create_dir(
            dir_path=args.workdir, folder_name=args.project
        )
    except FileExistsError:
        logger.error("Aborting configuration")
        return 0
    except FileNotFoundError:
        return 1
    logger.debug("Workdir set to %s", workdir)

    # Create templates
    aggregate_cwl: AggregateStep = AggregateStep()
    cwl_config: CWLConfig = CWLConfig()
    main_cwl: MainWorkflow = MainWorkflow()
    round_cwl: RoundWorkflow = RoundWorkflow()
    streamflow_config: StreamFlowFile = StreamFlowFile()
    training_cwl: TrainingStep = TrainingStep()
    logger.debug("StreamFlow and CWL templates created")

    # Add aggregation script reference
    cwl_config.content = dict(cwl_config.content)
    cwl_config.content |= {
        "script_aggregation": {
            "class": "File",
            "path": str(Path("py_scripts") / "aggregation_application.py"),
        },
    }

    # --- User-provided local data info ---
    executable_path, dataclass_path = _get_training_info()
    model_path, new_model_name = _get_model_info()
    local_workdir: str = check_input(
        text="Local workdir: ",
        warning_msg="Invalid path or blank.",
        control=lambda path: Path(path).is_dir(),
        is_local_path=True,
    )
    logger.debug(f"local_workdir: {local_workdir}")
    streamflow_config.add_root_step(str(local_workdir))

    # Number of iterations
    num_of_iterations: int = int(
        check_input(
            text="Number of iterations to the federated training: ",
            warning_msg="Insert an integer",
            control=lambda x: x.isdigit(),
        )
    )

    # Populate top-level CWL config
    cwl_config.add_inputs(
        {
            "executable": {
                "class": "File",
                "path": executable_path,
                "secondaryFiles": [{"class": "File", "path": dataclass_path}],
            },
            "model": {
                "class": "Directory" if Path(model_path).is_dir() else "File",
                "path": model_path,
            },
            "model_basename": new_model_name,
            "max_rounds": num_of_iterations,
        }
    )

    facilities: set[str] = set()
    while True:
        facility_name: str = check_input(
            text="Facility logic name: ",
            warning_msg="Already used.",
            control=lambda f: f not in facilities,
        )
        facilities.add(facility_name)
        _configure_facility(
            facility_name=facility_name,
            streamflow_config=streamflow_config,
            main_cwl=main_cwl,
            round_cwl=round_cwl,
            cwl_config=cwl_config,
        )
        another: str = check_input(
            text="Insert another facility? [y/n]: ",
            warning_msg="Answer not accepted.",
            control=lambda reply: reply.lower() in ["y", "yes", "n", "no"],
        )
        if another.lower() in ["n", "no"]:
            break

    # Finalize merge step and write files
    round_cwl.update_merge_step()
    _write_output_files(
        workdir=workdir,
        streamflow_config=streamflow_config,
        cwl_config=cwl_config,
        main_cwl=main_cwl,
        round_cwl=round_cwl,
        aggregate_cwl=aggregate_cwl,
        training_cwl=training_cwl,
    )

    return 0


def main(args: argparse.Namespace) -> int:
    """xFFL project's guided configuration entrypoint."""
    logger.info(
        "*** Cross-Facility Federated Learning (xFFL) - Configuration starting ***"
    )
    try:
        return config(args=args)
    except Exception as exception:
        logger.exception("Configuration failed: %s", exception)
        return 1
    finally:
        logger.info(
            "*** Cross-Facility Federated Learning (xFFL) - Configuration finished ***"
        )


if __name__ == "__main__":
    main(args=cli_parser.subparsers.choices["config"].parse_args())
