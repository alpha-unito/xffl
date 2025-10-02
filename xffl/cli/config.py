"""Guided configuration files creation for xFFL.

This script guides the user in the creation of the StreamFlow and CWL
configuration files necessary to run xFFL workloads across different HPCs.
"""

import argparse
import json
import re
import shutil
from logging import Logger, getLogger
from pathlib import Path
from typing import Tuple

import yaml

from xffl.cli.parser import subparsers
from xffl.cli.utils import check_and_create_dir, check_cli_arguments
from xffl.utils.constants import DEFAULT_xFFL_DIR
from xffl.utils.utils import check_input, resolve_path
from xffl.workflow.templates.cwl import (
    AggregateStep,
    CWLConfig,
    MainWorkflow,
    RoundWorkflow,
    TrainingStep,
)
from xffl.workflow.templates.streamflow import StreamFlowFile
from xffl.workflow.utils import from_args_to_cwl, import_from_path

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #


def _get_training_info() -> Tuple[Path, Path, str]:
    """Prompt user for training script path, parser path and parser name."""
    executable_path = resolve_path(
        check_input(
            "xFFL requires the Python training scripts\nTraining script path: ",
            "File {} does not exist.",
            lambda path: Path(path).is_file(),
            is_path=True,
        )
    )

    parser_path = resolve_path(
        check_input(
            "Custom arguments parser path: ",
            "File {} does not exist.",
            lambda path: Path(path).is_file(),
            is_path=True,
        )
    )

    parser_name = check_input(
        "Arguments parser module name: ",
        "Invalid name (letters, numbers, dashes, underscores, must start with alnum).",
        lambda x: re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", x),
    )

    return Path(executable_path), Path(parser_path), parser_name


def _get_model_info() -> Tuple[Path, str]:
    """Prompt user for model path and new model name."""
    model_path = resolve_path(
        check_input(
            "Python model path (file or directory): ",
            "{} does not exist.",
            lambda path: Path(path).exists(),
            is_path=True,
        )
    )

    new_model_name = check_input(
        "Name of the new model: ",
        "Invalid name (letters, numbers, dashes, underscores, must start with alnum).",
        lambda x: re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", x),
    )

    return Path(model_path), new_model_name


def _configure_facility(
    facility: str,
    parser_name: str,
    parser_path: Path,
    args: argparse.Namespace,
    streamflow_config: StreamFlowFile,
    main_cwl: MainWorkflow,
    round_cwl: RoundWorkflow,
    cwl_config: CWLConfig,
) -> None:
    """Configure a single facility interactively.

    NOTE: re-imports the user parser on purpose (keeps the original behaviour).
    """
    # SSH info
    address = input(f"{facility}'s frontend node address [IP:port]: ")
    username = input(f"{facility}'s username: ")
    ssh_key = check_input(
        f"Path to {facility}'s SSH key file: ",
        "{} does not exist.",
        lambda p: Path(p).exists(),
        is_path=True,
    )

    # SLURM template
    slurm_template = resolve_path(
        check_input(
            f"Path to {facility}'s SLURM template: ",
            "{} does not exist.",
            lambda p: Path(p).exists(),
            is_path=True,
        )
    )

    # Remote paths (must be resolved as remote)
    step_workdir = check_input(
        "Facility working directory: ",
        "Invalid path.",
        lambda x: resolve_path(x, is_local_path=False),
        is_path=True,
        is_local_path=False,
    )
    image_path = check_input(
        "Facility image file path: ",
        "Invalid path.",
        lambda x: resolve_path(x, is_local_path=False),
        is_path=True,
        is_local_path=False,
    )
    dataset_path = check_input(
        "Facility dataset directory path: ",
        "Invalid path.",
        lambda x: resolve_path(x, is_local_path=False),
        is_path=True,
        is_local_path=False,
    )

    # --- Re-import user ArgumentParser and extract args for this facility ---
    logger.debug(
        f"Dynamically loading ArgumentParser '{parser_name}' from file '{parser_path}' (facility {facility})"
    )
    try:
        executable_parser_module = import_from_path(
            module_name=parser_name, file_path=parser_path
        )
        logger.info(
            f"Command line argument parser '{parser_name}' from file '{parser_path}' correctly imported"
        )

        arg_to_bidding, arg_to_type, arg_to_value = from_args_to_cwl(
            parser=executable_parser_module.parser, arguments=args.arguments
        )
    except SystemExit as se:
        # The user parser called sys.exit (probably missing required args)
        logger.error(
            "The script requires some parameters. Add them using the --arguments option"
        )
        raise se
    except Exception:
        logger.exception(
            "Error while importing or parsing the user ArgumentParser for facility %s",
            facility,
        )
        raise

    # Populate CWL config for this facility
    main_cwl.add_inputs(facility_name=facility, extra_inputs=arg_to_type)
    round_cwl.add_inputs(facility_name=facility, extra_inputs=arg_to_type)
    cwl_config.add_inputs(
        facility_name=facility,
        extra_inputs={
            "image": {"class": "File", "path": Path(image_path).name},
            "dataset": {"class": "Directory", "path": Path(dataset_path).name},
        }
        | arg_to_value,
    )

    # Populate StreamFlow config for this facility
    streamflow_config.add_deployment(
        facility_name=facility,
        address=address,
        username=username,
        ssh_key=ssh_key,
        step_workdir=step_workdir,
        slurm_template=slurm_template,
    )

    streamflow_config.add_training_step(
        facility_name=facility,
        mapping={
            f"dataset_{facility}": str(Path(dataset_path).parent),
            f"image_{facility}": str(Path(image_path).parent),
        },
    )
    streamflow_config.add_inputs(facility_name=facility)

    logger.debug(
        "\n".join(
            [
                f"Inserted the following record for {facility} in the StreamFlow file:",
                json.dumps(streamflow_config.step_bindings[facility], indent=2),
                json.dumps(streamflow_config.deployments[facility], indent=2),
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

    def _dump_cwl(path: Path, data: dict) -> None:
        with path.open("w") as f:
            f.write("#!/usr/bin/env cwl-runner\n")
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

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
    # Validate CLI usage
    try:
        check_cli_arguments(args=args, parser=subparsers.choices["config"])
    except ValueError:
        logger.error("Aborting configuration")
        return 1

    # Prepare workdir
    try:
        workdir = check_and_create_dir(dir_path=args.workdir, folder_name=args.project)
    except FileExistsError:
        logger.error("Aborting configuration")
        return 0
    except FileNotFoundError:
        return 1
    logger.debug("Workdir set to %s", workdir)

    # Create templates
    aggregate_cwl = AggregateStep()
    cwl_config = CWLConfig()
    main_cwl = MainWorkflow()
    round_cwl = RoundWorkflow()
    streamflow_config = StreamFlowFile()
    training_cwl = TrainingStep()
    logger.debug("StreamFlow and CWL templates created")

    # Add aggregation script reference
    cwl_config.content |= {
        "script_aggregation": {
            "class": "File",
            "path": str(Path("py_scripts") / "aggregation_application.py"),
        },
    }

    # --- User-provided training/parser info ---
    executable_path, parser_path, parser_name = _get_training_info()

    # --- First import: extract arguments mapping to populate training inputs ---
    logger.debug(
        f"Dynamically loading ArgumentParser '{parser_name}' from file '{parser_path}' (initial import)"
    )
    try:
        executable_parser_module = import_from_path(
            module_name=parser_name, file_path=parser_path
        )
        logger.info(
            f"Command line argument parser '{parser_name}' from file '{parser_path}' correctly imported"
        )

        arg_to_bidding, arg_to_type, arg_to_value = from_args_to_cwl(
            parser=executable_parser_module.parser, arguments=args.arguments
        )
    except SystemExit as se:
        logger.error(
            "The script requires some parameters. Add them using the --arguments option"
        )
        raise se
    except Exception:
        logger.exception(
            "Error importing or parsing the user ArgumentParser (initial import)"
        )
        raise

    # Populate training step with the "bidding" info (inputs used by training)
    training_cwl.add_inputs(facility_name=None, extra_inputs=arg_to_bidding)

    # Model info
    model_path, new_model_name = _get_model_info()

    # Local workdir
    local_workdir = check_input(
        "Local workdir (leave blank for $TMPDIR): ",
        "Invalid path or blank.",
        lambda p: p == "" or resolve_path(p),
    )
    local_workdir = (
        resolve_path(local_workdir) if local_workdir else Path.cwd().joinpath("tmp")
    )
    logger.debug("local_workdir: %s", local_workdir)
    streamflow_config.add_root_step(local_workdir)

    # Number of iterations
    num_of_iterations = int(
        check_input(
            "Number of iterations to the federated training: ",
            "Insert an integer",
            lambda x: x.isdigit(),
        )
    )

    # Populate top-level CWL config
    cwl_config.content |= {
        "executable": {
            "class": "File",
            "path": str(executable_path),
            "secondaryFiles": [{"class": "File", "path": str(parser_path)}],
        },
        "model": {
            "class": "Directory" if model_path.is_dir() else "File",
            "path": str(model_path),
        },
        "model_basename": new_model_name,
        "max_rounds": num_of_iterations,
    }

    # Facility loop (each iteration re-imports the parser inside _configure_facility)
    facilities: set[str] = set()
    while True:
        facility = check_input(
            "Type facility's logic name: ",
            "Already used.",
            lambda f: f not in facilities,
        )
        facilities.add(facility)

        _configure_facility(
            facility=facility,
            parser_name=parser_name,
            parser_path=parser_path,
            args=args,
            streamflow_config=streamflow_config,
            main_cwl=main_cwl,
            round_cwl=round_cwl,
            cwl_config=cwl_config,
        )

        another = check_input(
            "Insert another facility? [y/n]: ",
            "Answer not accepted.",
            lambda reply: reply.lower() in ["y", "yes", "n", "no"],
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
        raise exception
    finally:
        logger.info(
            "*** Cross-Facility Federated Learning (xFFL) - Configuration finished ***"
        )


if __name__ == "__main__":
    main(args=subparsers.choices["config"].parse_args())
