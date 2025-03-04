"""Guided configuration files creation for xFFL

This script guides the user in the creation of the StreamFlow and CWL configuration files
necessary to run xFFL workloads across different HPCs
"""

import argparse
import json
import os
import re
import shutil
from logging import Logger, getLogger

import yaml

from xffl.cli.parser import config_parser
from xffl.cli.utils import check_and_create_workdir, check_cli_arguments
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


def config(args: argparse.Namespace) -> None:
    """Gathers from the user all the necessary parameters to generate a valid StreamFlow file for xFFL

    :param args: Command line arguments
    :type args: argparse.Namespace
    :raises FileNotFoundError: If the command-line provided workdir does not exists
    :raises FileExistsError: If the command-line provided project folder already exists
    """

    # Check the CLI arguments
    check_cli_arguments(args=args, parser=config_parser)

    # Project folder and path checks
    logger.debug(
        f"Verifying working directory {args.workdir} and project name {args.project}"
    )
    try:
        workdir = check_and_create_workdir(
            workdir_path=args.workdir, project_name=args.project
        )
    except (FileExistsError, FileNotFoundError) as e:
        raise e

    # Guided StreamFlow configuration
    logger.debug("Creating the StreamFlow and CWL templates")
    facilities = set()

    aggregate_cwl = AggregateStep()
    cwl_config = CWLConfig()
    main_cwl = MainWorkflow()
    round_cwl = RoundWorkflow()
    streamflow_config = StreamFlowFile()
    training_cwl = TrainingStep()
    logger.debug("StreamFlow and CWL templates created")

    cwl_config.content |= {
        "script_aggregation": {
            "class": "File",
            "path": os.path.join("py_scripts", "aggregation_application.py"),
        },
    }

    # Get executable file
    executable_path = resolve_path(
        check_input(
            "xFFL requires the Python training scripts, that should be present in your local machine\nTraining script path: ",
            "File {} does not exist. Insert an existing file",
            lambda path: os.path.isfile(path),
            is_path=True,
        )
    )
    parser_path = resolve_path(
        check_input(
            "\nIt is necessary to indicate also the arguments parse of the training script\nCustom arguments parser path: ",
            "File {} does not exist. Insert an existing file",
            lambda path: os.path.isfile(path),
            is_path=True,
        )
    )
    parser_name = check_input(
        "\nInput the arguments parser module name: ",
        "The name can contain letters, numbers, dashs or underscores. Moreover, it can start only with a letter or number",
        lambda x: re.match("^[a-zA-Z0-9][a-zA-Z0-9_-]*$", x),
    )

    # Get model
    model_path = resolve_path(
        check_input(
            "\nxFFL requires the Python model, that should be present in your local machine\nModel path: ",
            "File/directory {} does not exist. Insert an existing one",
            lambda path: os.path.exists(path),
            is_path=True,
        )
    )

    # New model name
    new_model_name = check_input(
        "\nName of the new model: ",
        "The name can contain letters, numbers, dashs or underscores. Moreover, it can start only with a letter or number",
        lambda x: re.match("^[a-zA-Z0-9][a-zA-Z0-9_-]*$", x),
    )

    # Local workdir
    local_workdir = check_input(
        "\nLocal workdir (if you leave blank the default $TMPDIR is used): ",
        "Insert a valid directory path or leave it blank for default",
        lambda path: path == "" or resolve_path(path),
    )
    local_workdir = (
        resolve_path(local_workdir)
        if local_workdir
        else os.environ.get("TMPDIR", "/tmp")
    )
    logger.info(f"local_workdir: {local_workdir}")
    streamflow_config.add_root_step(local_workdir)

    # Number of iteration of the training
    num_of_iterations = int(
        check_input(
            "\nNumber of iterations to the federated training: ",
            "Insert an integer",
            lambda x: x.isdigit(),
        )
    )

    cwl_config.content |= {
        "executable": {
            "class": "File",
            "path": executable_path,
            "secondaryFiles": [
                {
                    "class": "File",
                    "path": parser_path,
                }
            ],
        },
        "model": {
            "class": "Directory" if os.path.isdir(model_path) else "File",
            "path": model_path,
        },
        "model_basename": new_model_name,
        "max_rounds": num_of_iterations,
    }

    # Command line arguments extraction from user's parser arguments
    logger.debug(
        f"Dinamically loading ArgumentParser '{parser_name}' from file '{parser_path}'"
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
    except Exception as e:
        raise e

    # Populate training step
    training_cwl.add_inputs(facility_name=None, extra_inputs=arg_to_bidding)

    insert = True
    while insert:

        # Facility name
        facility = check_input(
            "\nType facility's logic facility: ",
            "Facility facility {} already used.",
            lambda facility: facility not in facilities,
        )
        facilities.add(facility)

        # SSH information
        address = input(
            "\nType {}'s frontend node address [IP:port]: ".format(facility)
        )
        username = input(f"{facility}'s username: ")
        ssh_key = check_input(
            f"Path to {facility}'s SSH key file: ",
            "{} does not exists.",
            lambda ssh_key: os.path.exists(ssh_key),
            is_path=True,
        )
        # TODO: add question to data mover if it is available on the facility

        # SLURM template path for the facility
        # TODO: list the needed pragmas
        slurm_template = check_input(
            "\nPath to {}'s SLURM template with the required directives: ".format(
                facility
            ),
            "{} does not exists.",
            lambda path: os.path.exists(path),
            is_path=True,
        )

        # Remote paths
        step_workdir = check_input(
            "\nPath to the facility's working directory: ",
            "The path is not well-formed",
            lambda x: resolve_path(x, is_local_path=False),
            is_path=True,
            is_local_path=False,
        )
        image_path = check_input(
            "\nPath to the facility's image file: ",
            "The path is not well-formed",
            lambda x: resolve_path(x, is_local_path=False),
            is_path=True,
            is_local_path=False,
        )
        dataset_path = check_input(
            "\nPath to the facility's dataset directory: ",
            "The path is not well-formed",
            lambda x: resolve_path(x, is_local_path=False),
            is_path=True,
            is_local_path=False,
        )

        # Creating CWL configuration
        logger.debug("CWL configuration population...")
        main_cwl.add_inputs(facility_name=facility, extra_inputs=arg_to_type)
        round_cwl.add_inputs(facility_name=facility, extra_inputs=arg_to_type)
        cwl_config.add_inputs(
            facility_name=facility,
            extra_inputs={
                "image": {"class": "File", "path": os.path.basename(image_path)},
                "dataset": {
                    "class": "Directory",
                    "path": os.path.basename(dataset_path),
                },
            }
            | arg_to_value,
        )

        # Creating StreamFlow configuration
        logger.debug("StreamFlow configuration population...")
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
                f"dataset_{facility}": os.path.dirname(dataset_path),
                f"image_{facility}": os.path.dirname(image_path),
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

        insert = check_input(
            "Insert another facility? [y/n]: ",
            "Answer {} not accepted.",
            lambda answer: answer.lower() in ["y", "yes", "n", "no"],
        ) in ["y", "yes"]
    round_cwl.update_merge_step()

    # YAML exportation
    # StreamFlow file
    with open(os.path.join(workdir, "streamflow.yml"), "w") as outfile:
        yaml.dump(
            streamflow_config.save(), outfile, default_flow_style=False, sort_keys=False
        )

    # CWL files
    os.makedirs(os.path.join(workdir, "cwl", "clt"))
    with open(os.path.join(workdir, "cwl", "main.cwl"), "w") as outfile:
        outfile.write("#!/usr/bin/env cwl-runner\n")
        yaml.dump(main_cwl.save(), outfile, default_flow_style=False, sort_keys=False)
    with open(os.path.join(workdir, "cwl", "round.cwl"), "w") as outfile:
        outfile.write("#!/usr/bin/env cwl-runner\n")
        yaml.dump(round_cwl.save(), outfile, default_flow_style=False, sort_keys=False)
    with open(os.path.join(workdir, "cwl", "clt", "aggregate.cwl"), "w") as outfile:
        outfile.write("#!/usr/bin/env cwl-runner\n")
        yaml.dump(
            aggregate_cwl.save(), outfile, default_flow_style=False, sort_keys=False
        )
    with open(os.path.join(workdir, "cwl", "clt", "training.cwl"), "w") as outfile:
        outfile.write("#!/usr/bin/env cwl-runner\n")
        yaml.dump(
            training_cwl.save(), outfile, default_flow_style=False, sort_keys=False
        )
    # CWL config file
    with open(os.path.join(workdir, "cwl", "config.yml"), "w") as outfile:
        yaml.dump(cwl_config.save(), outfile, default_flow_style=False, sort_keys=False)
    # Scripts
    shutil.copytree(
        os.path.join(DEFAULT_xFFL_DIR, "workflow", "scripts"),
        os.path.join(workdir, "cwl", "scripts"),
    )
    os.makedirs(os.path.join(workdir, "cwl", "py_scripts"))
    shutil.copy(
        os.path.join(
            DEFAULT_xFFL_DIR, "workflow", "templates", "aggregation_application.py"
        ),
        os.path.join(workdir, "cwl", "py_scripts"),
    )
    return


def main(args: argparse.Namespace) -> int:
    """xFFL project's guided configuration entrypoint

    :param args: Command line arguments
    :type args: argparse.Namespace
    :raises e: Exception raised during the configuration
    :return: Exit code
    :rtype: int
    """
    logger.info(
        "*** Cross-Facility Federated Learning (xFFL) - Guided configuration ***"
    )
    try:
        config(args=args)
    except Exception as e:
        logger.exception(e)
        raise e
    finally:
        logger.info(
            "*** Cross-Facility Federated Learning (xFFL) - Guided configuration ***"
        )
    return 0


if __name__ == "__main__":
    try:
        main(args=config_parser.parse_args())
    except KeyboardInterrupt as e:
        logger.exception(e)
