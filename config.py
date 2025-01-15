import argparse
import json
import os
from collections.abc import MutableMapping, Callable
from typing import Any
from pathlib import Path

import yaml


def resolve_path(path: str) -> str:
    """
    Converts a relative, shortened path into an absolute one
    """
    return str(Path(path).absolute())

def get_streamflow_config() -> MutableMapping[str, Any]:
    """
    """
    return {
        "version": "v1.0",
        "workflows": {
            "xffl": {
                "type": "cwl",
                "config": {
                    "file": "cwl/main.cwl",
                    "settings": "cwl/config.yaml",
                },
                "bindings": [],
            }
        },
        "deployments": {},
    }


def check_input(text: str, warning_msg: str, control: Callable, is_path: bool = False) -> str:
    """
    Receives and checks a user input based no the specified condition
    """
    condition = False
    while not condition:
        value = input(text)
        if is_path:
            value = resolve_path(value)
        if not (condition := control(value)):
            print(warning_msg.format(value))
    return value


def main(args):

    workdir = resolve_path(args.workdir)
    if not Path(workdir).exists():
        raise Exception(F"Workdir {workdir} not exist")  
    if not Path(workdir, args.project).exists():
        raise Exception(F"Impossible create project {args.project} directory on workdir {workdir}") 
    workdir = os.path.join(workdir, args.project)
    os.makedirs(workdir)

    insert = True
    facilities = set()

    streamflow_config = get_streamflow_config()
    while insert:

        name = check_input(
            "Type facility's logic name: ",
            "Facility name {} already used.",
            lambda name: name not in facilities,
        )
        facilities.add(name)

        address = input(f"Type {name}'s frontend node address [IP:port]: ")
        username = input(f"{name}'s username: ")

        key = check_input(
            f"Path to {name}'s SSH key file: ",
            "{} does not exists.",
            lambda key: os.path.exists(key),
            is_path=True,
        )

        workdir = input("Path to the facility's working directory: ")

        # todo: list the needed pragmas
        slurm_template = check_input(
            f"Path to {name}'s SLURM template with the required directives: ",
            "{} does not exists.",
            lambda path: os.path.exists(path),
            is_path=True,
        )

        step_config = {
            "step": f"/training_on_{name}",
            "target": [
                {"deployment": name, "service": "pragma"},
                # {
                #     "port": f"/repository_{name}",
                #     "target": {
                #         "deployment": f"{name}-ssh",
                #         "workdir": code_path,
                #     },
                # },
            ],
        }

        deployment_config = {
            f"{name}-ssh": {
                "type": "ssh",
                "config": {
                    "nodes": [address],
                    "username": username,
                    "sshKey": key,
                },
                "workdir": workdir,
            },
            name: {
                "type": "slurm",
                "config": {"services": {"pragma": {"file": slurm_template}}},
            },
        }

        streamflow_config["workflows"]["xffl"]["bindings"].append(step_config)
        streamflow_config["deployments"] |= deployment_config

        if args.verbose:
            print(f"Inserted the following record for {name} in the StreamFlow file:", json.dumps(step_config), json.dumps(deployment_config), sep="\n")

        insert = check_input(
            "Insert another facility? [y/n]: ",
            "Answer {} not accepted.",
            lambda answer: answer.lower() in ["y", "yes", "n", "no"],
        ) in ["y", "yes"]

    with open(os.path.join(workdir, "streamflow.yml"), "w") as outfile:
        yaml.dump(streamflow_config, outfile, default_flow_style=False, sort_keys=False)
    

if __name__ == "__main__":
    print("*** Cross-Facility Federated Learning (xFFL) - Guided configuration ***\n")
    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-w", "--workdir", help="Insert working directory path", type=str, required=True)
        parser.add_argument("-p", "--project", help="Insert a project name", type=str, required=True)
        parser.add_argument("-v", "--verbose", help="Increase verbosity level", action="store_true")
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print()
    pass

    print("*** Cross-Facility Federated Learning (xFFL) - Guided configuration terminated ***\n")
