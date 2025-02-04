import os

"""Collection of StreamFlow-related template code
"""

from collections.abc import MutableMapping
from typing import Any

from xffl.workflow.templates.streamflow import YamlConfig


class StreamFlowFile(YamlConfig):

    def __init__(self):
        super().__init__()
        self.deployments: MutableMapping[str, Any] = {}
        self.step_bindings: MutableMapping[str, Any] = {}

    @classmethod
    def get_default_content(cls) -> MutableMapping[str, Any]:
        """Returns a basic StreamFlow configuration to be updated and saved into a .yml

        :return: a basic StreamFlow configuration for xFFL
        :rtype: MutableMapping[str, Any]
        """

        return {
            "version": "v1.0",
            "workflows": {
                "xffl": {
                    "type": "cwl",
                    "config": {
                        "file": "cwl/main.cwl",
                        "settings": "cwl/config.yml",
                    },
                    "bindings": [],
                }
            },
            "deployments": {},
        }

    def add_deployment(
        self,
        facility_name: str,
        address: str,
        username: str,
        ssh_key: str,
        step_workdir: str,
        slurm_template: str,
    ) -> None:
        if facility_name in self.deployments.keys():
            raise Exception(f"Facility {facility_name} was already added")
        self.deployments[facility_name] = {
            f"{facility_name}-ssh": {
                "type": "ssh",
                "config": {
                    "nodes": [address],
                    "username": username,
                    "sshKey": ssh_key,
                },
                "workdir": step_workdir,
            },
            facility_name: {
                "type": "slurm",
                "config": {"services": {"pragma": {"file": slurm_template}}},
                "wraps": f"{facility_name}-ssh",
                "workdir": step_workdir,
            },
        }

    def add_step_binding(
        self,
        facility_name: str,
        code_path: str,
        dataset_path: str,
        model_path: str,
        image_path: str,
    ) -> None:
        if facility_name in self.step_bindings.keys():
            raise Exception(f"Facility {facility_name} was already added")
        self.step_bindings[facility_name] = [
            {
                "step": f"/iteration/training_on_{facility_name}",
                "target": [
                    {"deployment": facility_name, "service": "pragma"},
                ],
            },
            {
                "port": f"/repository_{facility_name}",
                "target": {
                    "deployment": facility_name,
                    "workdir": os.path.dirname(code_path),
                },
            },
            {
                "port": f"/dataset_{facility_name}",
                "target": {
                    "deployment": f"{facility_name}",
                    "workdir": os.path.dirname(dataset_path),
                },
            },
            {
                "port": f"/image_{facility_name}",
                "target": {
                    "deployment": facility_name,
                    "workdir": os.path.dirname(image_path),
                },
            },
            {
                # fixme: remote it
                "port": f"/model",
                "target": {
                    "deployment": facility_name,
                    "workdir": os.path.dirname(model_path),
                },
            },
        ]

    def add_inputs(self, facility_name: str) -> None:
        self.content["workflows"]["xffl"]["bindings"].extend(
            self.step_bindings[facility_name]
        )
        self.content["deployments"] |= self.deployments[facility_name]
