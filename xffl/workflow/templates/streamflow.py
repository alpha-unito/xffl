"""Collection of StreamFlow-related template code"""

from collections.abc import MutableMapping
from typing import Any, Optional

from xffl.custom.types import FileLike, FolderLike
from xffl.workflow.config import YamlConfig


class StreamFlowFile(YamlConfig):
    """Class modelling a StreamFlow configuration file as a YamlConfig object"""

    def __init__(self):
        """Creates a new StreamFlowFile instance with empty deployments and step bindings"""
        super().__init__()
        self.deployments: MutableMapping[str, Any] = {}
        self.step_bindings: MutableMapping[str, Any] = {}

    @classmethod
    def get_default_content(cls) -> MutableMapping[str, Any]:
        """Returns a basic StreamFlow configuration to be updated and saved into a .yml

        :return: Basic StreamFlow configuration for xFFL
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

    def add_inputs(self, facility_name: str) -> None:
        """Adds a new facility to the StreamFlow configuration file

        :param facility_name: Facility's name
        :type facility_name: str
        """
        self.content["workflows"]["xffl"]["bindings"].extend(
            self.step_bindings[facility_name]
        )
        self.content["deployments"] |= self.deployments[facility_name]

    def add_deployment(
        self,
        facility_name: str,
        address: str,
        username: str,
        ssh_key: FileLike,
        step_workdir: FolderLike,
        slurm_template: FileLike,
    ) -> None:
        """Adds a new facility deployment to the StreamFlow configuration

        :param facility_name: Facility's name
        :type facility_name: str
        :param address: Facility's address (IP, FQDN)
        :type address: str
        :param username: Username to use to log in the facility
        :type username: str
        :param ssh_key: SSH key to use to log in the facility
        :type ssh_key: FileLike
        :param step_workdir: Directory where to store temporary StreamFlow SSH files
        :type step_workdir: FolderLike
        :param slurm_template: Facility's SLURM file template
        :type slurm_template: FileLike
        :raises ValueError: If the facilty is already present in the StreamFlow configuration
        """
        if facility_name in self.deployments.keys():
            raise ValueError(
                f"Facility {facility_name} is already present in the StreamFlow configuration"
            )

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
        mapping: MutableMapping[str, str],
    ) -> None:
        """Adds new step bindings to the StreamFlow configuration

        :param facility_name: Facility's name
        :type facility_name: str
        :param mapping: Mapping between the StreamFlow binding names and their values
        :type mapping: MutableMapping[str, str]
        :raises ValueError: If the facilty is already present in the StreamFlow configuration
        """
        if facility_name in self.step_bindings.keys():
            raise ValueError(
                f"Facility {facility_name} is already present in the StreamFlow configuration"
            )

        self.step_bindings[facility_name] = [
            {
                "step": f"/iteration/training_on_{facility_name}",
                "target": [
                    {"deployment": facility_name, "service": "pragma"},
                ],
            },
        ] + [
            self.create_binding(
                name=name,
                value=value,
                facility=facility_name,
            )
            for name, value in mapping.items()
        ]

    def create_binding(
        self, name: str, value: FolderLike, facility: Optional[str] = None
    ) -> MutableMapping[str, Any]:
        """Creates a new StreamFlow binding

        :param name: Name of the StreamFlow binding
        :type name: str
        :param value: Value of the StreamFlow binding
        :type value: FolderLike
        :param facility: Facility's name, defaults to None
        :type facility: Optional[str], optional
        :return: A well-formattted StreamFlow binding
        :rtype: MutableMapping[str, Any]
        """
        return {
            "port": f"/{name}",
            "target": {
                "deployment": facility,
                "workdir": value,
            },
        }
