"""Collection of StreamFlow-related template code"""

import posixpath
from collections.abc import MutableMapping
from typing import Any, Optional

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
        ssh_key: str,
        step_workdir: str,
        queue_manager: str | None = None,
        template: str | None = None,
    ) -> None:
        """Adds a new facility deployment to the StreamFlow configuration

        :param facility_name: Facility's name
        :type facility_name: str
        :param address: Facility's address (IP, FQDN)
        :type address: str
        :param username: Username to use to log in the facility
        :type username: str
        :param ssh_key: SSH key to use to log in the facility
        :type ssh_key: str
        :param step_workdir: Directory where to store temporary StreamFlow SSH files
        :type step_workdir: str
        :param queue_manager: Facility's queue manager
        :type template: str | None
        :param template: Facility's file template
        :type template: str | None
        :raises ValueError: If the facility is already present in the StreamFlow configuration
        """
        if facility_name in self.deployments.keys():
            raise ValueError(
                f"Facility {facility_name} is already present in the StreamFlow configuration"
            )

        if template:
            with open(template) as f:
                content = f.read()
                if (
                    "{{streamflow_command}}" not in content
                    and "{{ streamflow_command }}" not in content
                ):
                    sf_placeholder = "{{streamflow_command}}"
                    raise Exception(
                        f"It is necessary to add the '{sf_placeholder}' placeholder in the template {template}"
                    )

        if queue_manager is not None:
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
                    "type": queue_manager,
                    "config": {
                        "services": {"pragma": {"file": template} if template else {}}
                    },
                    "wraps": f"{facility_name}-ssh",
                    "workdir": step_workdir,
                },
            }
        else:
            self.deployments[facility_name] = {
                f"{facility_name}": {
                    "type": "ssh",
                    "config": {
                        "nodes": [address],
                        "username": username,
                        "sshKey": ssh_key,
                        "checkHostKey": False,
                    }
                    | ({"services": {"pragma": template}} if template else {}),
                    "workdir": step_workdir,
                }
            }

    def add_training_step(
        self,
        facility_name: str,
        mapping: MutableMapping[str, str],
    ) -> None:
        """Adds new training step bindings to the StreamFlow configuration

        :param facility_name: Facility's name
        :type facility_name: str
        :param mapping: Mapping between the StreamFlow binding names and their values
        :type mapping: MutableMapping[str, str]
        :raises ValueError: If the facility is already present in the StreamFlow configuration
        """
        if facility_name in self.step_bindings.keys():
            raise ValueError(
                f"Facility {facility_name} is already present in the StreamFlow configuration"
            )

        step_name = posixpath.join(
            posixpath.sep, "iteration", f"training_on_{facility_name}", "client"
        )
        self.step_bindings[facility_name] = [
            self.create_binding(
                name=step_name,
                values={"service": "pragma"},
                location=facility_name,
                _type="step",
            ),
            *(
                self.create_binding(
                    name=posixpath.join(posixpath.sep, name),
                    values={"workdir": value},
                    location=facility_name,
                    _type="port",
                )
                for name, value in mapping.items()
            ),
        ]

    def add_root_step(self, workdir: str):
        self.step_bindings[posixpath.sep] = [
            self.create_binding(
                name=posixpath.sep, values={}, location="local", _type="step"
            )
        ]
        self.deployments["local"] = {
            "local": {"type": "local", "config": {}, "workdir": workdir}
        }
        self.content["workflows"]["xffl"]["bindings"].extend(
            self.step_bindings[posixpath.sep]
        )
        self.content["deployments"] |= self.deployments["local"]

    @staticmethod
    def create_binding(
        name: str,
        values: MutableMapping[str, Any],
        location: Optional[str] = None,
        _type: str = "step",
    ) -> MutableMapping[str, Any]:
        """Creates a new StreamFlow binding

        :param name: Name of the StreamFlow binding
        :type name: str
        :param values: Value of the StreamFlow binding
        :type values: MutableMapping
        :param location: Facility's name, defaults to None
        :type location: Optional[str], optional
        :param _type: Type of StreamFlow binding, defaults to "step"
        :type _type: str
        :return: A well-formatted StreamFlow binding
        :rtype: MutableMapping[str, Any]
        """
        return {
            _type: name,
            "target": {"deployment": location or "local", **values},
        }
