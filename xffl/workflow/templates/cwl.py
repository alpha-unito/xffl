"""Collection of CWL-related template code"""

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Any

from xffl.custom.types import FileLike, FolderLike
from xffl.workflow.config import YamlConfig


class CWLConfig(YamlConfig):
    """Class modelling a CWL configuration file as a YamlConfig object"""

    def __init__(self):
        """Creates a new CWLConfig instance with empty facilities data"""
        super().__init__()

    @classmethod
    def get_default_content(cls) -> MutableMapping[str, Any]:
        """Returns a basic CWL configuration to be updated and saved into a .yml

        :return: Basic CWL configuration for xFFL
        :rtype: MutableMapping[str, Any]
        """
        return {
            "script_train": {"class": "Directory", "path": "scripts"},
        }

    def add_inputs(
        self, facility_name: str, extra_inputs: MutableMapping[str, Any]
    ) -> None:
        """Adds the CWL inputs to the YAML content

        :param facility_name: Facility's name
        :type facility_name: str
        """
        self.content |= {f"facility_{facility_name}": facility_name} | {
            f"{name}_{facility_name}": value for name, value in extra_inputs.items()
        }


class Workflow(ABC):
    """Abstract base class describing workflow-like objects"""

    def __init__(self):
        """Creates a new Workflow instance with empty cwl data"""
        # todo: change mutablemapping to cwl_utils objs (PR #3)
        self.cwl: MutableMapping[str, Any] = type(self).get_default_definition()

    @classmethod
    def get_default_definition(cls) -> MutableMapping[str, Any]:
        """Returns the default workflow definition

        :return: Default workflow definition
        :rtype: MutableMapping[str, Any]
        """
        return {}

    def save(self) -> MutableMapping[str, Any]:
        """Returns the current CWL Workflow definition

        :return: Current CWL Workflow definition
        :rtype: MutableMapping[str, Any]
        """
        return self.cwl

    @abstractmethod
    def add_inputs(
        self, facility_name: str, extra_inputs: MutableMapping[str, Any]
    ) -> None:
        """Add the given extra inputs to the Workflow definition

        :param facility_name: Facility's name
        :type facility_name: str
        :param extra_inputs: Command line argument required by the executable script
        :type extra_inputs: MutableMapping[str, str]
        """
        ...


class AggregateStep(Workflow):
    """Workflow describing the Federated Learning aggregation step"""

    @classmethod
    def get_default_definition(cls) -> MutableMapping[str, Any]:
        """Get the CWL file standard content for a xFFL application aggregation step

        :return: Dict template of a CWL xFFL aggregation step
        :rtype: MutableMapping[str, Any]
        """
        return {
            "cwlVersion": "v1.2",
            "class": "CommandLineTool",
            "requirements": {
                "InlineJavascriptRequirement": {},
                # todo: create the image llm-aggregator on alphaunito DockerHub
                # "DockerRequirement": {"dockerPull": "alphaunito/llm-aggregator"}
            },
            "baseCommand": ["python"],
            "arguments": [
                {
                    "position": 5,
                    "valueFrom": "$(inputs.model_basename)-merged-round$(inputs.round)",
                    "prefix": "--outname",
                }
            ],
            "inputs": {
                "script": {"type": "File", "inputBinding": {"position": 1}},
                "models": {
                    "type": {
                        "type": "array",
                        "items": "Directory",
                        "inputBinding": {"position": 2, "prefix": "--model"},
                    }
                },
                "model_basename": {"type": "string"},
                "round": {"type": "int"},
            },
            "outputs": {
                "new_model": {
                    "type": "Directory",
                    "outputBinding": {
                        "glob": "$(inputs.model_basename)-merged-round$(inputs.round)"
                    },
                }
            },
        }

    def add_inputs(
        self, facility_name: str, extra_inputs: MutableMapping[str, Any]
    ) -> None:
        """Add the given extra inputs to the AggregateStep definition

        :param facility_name: Facility's name
        :type facility_name: str
        :param extra_inputs: Command line argument required by the executable script
        :type extra_inputs: MutableMapping[str, str]
        """
        pass


class MainWorkflow(Workflow):
    """Main description of a Federated Learning workflow"""

    @classmethod
    def get_default_definition(cls) -> MutableMapping[str, Any]:
        """Get the CWL main file standard content for a xFFL application

        :return: main.cwl template
        :rtype: MutableMapping[str, Any]
        """
        return {
            "cwlVersion": "v1.2",
            "class": "Workflow",
            "$namespaces": {"cwltool": "http://commonwl.org/cwltool#"},
            "requirements": {
                "SubworkflowFeatureRequirement": {},
                "InlineJavascriptRequirement": {},
                "StepInputExpressionRequirement": {},
            },
            "inputs": {
                "script_train": "Directory",
                "script_aggregation": "File",
                "model": "Directory",
                "executable": "File",
                "model_basename": "string",
                "max_rounds": "int",
            },
            "outputs": {
                "new_model": {
                    "type": "Directory",
                    "outputSource": "iteration/new_model",
                }
            },
            "steps": {
                "iteration": {
                    "run": "round.cwl",
                    "in": {
                        "script_train": "script_train",
                        "script_aggregation": "script_aggregation",
                        "model": "model",
                        "executable": "executable",
                        "model_basename": "model_basename",
                        "round": {"default": 0},
                        "max_rounds": "max_rounds",
                    },
                    "out": ["new_model"],
                    "requirements": {
                        "cwltool:Loop": {
                            "loopWhen": "$(inputs.round < inputs.max_rounds)",
                            "loop": {
                                "round": {"valueFrom": "$(inputs.round + 1)"},
                                "model": "new_model",
                            },
                            "outputMethod": "last",
                        }
                    },
                }
            },
        }

    def add_inputs(
        self, facility_name: str, extra_inputs: MutableMapping[str, str]
    ) -> None:  # TODO: remove all the redundancy
        """Add the given extra inputs to the MainWorkflow definition

        :param facility_name: Facility's name
        :type facility_name: str
        :param extra_inputs: Command line argument required by the executable script [name: cwl type]
        :type extra_inputs: MutableMapping[str, str]
        """
        self.cwl["inputs"] |= {
            f"image_{facility_name}": "File",
            f"facility_{facility_name}": "string",
            f"dataset_{facility_name}": "Directory",
        } | {f"{name}_{facility_name}": _type for name, _type in extra_inputs.items()}

        self.cwl["steps"]["iteration"]["in"] |= {
            f"image_{facility_name}": f"image_{facility_name}",
            f"facility_{facility_name}": f"facility_{facility_name}",
            f"dataset_{facility_name}": f"dataset_{facility_name}",
        } | {
            f"{name}_{facility_name}": f"{name}_{facility_name}"
            for name in extra_inputs.keys()
        }


class RoundWorkflow(Workflow):
    """Round workflow CWL description"""

    @classmethod
    def get_default_definition(cls) -> MutableMapping[str, Any]:
        """Get the CWL round file standard content for a xFFL application

        :return: round.cwl template
        :rtype: MutableMapping[str, Any]
        """
        return {
            "cwlVersion": "v1.2",
            "class": "Workflow",
            "inputs": {
                "script_train": "Directory",
                "script_aggregation": "File",
                "model": "Directory",
                "executable": "File",
                "model_basename": "string",
                "max_rounds": "int",
                "round": "int",
            },
            "outputs": {
                "new_model": {
                    "type": "Directory",
                    "outputSource": "aggregate/new_model",
                }
            },
            "steps": {
                "merge": {
                    "run": {
                        "class": "ExpressionTool",
                        "inputs": {},
                        "outputs": {"models": "Directory[]"},
                        "expression": [],
                    },
                    "in": {},
                    "out": ["models"],
                },
                "aggregate": {
                    "run": "clt/aggregate.cwl",
                    "in": {
                        "model_basename": "model_basename",
                        "round": "round",
                        "script": "script_aggregation",
                        "models": "merge/models",
                    },
                    "out": ["new_model"],
                },
            },
        }

    @classmethod
    def get_training_step(cls, name: str) -> MutableMapping[str, Any]:
        """Get the CWL file standard content for a xFFL application step

        :param name: Name of the facility on which the step will be executed
        :type name: str
        :return: Dict template of a CWL xFFL step
        :rtype: MutableMapping[str, Any]
        """
        return {
            f"training_on_{name}": {
                "run": "clt/training.cwl",
                "in": {
                    "script": "script_train",
                    "executable": "executable",
                    "model": "model",
                    "facility": f"facility_{name}",
                    "image": f"image_{name}",
                    "dataset": f"dataset_{name}",
                    "model_basename": "model_basename",
                    "round": "round",
                },
                "out": ["new_model"],
            }
        }

    def add_inputs(
        self, facility_name: str, extra_inputs: MutableMapping[str, str]
    ) -> None:
        """Add the given extra inputs to the RoundWorkflow definition

        :param facility_name: Facility's name
        :type facility_name: str
        :param extra_inputs: Command line argument required by the executable script [name: cwl type]
        :type extra_inputs: MutableMapping[str, str]
        """
        self.cwl["inputs"] |= {
            f"facility_{facility_name}": "string",
            f"image_{facility_name}": "File",
            f"dataset_{facility_name}": "Directory",
        } | {f"{name}_{facility_name}": _type for name, _type in extra_inputs.items()}
        self.cwl["steps"] |= RoundWorkflow.get_training_step(facility_name)
        self.cwl["steps"][f"training_on_{facility_name}"]["in"] |= {
            f"{name}": f"{name}_{facility_name}" for name in extra_inputs.keys()
        }
        self.cwl["steps"]["merge"]["in"] |= {
            facility_name: f"training_on_{facility_name}/new_model"
        }
        self.cwl["steps"]["merge"]["run"]["inputs"] |= {facility_name: "Directory"}
        self.cwl["steps"]["merge"]["run"]["expression"].append(
            f"inputs.{facility_name}"
        )

    def update_merge_step(self) -> None:
        """Updates the merge step"""
        self.cwl["steps"]["merge"]["run"]["expression"] = (
            "$({'models': ["
            + ",".join(self.cwl["steps"]["merge"]["run"]["expression"])
            + "] })"
        )


class TrainingStep(Workflow):
    """Workflow modelling a training step"""

    def __init__(self):
        super().__init__()
        self.position = self.get_fst_position()

    @classmethod
    def get_default_definition(cls) -> MutableMapping[str, Any]:
        """Get the CWL file standard content for a xFFL application training step

        :return: Dict template of a CWL xFFL training step
        :rtype: MutableMapping[str, Any]
        """
        return {
            "cwlVersion": "v1.2",
            "class": "CommandLineTool",
            "requirements": {
                "InlineJavascriptRequirement": {},
                "EnvVarRequirement": {
                    "envDef": {
                        "XFFL_MODEL_FOLDER": "$(inputs.model.path)",
                        "XFFL_DATASET_FOLDER": "$(inputs.dataset.path)",
                        "XFFL_IMAGE": "$(inputs.image.path)",
                        "XFFL_FACILITY": "$(inputs.facility)",
                        "XFFL_OUTPUT_FOLDER": "$(runtime.outdir)",
                        "XFFL_EXECUTABLE_FOLDER": "$(inputs.executable.dirname)",
                    }
                },
            },
            "arguments": [
                {
                    "position": 1,
                    "valueFrom": "$(inputs.script.path)/facilitator.sh",
                },
                {
                    "position": 3,
                    "valueFrom": "$(inputs.model_basename)",
                    "prefix": "--output-model",
                },
                {
                    "position": 4,
                    "valueFrom": "$(runtime.outdir)",
                    "prefix": "--output-path",
                },
            ],
            "inputs": {
                "script": {
                    "type": "Directory",
                },
                "executable": {
                    "type": "File",
                    "inputBinding": {"position": 2},
                },
                "image": {
                    "type": "File",
                },
                "facility": {
                    "type": "string",
                },
                "model": {
                    "type": "Directory",
                },
                "dataset": {
                    "type": "Directory",
                },
                "model_basename": {"type": "string"},
                "round": {"type": "int"},  # num of the current iteration
            },
            "outputs": {
                "new_model": {
                    "type": "Directory",
                    "outputBinding": {"glob": "$(inputs.model_basename)"},
                }
            },
        }

    def add_inputs(
        self, facility_name: str, extra_inputs: MutableMapping[str, str]
    ) -> None:
        """Add the given extra inputs to the TrainingStep definition

        :param facility_name: Facility's name
        :type facility_name: str
        :param extra_inputs: Command line argument required by the executable script [name: cwl type]
        :type extra_inputs: MutableMapping[str, str]
        """
        self.cwl["inputs"] |= {
            f"facility": "string",
            f"image": "File",
            f"dataset": "Directory",
        } | {f"{name}": _type for name, _type in extra_inputs.items()}

    def get_fst_position(self) -> int:
        position = 0
        # TODO: change it when cwl_utils will be used
        for _input in self.cwl["inputs"].values():
            if position < (
                curr_pos := _input.get("inputBinding", {}).get("position", 0)
            ):
                position = curr_pos
        for _input in self.cwl["arguments"]:
            if position < (curr_pos := _input.get("position", 0)):
                position = curr_pos
        return position + 1
