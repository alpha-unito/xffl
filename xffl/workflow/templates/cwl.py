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

    def __init__(self):
        # todo: change mutablemapping to cwl_utils objs (PR #3)
        self.cwl: MutableMapping[str, Any] = type(self).get_default_definition()

    @classmethod
    def get_default_definition(cls) -> MutableMapping[str, Any]:
        return {}

    def save(self) -> MutableMapping[str, Any]:
        return self.cwl

    @abstractmethod
    def add_inputs(
        self, facility_name: str, extra_inputs: MutableMapping[str, str]
    ) -> None: ...


class AggregateStep(Workflow):
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
                "output_model": {
                    "type": "Directory",
                    "outputBinding": {
                        "glob": "$(inputs.model_basename)-merged-round$(inputs.round)"
                    },
                }
            },
        }

    def add_inputs(
        self, facility_name: str, extra_inputs: MutableMapping[str, str]
    ) -> None:
        pass


class MainWorkflow(Workflow):

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
                "output_model": {
                    "type": "Directory",
                    "outputSource": "iteration/output_model",
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
                    "out": ["output_model"],
                    "requirements": {
                        "cwltool:Loop": {
                            "loopWhen": "$(inputs.round < inputs.max_rounds)",
                            "loop": {
                                "round": {"valueFrom": "$(inputs.round + 1)"},
                                "model": "output_model",
                            },
                            "outputMethod": "last",
                        }
                    },
                }
            },
        }

    def add_inputs(
        self, facility_name: str, extra_inputs: MutableMapping[str, str]
    ) -> None:
        self.cwl["inputs"] |= {
            f"image_{facility_name}": "File",
            f"facility_{facility_name}": "string",
            f"dataset_{facility_name}": "Directory",
            f"repository_{facility_name}": "Directory",
        } | {f"{name}_{facility_name}": _type for name, _type in extra_inputs.items()}

        self.cwl["steps"]["iteration"]["in"] |= {
            f"image_{facility_name}": f"image_{facility_name}",
            f"facility_{facility_name}": f"facility_{facility_name}",
            f"dataset_{facility_name}": f"dataset_{facility_name}",
            f"repository_{facility_name}": f"repository_{facility_name}",
        } | {
            f"{name}_{facility_name}": f"{name}_{facility_name}"
            for name in extra_inputs.keys()
        }


class RoundWorkflow(Workflow):

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
                "output_model": {
                    "type": "Directory",
                    "outputSource": "aggregate/output_model",
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
                    "out": ["output_model"],
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
                    "repository": f"repository_{name}",
                    "image": f"image_{name}",
                    "dataset": f"dataset_{name}",
                },
                "out": ["output_model"],
            }
        }

    def add_inputs(
        self, facility_name: str, extra_inputs: MutableMapping[str, str]
    ) -> None:
        self.cwl["inputs"] |= {
            f"facility_{facility_name}": "string",
            f"repository_{facility_name}": "Directory",
            f"image_{facility_name}": "File",
            f"dataset_{facility_name}": "Directory",
        } | {f"{name}_{facility_name}": _type for name, _type in extra_inputs.items()}
        self.cwl["steps"] |= RoundWorkflow.get_training_step(facility_name)
        self.cwl["steps"][f"training_on_{facility_name}"]["in"] |= {
            f"{name}": f"{name}_{facility_name}" for name in extra_inputs.keys()
        }
        # rm -r project ; pip install . ;
        self.cwl["steps"]["merge"]["in"] |= {
            facility_name: f"training_on_{facility_name}/output_model"
        }
        self.cwl["steps"]["merge"]["run"]["inputs"] |= {facility_name: "Directory"}
        self.cwl["steps"]["merge"]["run"]["expression"].append(
            f"inputs.{facility_name}"
        )

    def update_merge_step(self):
        self.cwl["steps"]["merge"]["run"]["expression"] = (
            "$({'models': ["
            + ",".join(self.cwl["steps"]["merge"]["run"]["expression"])
            + "] })"
        )


class TrainingStep(Workflow):

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
                        "CODE_FOLDER": "$(inputs.repository.path)",
                        "MODEL_FOLDER": "$(inputs.model.path)",
                        "DATASET_FOLDER": "$(inputs.dataset.path)",
                        "IMAGE": "$(inputs.image.path)",
                        "FACILITY": "$(inputs.facility)",
                        "OUTPUT_FOLDER": "$(runtime.outdir)",
                        "EXECUTABLE_FOLDER": "$(inputs.executable.dirname)",
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
                "repository": {
                    "type": "Directory",
                },
                "model_basename": {"type": "string"},
                "round": {"type": "int"},  # num of the current iteration
                # "train_batch_size": {
                #     "type": "int",
                #     "inputBinding": {"position": 3, "prefix": "--train-batch-size"},
                # },
                # "val_batch_size": {
                #     "type": "int",
                #     "inputBinding": {"position": 4, "prefix": "--val-batch-size"},
                # },
                # "subsampling": {
                #     "type": "int",
                #     "inputBinding": {"position": 5, "prefix": "--subsampling"},
                # },
                # "seed": {
                #     "type": "int",
                #     "inputBinding": {"position": 6, "prefix": "--seed"},
                #     "default": 42,
                # },
            },
            "outputs": {
                "output_model": {
                    "type": "Directory",
                    "outputBinding": {"glob": "$(inputs.model_basename)"},
                }
            },
        }

    def add_inputs(
        self, facility_name: str, extra_inputs: MutableMapping[str, str]
    ) -> None:
        self.cwl["inputs"] |= {
            f"facility": "string",
            f"repository": "Directory",
            f"image": "File",
            f"dataset": "Directory",
        } | {f"{name}": _type for name, _type in extra_inputs.items()}
