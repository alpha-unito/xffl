"""Collection of CWL-related template code
"""

import os
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Any


class YamlConfig:
    def __init__(self):
        self.content: MutableMapping[str, Any] = type(self).get_content()

    @classmethod
    def get_default_content(cls) -> MutableMapping[str, Any]:
        return {}

    def save(self) -> MutableMapping[str, Any]:
        return self.content


class CWLConfig(YamlConfig):

    def __init__(self):
        super().__init__()
        self.facility_data: MutableMapping[str, Any] = {}

    @classmethod
    def get_default_content(cls) -> MutableMapping[str, Any]:
        return {
            "script_train": {"class": "Directory", "path": "scripts"},
        }

    def add_input_values(
        self,
        facility_name: str,
        code_path: str,
        image_path: str,
        dataset_path: str,
        val_batch_size: int,
        train_batch_size: int,
        subsampling: int,
    ) -> None:
        self.facility_data[facility_name] = {
            f"facility_{facility_name}": facility_name,
            f"repository_{facility_name}": {
                "class": "Directory",
                "path": os.path.basename(code_path),
            },
            f"image_{facility_name}": {
                "class": "File",
                "path": os.path.basename(image_path),
            },
            f"dataset_{facility_name}": {
                "class": "Directory",
                "path": os.path.basename(dataset_path),
            },
            f"val_batch_size_{facility_name}": val_batch_size,
            f"train_batch_size_{facility_name}": train_batch_size,
            f"subsampling_{facility_name}": subsampling,
        }

    def add_inputs(self, facility_name: str) -> None:
        self.content |= self.faciltiry_data[facility_name]


class Workflow(YamlConfig, ABC):
    def __init__(self):
        self.cwl: MutableMapping[str, any] = type(self).get_cwl()

    @abstractmethod
    def add_inputs(self, facility_name: str) -> None: ...

    def save(self) -> MutableMapping[str, Any]:
        return self.cwl


class AggregateStep(Workflow):
    @classmethod
    def get_default_content(cls) -> MutableMapping[str, Any]:
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


class MainWorkflow(Workflow):

    @classmethod
    def get_default_content(cls) -> MutableMapping[str, Any]:
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
                "executable": "string",
                "epochs": "int",
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
                        "epochs": "epochs",
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

    def add_inputs(self, facility_name: str) -> None:
        self.content["inputs"] |= {
            f"facility_{facility_name}": "string",
            f"repository_{facility_name}": "Directory",
            f"train_batch_size_{facility_name}": "int",
            f"val_batch_size_{facility_name}": "int",
            f"subsampling_{facility_name}": "int",
            f"repository_{facility_name}": "Directory",
            f"image_{facility_name}": "File",
            f"dataset_{facility_name}": "Directory",
        }
        self.content["steps"]["iteration"]["in"] |= {
            f"facility_{facility_name}": f"facility_{facility_name}",
            f"repository_{facility_name}": f"repository_{facility_name}",
            f"val_batch_size_{facility_name}": f"val_batch_size_{facility_name}",
            f"train_batch_size_{facility_name}": f"train_batch_size_{facility_name}",
            f"subsampling_{facility_name}": f"subsampling_{facility_name}",
            f"repository_{facility_name}": f"repository_{facility_name}",
            f"image_{facility_name}": f"image_{facility_name}",
            f"dataset_{facility_name}": f"dataset_{facility_name}",
        }


class RoundWorkflow(Workflow):

    @classmethod
    def get_default_content() -> MutableMapping[str, Any]:
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
                "executable": "string",
                "epochs": "int",
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
    def get_default_content(cls, name: str) -> MutableMapping[str, Any]:
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
                    # todo: add `wandb` and `seed` inputs
                    "script": "script_train",
                    "executable": "executable",
                    "facility": f"facility_{name}",
                    "train_batch_size": f"train_batch_size_{name}",
                    "val_batch_size": f"val_batch_size_{name}",
                    "subsampling": f"subsampling_{name}",
                    "repository": f"repository_{name}",
                    "image": f"image_{name}",
                    "dataset": f"dataset_{name}",
                    "model": "model",
                    "epochs": "epochs",
                    "model_basename": "model_basename",
                    "round": "round",
                },
                "out": ["output_model"],
            }
        }

    def add_inputs(self, facility_name: str) -> None:
        self.content["inputs"] |= {
            f"facility_{facility_name}": "string",
            f"repository_{facility_name}": "Directory",
            f"train_batch_size_{facility_name}": "int",
            f"val_batch_size_{facility_name}": "int",
            f"subsampling_{facility_name}": "int",
            f"repository_{facility_name}": "Directory",
            f"image_{facility_name}": "File",
            f"dataset_{facility_name}": "Directory",
        }
        self.content["steps"] |= RoundWorkflow.get_training_step(facility_name)
        self.content["steps"]["merge"]["in"] |= {
            facility_name: f"training_on_{facility_name}/output_model"
        }
        self.content["steps"]["merge"]["run"]["inputs"] |= {facility_name: "Directory"}
        self.content["steps"]["merge"]["run"]["expression"].append(
            f"inputs.{facility_name}"
        )


class TrainingStep(Workflow):

    @classmethod
    def get_default_content(cls) -> MutableMapping[str, Any]:
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
                    }
                },
            },
            "arguments": [
                {
                    "position": 1,
                    "valueFrom": "$(inputs.script.path)/facilitator.sh",
                },
                {
                    "position": 8,
                    "valueFrom": "$(inputs.model_basename)-round$(inputs.round)",
                    "prefix": "--output-model",
                },
                {
                    "position": 9,
                    "valueFrom": "$(runtime.outdir)",
                    "prefix": "--output-path",
                },
            ],
            "inputs": {
                "script": {
                    "type": "Directory",
                },
                "executable": {
                    "type": "string",  # todo: fix me. It is a relative path of the `repository` input
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
                "train_batch_size": {
                    "type": "int",
                    "inputBinding": {"position": 3, "prefix": "--train-batch-size"},
                },
                "val_batch_size": {
                    "type": "int",
                    "inputBinding": {"position": 4, "prefix": "--val-batch-size"},
                },
                "subsampling": {
                    "type": "int",
                    "inputBinding": {"position": 5, "prefix": "--subsampling"},
                },
                "seed": {
                    "type": "int",
                    "inputBinding": {"position": 6, "prefix": "--seed"},
                    "default": 42,
                },
                "debug": {
                    "type": "str?",  # value must be "-dbg"
                    "inputBinding": {"position": 7},
                },
                "model_basename": {"type": "string"},
                "round": {"type": "int"},
            },
            "outputs": {
                "output_model": {
                    "type": "Directory",
                    "outputBinding": {
                        "glob": "$(inputs.model_basename)-round$(inputs.round)"
                    },
                }
            },
        }

    def add_inputs(self, facility_name: str) -> None:
        pass
