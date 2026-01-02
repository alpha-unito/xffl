import json
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, cast

import cwl_utils.parser.cwl_v1_2 as cwl
import yaml

from xffl.utils.constants import XFFL_CACHE_DIR
from xffl.workflow.config import YamlConfig

logger: Logger = getLogger(__name__)


def _dump_cwl(path: Path, data: MutableMapping[str, Any]) -> None:
    """Helper to write CWL files with the correct shebang."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("#!/usr/bin/env cwl-runner\n")
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def cwl_client_record_type() -> cwl.CWLRecordSchema:
    return cwl.CWLRecordSchema(
        type_="record",
        fields=[
            cwl.CWLRecordField(name="dataset", type_="Directory"),
            cwl.CWLRecordField(name="facility", type_="string"),
            cwl.CWLRecordField(name="image", type_="File"),
        ],
    )


class CWLConfig(YamlConfig):
    """Class modelling a CWL configuration file as a YamlConfig object"""

    @classmethod
    def get_default_content(cls) -> MutableMapping[str, Any]:
        return {"script_train": {"class": "Directory", "path": "scripts"}}

    def add_inputs(
        self, facility_name: str, extra_inputs: MutableMapping[str, Any] = None
    ) -> None:
        extra = extra_inputs or {}
        self.content |= {f"facility_{facility_name}": facility_name} | {
            f"{name}_{facility_name}": value for name, value in extra.items()
        }


class Workflow(ABC):
    """Abstract base class describing workflow-like objects"""

    def __init__(self):
        self.cwl: cwl.Process = type(self).get_default_definition()
        self.workflow_dir: Path = self._workdir()
        self.workflow_dir.mkdir(parents=True, exist_ok=True)

    def _workdir(self) -> Path:
        return Path(XFFL_CACHE_DIR, "workflow")

    @classmethod
    @abstractmethod
    def get_default_definition(cls) -> cwl.Process:
        """Subclasses must define their CWL structure here."""
        pass

    @abstractmethod
    def dumps(self) -> None:
        """Subclasses define where and how to save their CWL files."""
        pass

    def _safe_dump(self, filename: str) -> None:
        """Internal helper to avoid code duplication in subclasses."""
        target_path = self.workflow_dir / filename
        if not target_path.exists():
            _dump_cwl(target_path, self.save())

    def save(self) -> MutableMapping[str, Any]:
        return self.cwl.save(top=True)


class AggregateStep(Workflow):
    def _workdir(self) -> Path:
        return super()._workdir() / "clt"

    def dumps(self) -> None:
        self._safe_dump("aggregate.cwl")

    @classmethod
    def get_default_definition(cls) -> cwl.Process:
        """Get the CWL file standard content for a xFFL application aggregation step

        :return: CWL object instance of the xFFL aggregation step
        :rtype: cwl.Process()
        """
        return cwl.CommandLineTool(
            arguments=[
                cwl.CommandLineBinding(
                    position=3,
                    prefix="--outname",
                    valueFrom="$(inputs.model_basename)-merged-round$(inputs.round)",
                )
            ],
            baseCommand=["python"],
            cwlVersion="v1.2",
            id="aggregate",
            inputs=[
                cwl.CommandInputParameter(
                    id="script",
                    type_="File",
                    inputBinding=cwl.CommandLineBinding(position=1),
                ),
                cwl.CommandInputParameter(
                    id="models",
                    type_=cwl.CommandInputArraySchema(
                        name="models",
                        type_="array",
                        items="Directory",
                        inputBinding=cwl.CommandLineBinding(
                            prefix="--model", position=2
                        ),
                    ),
                ),
                cwl.CommandInputParameter(
                    id="model_basename",
                    type_="string",
                ),
                cwl.CommandInputParameter(
                    id="round",
                    type_="int",
                ),
            ],
            outputs=[
                cwl.CommandOutputParameter(
                    id="new_model",
                    type_="Directory",
                    outputBinding=cwl.CommandOutputBinding(
                        glob="$(inputs.model_basename)-merged-round$(inputs.round)"
                    ),
                )
            ],
            requirements=[
                cwl.InlineJavascriptRequirement(),
            ],
        )


class MainWorkflow(Workflow):
    def __init__(self, num_clients: int):
        super().__init__()
        self.num_clients: int = num_clients
        self._add_clients()

    def _add_clients(self) -> None:
        workflow_cwl = cast(cwl.Workflow, self.cwl)
        record = cwl_client_record_type()

        for i in range(self.num_clients):
            workflow_cwl.inputs.extend(
                [
                    cwl.WorkflowInputParameter(id=f"{f.name}_{i}", type_=f.type_)
                    for f in record.fields
                ]
            )
            workflow_cwl.steps.append(
                cwl.WorkflowStep(
                    id=f"compact_{i}",
                    in_=[
                        cwl.WorkflowStepInput(id=f.name, source=f"{f.name}_{i}")
                        for f in record.fields
                    ],
                    out=[cwl.WorkflowStepOutput(id="client")],
                    run="et/compact.cwl",
                )
            )

        # Safer lookup for the loop step
        loop_step = next((s for s in workflow_cwl.steps if s.id == "loop"), None)
        if loop_step:
            loop_step.in_.append(
                cwl.WorkflowStepInput(
                    id="clients",
                    source=[f"compact_{i}/client" for i in range(self.num_clients)],
                    linkMerge="merge_flattened",
                )
            )

    def dumps(self) -> None:
        self._safe_dump(f"main_{self.num_clients}_clients.cwl")
        CompactStep().dumps()
        RoundWorkflow().dumps()

    @classmethod
    def get_default_definition(cls) -> cwl.Process:
        loading_options = cwl.LoadingOptions(
            namespaces={"cwltool": "http://commonwl.org/cwltool#"}
        )
        return cwl.Workflow(
            cwlVersion="v1.2",
            id="main",
            inputs=[
                cwl.WorkflowInputParameter(id="script_train", type_="Directory"),
                cwl.WorkflowInputParameter(id="script_aggregation", type_="File"),
                cwl.WorkflowInputParameter(id="model", type_="Directory"),
                cwl.WorkflowInputParameter(id="executable", type_="File"),
                cwl.WorkflowInputParameter(id="model_basename", type_="string"),
                cwl.WorkflowInputParameter(id="max_rounds", type_="int"),
            ],
            loadingOptions=loading_options,
            outputs=[
                cwl.WorkflowOutputParameter(
                    id="new_model",
                    type_="Directory",
                    outputSource="loop/new_model",
                )
            ],
            steps=[
                cwl.WorkflowStep(
                    id="loop",
                    in_=[
                        cwl.WorkflowStepInput(
                            id="script_train",
                            source="script_train",
                        ),
                        cwl.WorkflowStepInput(
                            id="script_aggregation",
                            source="script_aggregation",
                        ),
                        cwl.WorkflowStepInput(
                            id="model",
                            source="model",
                        ),
                        cwl.WorkflowStepInput(
                            id="executable",
                            source="executable",
                        ),
                        cwl.WorkflowStepInput(
                            id="model_basename",
                            source="model_basename",
                        ),
                        cwl.WorkflowStepInput(
                            default=0,
                            id="round",
                        ),
                        cwl.WorkflowStepInput(
                            id="max_rounds",
                            source="max_rounds",
                        ),
                    ],
                    out=[cwl.WorkflowStepOutput(id="new_model")],
                    requirements=[
                        cwl.Loop(
                            loadingOptions=loading_options,
                            loop=[
                                cwl.LoopInput(id="model", loopSource="new_model"),
                                cwl.LoopInput(
                                    id="round",
                                    loopSource="round",
                                    valueFrom="$(inputs.round + 1)",
                                ),
                            ],
                            loopWhen="$(inputs.round < inputs.max_rounds)",
                            outputMethod="last",
                        )
                    ],
                    run="round.cwl",
                ),
            ],
            requirements=[
                cwl.InlineJavascriptRequirement(),
                cwl.MultipleInputFeatureRequirement(),
                cwl.StepInputExpressionRequirement(),
                cwl.SubworkflowFeatureRequirement(),
            ],
        )


class RoundWorkflow(Workflow):
    def dumps(self) -> None:
        self._safe_dump("round.cwl")
        AggregateStep().dumps()
        TrainingStep().dumps()

    @classmethod
    def get_default_definition(cls) -> cwl.Process:
        check_step = cwl.CommandLineTool(
            baseCommand=["true"],
            cwlVersion="v1.2",
            id="check",
            inputs=[
                cwl.CommandInputParameter(
                    id="model",
                    type_="Directory",
                ),
            ],
            outputs=[],
        )

        training_subworkflow = cwl.Workflow(
            id="training",
            inputs=[
                cwl.WorkflowInputParameter(id="script", type_="Directory"),
                cwl.WorkflowInputParameter(id="model", type_="Directory"),
                cwl.WorkflowInputParameter(id="executable", type_="File"),
                cwl.WorkflowInputParameter(id="facility", type_="string"),
                cwl.WorkflowInputParameter(id="client",type_=cwl_client_record_type()),
                cwl.WorkflowInputParameter(id="model_basename", type_="string"),
                cwl.WorkflowInputParameter(id="round", type_="int"),
            ],
            outputs=[
                cwl.WorkflowOutputParameter(
                    id="new_model",
                    type_="Directory",
                    outputSource="client/new_model",
                )
            ],
            steps=[
                cwl.WorkflowStep(
                    id="client",
                    in_=[
                        cwl.WorkflowStepInput(
                            id="executable",
                            source="executable",
                        ),
                        cwl.WorkflowStepInput(
                            id="model",
                            source="model",
                        ),
                        cwl.WorkflowStepInput(
                            id="script",
                            source="script",
                        ),
                        cwl.WorkflowStepInput(
                            id="facility",
                            valueFrom="$(inputs.client.facility)",
                        ),
                        cwl.WorkflowStepInput(
                            id="client",
                            source="client",
                        ),
                        cwl.WorkflowStepInput(
                            id="model_basename",
                            source="model_basename",
                        ),
                        cwl.WorkflowStepInput(
                            id="round",
                            source="round",
                        ),
                    ],
                    out=[cwl.WorkflowStepOutput(id="new_model")],
                    run="clt/training.cwl",
                ),
                cwl.WorkflowStep(
                    id="check",
                    in_=[
                        cwl.WorkflowStepInput(
                            id="model",
                            source="client/new_model",
                        ),
                    ],
                    out=[],
                    run=check_step,
                ),
            ],
        )

        return cwl.Workflow(
            cwlVersion="v1.2",
            id="round",
            inputs=[
                cwl.WorkflowInputParameter(id="script_train", type_="Directory"),
                cwl.WorkflowInputParameter(id="script_aggregation", type_="File"),
                cwl.WorkflowInputParameter(id="model", type_="Directory"),
                cwl.WorkflowInputParameter(id="executable", type_="File"),
                cwl.WorkflowInputParameter(id="model_basename", type_="string"),
                cwl.WorkflowInputParameter(id="max_rounds", type_="int"),
                cwl.WorkflowInputParameter(id="round", type_="int"),
                cwl.WorkflowInputParameter(
                    id="clients",
                    type_=cwl.CWLArraySchema(
                        type_="array", items=cwl_client_record_type()
                    ),
                ),
            ],
            outputs=[
                cwl.WorkflowOutputParameter(
                    id="new_model",
                    type_="Directory",
                    outputSource="aggregate/new_model",
                )
            ],
            steps=[
                cwl.WorkflowStep(
                    id="aggregate",
                    in_=[
                        cwl.WorkflowStepInput(
                            id="model_basename",
                            source="model_basename",
                        ),
                        cwl.WorkflowStepInput(
                            id="round",
                            source="round",
                        ),
                        cwl.WorkflowStepInput(
                            id="script",
                            source="script_aggregation",
                        ),
                        cwl.WorkflowStepInput(
                            id="models",
                            source="training/new_model",
                        ),
                    ],
                    out=[cwl.WorkflowStepOutput(id="new_model")],
                    run="clt/aggregate.cwl",
                ),
                cwl.WorkflowStep(
                    id="training",
                    in_=[
                        cwl.WorkflowStepInput(
                            id="executable",
                            source="executable",
                        ),
                        cwl.WorkflowStepInput(
                            id="model",
                            source="model",
                        ),
                        cwl.WorkflowStepInput(
                            id="script",
                            source="script_train",
                        ),
                        cwl.WorkflowStepInput(
                            id="client",
                            source="clients",
                        ),
                        cwl.WorkflowStepInput(
                            id="model_basename",
                            source="model_basename",
                        ),
                        cwl.WorkflowStepInput(
                            id="round",
                            source="round",
                        ),
                    ],
                    out=[cwl.WorkflowStepOutput(id="new_model")],
                    run=training_subworkflow,
                    scatter=["client"]
                ),
            ],
            requirements=[
                cwl.ScatterFeatureRequirement()
            ]
        )


class TrainingStep(Workflow):
    """Workflow modelling a training step"""

    def _workdir(self) -> Path:
        return super()._workdir() / "clt"

    def dumps(self):
        self._safe_dump("training.cwl")

    @classmethod
    def get_default_definition(cls) -> cwl.Process:
        """Get the CWL file standard content for a xFFL application training step

        :return: CWL object instance of a xFFL training step
        :rtype: cwl.Process
        """
        return cwl.CommandLineTool(
            arguments=[
                cwl.CommandLineBinding(
                    position=1,
                    valueFrom="$(inputs.script.path)/facilitator.sh",
                )
            ],
            cwlVersion="v1.2",
            id="training",
            inputs=[
                cwl.CommandInputParameter(
                    id="script",
                    type_="Directory",
                ),
                cwl.CommandInputParameter(
                    id="executable",
                    type_="File",
                    inputBinding=cwl.CommandLineBinding(
                        position=2,
                        prefix="--executable",
                    ),
                ),
                cwl.CommandInputParameter(
                    id="facility",
                    type_="string",
                    inputBinding=cwl.CommandLineBinding(
                        position=4, prefix="--facility"
                    ),
                ),
                cwl.CommandInputParameter(
                    id="model",
                    type_="Directory",
                    inputBinding=cwl.CommandLineBinding(position=5, prefix="--model"),
                ),
                cwl.CommandInputParameter(
                    id="client",
                    type_=cwl.RecordSchema(
                        type_="record",
                        fields=[
                            cwl.CommandInputRecordField(
                                name="dataset",
                                type_="Directory",
                                inputBinding=cwl.CommandLineBinding(
                                    position=6, prefix="--dataset"
                                ),
                            ),
                            cwl.CommandInputRecordField(
                                name="facility", type_="string"
                            ),
                            cwl.CommandInputRecordField(
                                name="image",
                                type_="File",
                                inputBinding=cwl.CommandLineBinding(
                                    position=3,
                                    prefix="--singularity-image",
                                ),
                            ),
                        ],
                    ),
                ),
                cwl.CommandInputParameter(
                    id="model_basename",
                    type_="string",
                    inputBinding=cwl.CommandLineBinding(
                        position=7, prefix="--output-model"
                    ),
                ),
                cwl.CommandInputParameter(
                    id="round",
                    type_="int",
                ),
            ],
            outputs=[
                cwl.CommandOutputParameter(
                    id="new_model",
                    type_="Directory",
                    outputBinding=cwl.CommandOutputBinding(
                        glob="$(inputs.model_basename)"
                    ),
                )
            ],
            requirements=[
                cwl.InlineJavascriptRequirement(),
                cwl.EnvVarRequirement(
                    envDef=[
                        cwl.EnvironmentDef(
                            envName="XFFL_OUTDIR",
                            envValue="$(runtime.outdir)",
                        ),
                        cwl.EnvironmentDef(
                            envName="XFFL_TMPDIR",
                            envValue="$(runtime.tmpdir)",
                        ),
                    ]
                ),
            ],
        )


class CompactStep(Workflow):
    def _workdir(self) -> Path:
        return super()._workdir() / "et"

    def dumps(self):
        self._safe_dump("compact.cwl")

    @classmethod
    def get_default_definition(cls) -> cwl.Process:
        record = cwl_client_record_type()
        expr = json.dumps(
            {"client": {field.name: f"inputs.{field.name}" for field in record.fields}}
        )
        for field in record.fields:
            expr = expr.replace(f'"inputs.{field.name}"', f"inputs.{field.name}")
        return cwl.ExpressionTool(
            cwlVersion="v1.2",
            id="compact",
            inputs=[
                cwl.WorkflowInputParameter(id=field.name, type_=field.type_)
                for field in record.fields
            ],
            outputs=[
                cwl.CommandOutputParameter(id="client", type_=cwl_client_record_type())
            ],
            expression=f"$({expr})",
        )
