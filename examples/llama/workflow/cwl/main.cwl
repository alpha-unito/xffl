#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: Workflow

$namespaces:
  cwltool: "http://commonwl.org/cwltool#"
  s: https://schema.org/

$schemas:
 - https://schema.org/version/latest/schemaorg-current-http.rdf


requirements:
  InlineJavascriptRequirement: {}
  SubworkflowFeatureRequirement: {}
  StepInputExpressionRequirement: {}
  
inputs:
  facility_a: string
  repository_a: Directory
  test_samples_a: int
  train_samples_a: int
  gpus_per_node_a: int

  facility_b: string
  repository_b: Directory
  test_samples_b: int
  train_samples_b: int
  gpus_per_node_b: int

  facility_c: string
  repository_c: Directory
  test_samples_c: int
  train_samples_c: int
  gpus_per_node_c: int

  facility_d: string
  repository_d: Directory
  test_samples_d: int
  train_samples_d: int
  gpus_per_node_d: int

  script_train: File
  script_aggregation: File

  model: Directory
  tokenizer: Directory
  epochs: int
  model_basename: string

  max_rounds: int
outputs:
  result:
    type: Directory
    outputSource: iteration/output_model
steps:
  iteration:
    in:
      facility_a: facility_a
      repository_a: repository_a
      train_samples_a: train_samples_a
      test_samples_a: test_samples_a
      gpus_per_node_a: gpus_per_node_a
      facility_b: facility_b
      repository_b: repository_b
      train_samples_b: train_samples_b
      test_samples_b: test_samples_b
      gpus_per_node_b: gpus_per_node_b
      facility_c: facility_c
      repository_c: repository_c
      train_samples_c: train_samples_c
      test_samples_c: test_samples_c
      gpus_per_node_c: gpus_per_node_c
      facility_d: facility_d
      repository_d: repository_d
      train_samples_d: train_samples_d
      test_samples_d: test_samples_d
      gpus_per_node_d: gpus_per_node_d
      script_train: script_train
      script_aggregation: script_aggregation
      model: model
      tokenizer: tokenizer
      epochs: epochs
      model_basename: model_basename
      round:
        default: 0
      max_rounds: max_rounds
    out: 
      [ output_model ]
    requirements:
      cwltool:Loop:
        loopWhen: $(inputs.round < inputs.max_rounds)
        loop:
          round:
            valueFrom: $(inputs.round + 1)
          model: output_model
        outputMethod: last
    run: round.cwl
