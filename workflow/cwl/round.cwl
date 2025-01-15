#!/usr/bin/env cwl-runner

cwlVersion: v1.2
class: Workflow

$namespaces:
  cwltool: "http://commonwl.org/cwltool#"
  s: https://schema.org/

$schemas:
 - https://schema.org/version/latest/schemaorg-current-http.rdf

requirements:
  SubworkflowFeatureRequirement: {}
  MultipleInputFeatureRequirement: {}

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

  round: int
  max_rounds: int

outputs:
  output_model:
    type: Directory
    outputSource: aggregate/output_model

steps:
  train_dataset_a:
    run: clt/train_replica.cwl
    in:
      script: script_train
      facility: facility_a
      train_samples: train_samples_a
      test_samples: test_samples_a
      repository: repository_a
      replica: gpus_per_node_a
      model: model
      tokenizer: tokenizer
      epochs: epochs
      model_basename: model_basename
      round: round
    out:
      [ output_model ]

  train_dataset_b:
    run: clt/train.cwl    
    in:
      script: script_train
      facility: facility_b
      train_samples: train_samples_b
      test_samples: test_samples_b
      repository: repository_b
      model: model
      tokenizer: tokenizer
      epochs: epochs
      model_basename: model_basename
      round: round
    out:
      [ output_model ]


  train_dataset_c:
    run: clt/train_replica.cwl    
    in:
      script: script_train
      facility: facility_c
      train_samples: train_samples_c
      test_samples: test_samples_c
      repository: repository_c
      replica: gpus_per_node_c
      model: model
      tokenizer: tokenizer
      epochs: epochs
      model_basename: model_basename
      round: round
    out:
      [ output_model ]
  
  train_dataset_d:
    run: clt/train_replica.cwl    
    in:
      script: script_train
      facility: facility_d
      train_samples: train_samples_d
      test_samples: test_samples_d
      repository: repository_d
      replica: gpus_per_node_d
      model: model
      tokenizer: tokenizer
      epochs: epochs
      model_basename: model_basename
      round: round
    out:
      [ output_model ]

  aggregate:
    run: clt/aggregate.cwl
    in:
      model_basename: model_basename
      round: round 
      script: script_aggregation
      models:
        source:
          - train_dataset_a/output_model
          - train_dataset_b/output_model
          - train_dataset_c/output_model
          - train_dataset_d/output_model 
    out:
      [ output_model ]
