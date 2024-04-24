#!/usr/bin/env cwl-runner

cwlVersion: v1.2
class: CommandLineTool

$namespaces:
  cwltool: "http://commonwl.org/cwltool#"
  s: https://schema.org/

$schemas:
 - https://schema.org/version/latest/schemaorg-current-http.rdf

s:author:
  - class: s:Person
    s:identifier: https://orcid.org/0000-0001-9290-2017
    s:email: mailto:iacopo.colonnelli@unito.it
    s:name: Iacopo Colonnelli
  - class: s:Person
    s:identifier: https://orcid.org/0000-0003-1144-3707
    s:email: mailto:robert.birke@unito.it
    s:name: Robert Birke
  - class: s:Person
    s:identifier: https://orcid.org/0009-0006-4862-7429
    s:email: mailto:giulio.malenza@unito.it
    s:name: Giulio Malenza
  - class: s:Person
    s:identifier: https://orcid.org/0000-0002-1887-6911
    s:email: mailto:gianluca.mittone@unito.it
    s:name: Gianluca Mittone
  - class: s:Person
    s:identifier: https://orcid.org/0009-0009-2600-613X
    s:email: mailto:alberto.mulone@unito.it
    s:name: Alberto Mulone
  - class: s:Person
    s:identifier: https://orcid.org/0000-0001-8788-0829
    s:email: mailto:marco.aldinucci@unito.it
    s:name: Marco Aldinucci


s:codeRepository: https://github.com/alpha-unito/xffl
s:dateCreated: "2024-04-20"
s:license: https://spdx.org/licenses/LGPL-3.0-only
s:programmingLanguage: Python


requirements:
  InlineJavascriptRequirement: {}
  ShellCommandRequirement: {}
  EnvVarRequirement:
    envDef:
      TMPDIR: ""

baseCommand: [ "torchrun" ]
arguments:
  - position: 1
    valueFrom: $N_NODE
    prefix: --nnodes
    shellQuote: false
  - position: 2
    valueFrom: $NPROC_PER_NODE
    prefix: --nproc_per_node
    shellQuote: false
  - position: 3
    valueFrom: "8884"
    prefix: --rdzv_id
    shellQuote: false
  - position: 4
    valueFrom: "c10d"
    prefix: --rdzv_backend
    shellQuote: false
  - position: 5 
    valueFrom: $HEAD_NODE_IP:29500
    prefix: --rdzv_endpoint
    shellQuote: false
  - position: 7
    valueFrom: "--enable_fsdp"
    shellQuote: false
  - position: 9
    valueFrom: "$(inputs.input_model.basename)-fine-tuned"
    prefix: --output_dir
    shellQuote: false
  - position: 12
    valueFrom: "1"
    prefix: --num_epochs
    shellQuote: false
  - position: 13
    valueFrom: "4"
    prefix: --num_workers_dataloader
    shellQuote: false
  - position: 14
    valueFrom: $RANDOM
    shellQuote: false
    prefix: --seed
      
inputs:
  train_script:
    type: File
    inputBinding:
      position: 6
      shellQuote: false
  dataset:
    type: string
    inputBinding:
      position: 10
      prefix: --dataset
      shellQuote: false
  dataset_path:
    type: Directory
    inputBinding:
      position: 11
      prefix: --dataset_path
      shellQuote: false
  input_model:
    type: Directory
    inputBinding:
      position: 8
      prefix: --model_name
      shellQuote: false
  tokenizer:
    type: Directory 
    inputBinding:
      position: 16
      prefix: --tokenizer
      shellQuote: false

outputs:
  output_model:
    type: Directory
    outputBinding:
      glob: "$(inputs.input_model.basename)-fine-tuned"