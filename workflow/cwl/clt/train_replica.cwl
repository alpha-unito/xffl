cwlVersion: v1.2
class: CommandLineTool
requirements:
  InlineJavascriptRequirement: {}

arguments:
  - position: 4
    valueFrom: "$(runtime.outdir)"
    prefix: --workdir
  - position: 5
    valueFrom: "$(inputs.model_basename)-round$(inputs.round)"
    prefix: --output

inputs:
  script:
    type: File
    inputBinding:
      position: 1
  image:
    type: File 
    inputBinding:
      position: 2
      prefix: --image
  facility:
    type: string
    inputBinding:
      position: 2
      prefix: --facility
  model:
    type: Directory
    inputBinding:
      position: 2
      prefix: --model
  tokenizer:
    type: Directory
    inputBinding:
      position: 3
      prefix: --tokenizer
  dataset:
    type: Directory
    inputBinding:
      position: 3
      prefix: --dataset
  repository:
    type: Directory
    inputBinding:
      position: 3
      prefix: --repository
  train_samples:
    type: int 
    inputBinding:
      position: 3
      prefix: --train
  test_samples:
    type: int 
    inputBinding:
      position: 3
      prefix: --validation
  replica:
    type: int 
    inputBinding:
      position: 4
      prefix: --replica
  epochs:
    type: int 
    inputBinding:
      position: 4
      prefix: --epochs
  model_basename:
    type: string
  round:
    type: int 

outputs:
  output_model:
    type: Directory
    outputBinding:
      glob: "$(inputs.model_basename)-round$(inputs.round)"