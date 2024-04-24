import shutil
import sys
import os

from transformers import LlamaForCausalLM
from torch import nn

shutil.unpack_archive(sys.argv[1])
shutil.unpack_archive(sys.argv[2])
os.remove(sys.argv[1])
os.remove(sys.argv[2])

model_1=LlamaForCausalLM.from_pretrained(sys.argv[1].split("/")[-1].replace(".zip", ""))
model_2=LlamaForCausalLM.from_pretrained(sys.argv[2].split("/")[-1].replace(".zip", ""))

def average_parameters(param_1, param_2):
	return nn.Parameter((param_1+param_2)/2)

for (_, value_1), (_, value_2) in zip(model_1.named_parameters(), model_1.named_parameters()):
	value_1=average_parameters(value_1, value_2)

model_1.half().save_pretrained(sys.argv[3].replace(".zip", ""), safe_serialization=False)
shutil.make_archive(sys.argv[3].replace(".zip", ""), 'zip')

os.remove(sys.argv[1].split("/")[-1].replace(".zip", ""))
os.remove(sys.argv[2].split("/")[-1].replace(".zip", ""))
os.remove(sys.argv[3].replace(".zip", ""))
