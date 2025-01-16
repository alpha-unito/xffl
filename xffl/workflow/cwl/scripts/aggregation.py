import argparse
import torch
from transformers import LlamaForCausalLM
from torch import nn
from tqdm import tqdm
 

def main(args):
	print("Starting aggreagation...")
	buffer_model=None
	state_dict_list=[]
	for index, model_path in enumerate(args.model):
		print(f"\t Loading model {index+1} of {len(args.model)} models...")
		buffer_model = LlamaForCausalLM.from_pretrained(model_path)
		state_dict_list.append(buffer_model.state_dict())

	def average_parameters(values):
		return sum(values)/len(values)

	print(f"Averaging {len(state_dict_list[0])} layers of {len(state_dict_list)} models...")
	for key in tqdm(state_dict_list[0]):
		state_dict_list[0][key]=average_parameters([state_dict[key] for state_dict in state_dict_list])
	print("Loading the new set of weights...")
	buffer_model.load_state_dict(state_dict_list[0])
	print(f"Saving the aggregated model to {args.outname}")
	buffer_model.half().save_pretrained(args.outname, safe_serialization=False)
	print("Done!")

if __name__ == "__main__":
	try:
		parser = argparse.ArgumentParser()
		parser.add_argument('-m', '--model', action='append', help='Input path of a local model', type=str, required=True)
		parser.add_argument('-o', '--outname', help='Output path of the aggregated model', type=str, required=True)
		args = parser.parse_args()
		main(args)
	except KeyboardInterrupt:
		print("Interrupted!")
	pass
