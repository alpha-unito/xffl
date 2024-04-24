import os
import sys
import datasets

from transformers import LlamaTokenizer
from itertools import chain

class Concatenator(object):
	def __init__(self, chunk_size=2048):
		self.chunk_size=chunk_size
		self.residual = {"input_ids": [], "attention_mask": []}
		
	def __call__(self, batch):
		concatenated_samples = {
			k: v + list(chain(*batch[k])) for k, v in self.residual.items()
		}

		total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

		if total_length >= self.chunk_size:
			chunk_num = total_length // self.chunk_size
			result = {
				k: [
					v[i : i + self.chunk_size]
					for i in range(0, chunk_num * self.chunk_size, self.chunk_size)
				]
				for k, v in concatenated_samples.items()
			}
			self.residual = {
				k: v[(chunk_num * self.chunk_size) :]
				for k, v in concatenated_samples.items()
			}
		else:
			result = concatenated_samples
			self.residual = {k: [] for k in concatenated_samples.keys()}

		result["labels"] = result["input_ids"].copy()

		return result


def get_preprocessed_dataset(dataset, version, tokenizer, samples, split):
	if samples>0:
		data = datasets.load_dataset(dataset, version, split=datasets.ReadInstruction(split, from_=0, to=samples, unit='abs'), trust_remote_code=True)
	else:
		data = datasets.load_dataset(dataset, version, split=split, trust_remote_code=True)

	if dataset == "gsarti/clean_mc4_it":
		prompt = (
			f"Scrivi un documento: {{text}}{{eos_token}}"
		)
	elif dataset=="mc4" and version=="cs":
		prompt = (
			f"Napište dokument: {{text}}{{eos_token}}"
		)
	else:
		prompt = (
			f"{{text}}{{eos_token}}"
		)
		print("Dataset not supported, not adding any prompt")

	def apply_prompt_template(sample):
		return {
			"input": prompt.format(
				text=sample["text"],
				eos_token=tokenizer.eos_token,
			)
		}
	
	data = data.map(apply_prompt_template, remove_columns=list(data.features))

	data = data.map(
		lambda sample: tokenizer(sample["input"]),
		batched=True,
		remove_columns=list(data.features),
	).map(Concatenator(), batched=True)

	return data

dataset=sys.argv[1]
version=sys.argv[2]
samples=int(sys.argv[3])

tokenizer = LlamaTokenizer.from_pretrained("./tokenizer")
tokenizer.add_special_tokens(
		{
			"pad_token": "<PAD>",
		}
	)
dataset_train = get_preprocessed_dataset(
		dataset,
		version,
		tokenizer,
		samples,
		split="train",
	)

dataset_train.save_to_disk(dataset.replace("/", "_") + '_train')
print(f"--> Training Set Length = {len(dataset_train)}")

dataset_val = get_preprocessed_dataset(
		dataset,
		version,
		tokenizer,
		samples,
		split="validation",
	)
dataset_val.save_to_disk(dataset.replace("/", "_") + '_val')
print(f"--> Validation Set Length = {len(dataset_val)}")
