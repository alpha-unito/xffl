# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from functools import partial

from llama_recipes.llama_datasets.grammar_dataset.grammar_dataset import get_dataset as get_grammar_dataset
from llama_recipes.llama_datasets.alpaca_dataset import InstructionDataset as get_alpaca_dataset
from llama_recipes.llama_datasets.custom_dataset import get_custom_dataset,get_data_collator
from llama_recipes.llama_datasets.samsum_dataset import get_preprocessed_samsum as get_samsum_dataset
from llama_recipes.llama_datasets.toxicchat_dataset import get_llamaguard_toxicchat_dataset as get_llamaguard_toxicchat_dataset
from llama_recipes.llama_datasets.clean_mc4_it_dataset import get_preprocessed_clean_mc4_it as get_clean_mc4_it_dataset

DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "custom_dataset": get_custom_dataset,
    "llamaguard_toxicchat_dataset": get_llamaguard_toxicchat_dataset,
	"clean_mc4_it_dataset": get_clean_mc4_it_dataset
}

DATALOADER_COLLATE_FUNC = {
    "custom_dataset": get_data_collator
}
