# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from llama_recipes.datasets.grammar_dataset.grammar_dataset import get_dataset as get_grammar_dataset
from llama_recipes.datasets.alpaca_dataset import InstructionDataset as get_alpaca_dataset
from llama_recipes.datasets.samsum_dataset import get_preprocessed_samsum as get_samsum_dataset
from llama_recipes.datasets.mc4_cs_dataset import get_preprocessed_mc4_cs as get_mc4_cs_dataset
from llama_recipes.datasets.gsarti_clean_mc4_it_dataset import get_preprocessed_gsarti_clean_mc4_it as get_gsarti_clean_mc4_it_dataset