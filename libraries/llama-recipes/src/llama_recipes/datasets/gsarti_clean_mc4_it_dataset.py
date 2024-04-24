# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import datasets

SAMPLES=100

def get_preprocessed_gsarti_clean_mc4_it(dataset_config, tokenizer, split, path):
    return datasets.load_from_disk(path+"/gsarti_clean_mc4_it_"+split)#.select(list(range(0, SAMPLES)))