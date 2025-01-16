# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import datasets

def get_preprocessed_clean_mc4_it(dataset_config, tokenizer, split):

    if split=="train":
        dataset = datasets.load_from_disk(dataset_config.train_path)
        if dataset_config.train_samples > 0 and dataset_config.train_samples < len(dataset):
            dataset = dataset.select(list(range(0, dataset_config.train_samples)))
    else:
        dataset = datasets.load_from_disk(dataset_config.train_path) # Per il momento serve usare il training set come test per avere più samples
        if dataset_config.test_samples > 0 and dataset_config.test_samples < len(dataset):
            dataset = dataset.select(list(range(dataset_config.train_samples, dataset_config.train_samples + dataset_config.test_samples)))
        
    return dataset
