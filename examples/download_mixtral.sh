#!/bin/bash

# Requires huggingface

echo "Mixtral-8x7B-v0.1 model download..."
huggingface-cli login
huggingface-cli download --local-dir mixtral-8x7b-v0.1 mistralai/Mixtral-8x7B-v0.1