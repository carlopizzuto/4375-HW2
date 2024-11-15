#!/bin/bash

# Get command line arguments for hidden_dim and epochs
hidden_dim=${1:-32}  # Default to 32 if not provided
epochs=${2:-10}      # Default to 10 if not provided


python rnn.py --hidden_dim $hidden_dim --epochs $epochs \
    --train_data ./data/training.json \
    --val_data ./data/validation.json \
    --test_data ./data/test.json