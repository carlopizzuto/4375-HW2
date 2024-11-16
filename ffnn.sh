#!/bin/bash

hidden_dim=${1:-10}  # Default to 10 if not provided
epochs=${2:-1}      # Default to 1 if not provided

python ffnn.py --hidden_dim $hidden_dim --epochs $epochs \
    --train_data ./data/training.json \
    --val_data ./data/validation.json \
    --test_data ./data/test.json \
    --do_train \
    --do_infer
