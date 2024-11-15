#!/bin/bash

python ffnn.py --hidden_dim 5 --epochs 5 \
    --train_data ./data/training.json \
    --val_data ./data/validation.json \
    --test_data ./data/test.json \
    --do_train \
    --do_infer
