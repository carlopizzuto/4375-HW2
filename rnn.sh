#!/bin/bash

python rnn.py --hidden_dim 32 --epochs 10 \
    --train_data ./data/training.json \
    --val_data ./data/validation.json