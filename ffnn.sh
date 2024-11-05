#!/bin/bash

python ffnn.py --hidden_dim 10 --epochs 5 \
    --train_data ./data/training.json \
    --val_data ./data/validation.json
