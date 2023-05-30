#!/bin/bash
set -e

python dataset/prepare_sum_data.py \
    --data_path /content/drive/MyDrive/QG/data/full_data/train_viquad.json \
    --model_name_or_path NlpHUST/t5-en-vi-small \
    --length_threshold 1000 \
    --ppl_threshold 4 \
    --output_path /content/drive/MyDrive/QG/data/new_data.json \
    --save_data True \
