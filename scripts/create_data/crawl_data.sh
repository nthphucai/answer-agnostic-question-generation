#!/bin/bash

python questgen/dataset/create_data/crawl_data/crawl_data.py \
    --domain "history" \
    --task "multiple-choice" \
    --source "chatgpt" \
    --data_path data/fschool/t03/crawl_data/temp.json \
    --accounts_path data/accounts.json \
    --database_config_path configs/crawling_database_config.yml \
    --is_preprocessing true
