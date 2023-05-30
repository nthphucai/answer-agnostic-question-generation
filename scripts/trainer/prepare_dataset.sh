#!/bin/bash
set -e

export MODEL_PATH="VietAI/vit5-base"
export OUPUT_DIR="output/data/history/simple/question-answering"
export MAIN_TASK="question-answering"
export SUB_TASK="simple_gcn"

python questgen/dataset/build_dataset.py \
    --main_task $MAIN_TASK \
    --sub_task $SUB_TASK \
    --model_type t5 \
    --model_name_or_path $MODEL_PATH \
    --pretrained_tokenizer_name_or_path $MODEL_PATH \
    --customized_tokenizer_save_path t5_qg_tokenizer \
    --output_dir $OUPUT_DIR \
    --valid_for_qg_only \
    --qg_format highlight_qg_format \
    --dataset_train_path data/mc/history/mcqg_history_list_triples_data_2600_v1.1.2.json \
    --dataset_valid_path data/mc/history/mcqg_history_list_triples_data_300_v1.1.2.json \
    --dataset_test_path data/mc/history/mcqg_history_list_triples_data_300_v1.1.2.json  \
    --max_source_length 512 \
    --max_target_length 128 \
    --train_file_name train_data_hl_t5.pt \
    --valid_file_name valid_data_hl_t5.pt \
    --test_file_name test_data_hl_t5.pt
