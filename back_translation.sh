#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES="0"

model_path="/mnt/ceph_home/xudong2022/auto-alignment/models/Humback-Myx"
data_filepath="/mnt/ceph_home/xudong2022/auto-alignment/data/Humback/falcon-refinedweb-sampled.jsonl"
save_filepath="outputs/m1/unlabelled_gen_instruction.jsonl"
prompt_column_name="content"

python src/autoalign/data/instruction/back_translation.py \
    --reverse \
    --model_path=${model_path} \
    --data_filepath=${data_filepath} \
    --save_filepath=${save_filepath} \
    --prompt_column_name=${prompt_column_name} \
    --tensor_parallel_size=1
