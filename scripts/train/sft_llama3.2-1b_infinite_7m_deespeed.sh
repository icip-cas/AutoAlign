#!/bin/sh

export DATA_PATH=data/infinite_7m.json
export CONV_TEMPLATE=llama-3-instruct
export OUTPUT_DIR=saved_models/llama-3.2_1b_infinite_7m
export MODEL_PATH=/mnt/data2/hf_models/Llama-3.2-1B
export GA=8
export REPORT_TO="wandb"
export LR=5e-6
export DS_CONFIG=configs/zero2.json
export TRAIN_BATCH_SIZE=4
export PACKING="True"
export LAZY="True"
bash scripts/train_sft.sh