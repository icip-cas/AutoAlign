#!/bin/sh

export DATA_PATH=data/infinite_7m.json
export CONV_TEMPLATE=chatml-with-empty-think
export OUTPUT_DIR=saved_models/qwen3-1.7b_infinite_7m
export MODEL_PATH=/mnt/data1/hf_models/Qwen3-1.7B
export GA=8
export REPORT_TO="wandb"
export LR=5e-6
export DS_CONFIG=configs/zero2.json
export TRAIN_BATCH_SIZE=4
export PACKING="True"
export LAZY="True"
bash scripts/train_sft.sh