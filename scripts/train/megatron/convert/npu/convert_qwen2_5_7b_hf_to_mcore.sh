#!/bin/bash
# Convert Qwen2.5-7B-Instruct HF checkpoint to Megatron-Core format (TP=2, PP=2)
set -e

# ==============================
# Environment Variables
# ==============================
export MEGATRON_LM_PATH=/home/ma-user/Megatron-LM
export PYTHONPATH=$PYTHONPATH:/home/ma-user/Megatron-LM:/home/ma-user/MindSpeed:/home/ma-user/Pai-Megatron-Patch

# ==============================
# Paths
# ==============================
HF_CKPT_PATH=/home/ma-user/hf_models/Qwen/Qwen2.5-7B-Instruct
TARGET_CKPT_PATH=./mg_models/Qwen2.5-7B-hf-to-mcore-tp2-pp2
TP=2
PP=2

# ==============================
# Run
# ==============================
HF_CKPT_PATH=$HF_CKPT_PATH \
TARGET_CKPT_PATH=$TARGET_CKPT_PATH \
TP=$TP PP=$PP \
bash scripts/train/megatron/convert/npu/convert_hf_to_mcore.sh
