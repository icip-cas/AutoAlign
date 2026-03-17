#!/bin/bash
# A100 DPO Training Script for AutoAlign Megatron
# Usage: bash scripts/train/megatron/train/a100/dpo_conv.sh

set -e

echo "=== A100 DPO Training Script ==="
echo "Starting DPO training on GPU 0-7 with TP=2 PP=2 CP=1 seq=4K"

# Environment Setup
if [ -z "$MEGATRON_LM_PATH" ]; then
  export PATH=/ceph_home/zhangkaiqi2024/luxinyu_data/envs/ata_megatron/bin:$PATH
  export MEGATRON_LM_PATH=/ceph_home/zhangkaiqi2024/luxinyu_data/github/Megatron-LM
  export PYTHONPATH=$MEGATRON_LM_PATH:$PYTHONPATH
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_P2P_DISABLE=1

# Training Configuration
MASTER_PORT=${MASTER_PORT:-$(shuf -n 1 -i 20000-29999)}
HF_MODEL_PATH="/ceph_home/arknet/hf_models/Qwen/Qwen2.5-7B-Instruct"
CHECKPOINT_PATH="./mg_models/Qwen2.5-7B-Instruct-mcore-te-tp2-pp2"
DATA_PATH="./data/ultrafeedback_dpo.json"
SAVE_PATH="./checkpoints/dpo/qwen2.5-7b-dpo-tp2-pp2-seq4k"

echo "Environment configured:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  HF_MODEL_PATH: $HF_MODEL_PATH"
echo "  CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "  DATA_PATH: $DATA_PATH"
echo "  SAVE_PATH: $SAVE_PATH"

mkdir -p "$SAVE_PATH"

echo "Starting torchrun..."
torchrun \
  --nproc_per_node 8 \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr localhost \
  --master_port $MASTER_PORT \
  -m autoalign.megatron.entries.dpo \
  --model-path "$HF_MODEL_PATH" \
  --load "$CHECKPOINT_PATH" \
  --save "$SAVE_PATH" \
  --data-path "$DATA_PATH" \
  --dataset json \
  --template chatml-idsys \
  --split 100,0,0 \
  --lr 5e-7 \
  --min-lr 0.0 \
  --lr-decay-style cosine \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --lr-warmup-fraction 0.1 \
  --epochs 1 \
  --micro-batch-size 1 \
  --global-batch-size 8 \
  --seq-length 4096 \
  --max-padding-length 4096 \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 2 \
  --context-parallel-size 1 \
  --sequence-parallel \
  --use-distributed-optimizer \
  --overlap-grad-reduce \
  --bf16 \
  --use-flash-attn \
  --attention-backend flash \
  --transformer-impl transformer_engine \
  --eod-mask-loss \
  --train-mode dpo \
  --beta 0.1 \
  --loss-type sigmoid \
  --label-smoothing 0.0 \
  --log-interval 1 \
  --save-interval 500 \
  --eval-interval 10000 \
  --eval-iters 10

echo "=== Training completed ==="
echo "Checkpoints saved to: $SAVE_PATH"
