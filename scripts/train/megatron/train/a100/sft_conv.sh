#!/bin/bash
# A100 SFT Training Script for AutoAlign Megatron
# Usage: bash run_a100_sft.sh

set -e

echo "=== A100 SFT Training Script ==="
echo "Starting SFT training on GPU 0-7 with TP=2 PP=2 CP=2"

# Environment Setup
# When running inside Docker (autoalign-megatron-nvidia:dev), PATH/PYTHONPATH are
# already configured by the image. When running on the host with conda, set these:
if [ -z "$MEGATRON_LM_PATH" ]; then
  export PATH=/ceph_home/zhangkaiqi2024/luxinyu_data/envs/ata_megatron/bin:$PATH
  export MEGATRON_LM_PATH=/ceph_home/zhangkaiqi2024/luxinyu_data/github/Megatron-LM
  export PYTHONPATH=$MEGATRON_LM_PATH:$PYTHONPATH
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ASYNC_ERROR_HANDLING=0
export NCCL_DEBUG=WARN

# Training Configuration
MASTER_PORT=${MASTER_PORT:-$(shuf -n 1 -i 20000-29999)}
HF_MODEL_PATH="/ceph_home/arknet/hf_models/Qwen/Qwen2.5-7B-Instruct"
CHECKPOINT_PATH="./mg_models/Qwen2.5-7B-Instruct-mcore-te-tp2-pp2"
DATA_PATH="./data/litecoder_sft.json"
SAVE_PATH="./checkpoints/sft/qwen2.5-7b-sft-tp2-pp2-cp2-seq32k"

echo "Environment configured:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  HF_MODEL_PATH: $HF_MODEL_PATH"
echo "  CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "  SAVE_PATH: $SAVE_PATH"

# Create output directory
mkdir -p "$SAVE_PATH"

# Run SFT Training
echo "Starting torchrun..."
torchrun \
  --nproc_per_node 8 \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr localhost \
  --master_port $MASTER_PORT \
  -m autoalign.megatron.entries.sft \
  --model-path "$HF_MODEL_PATH" \
  --load "$CHECKPOINT_PATH" \
  --save "$SAVE_PATH" \
  --data-path "$DATA_PATH" \
  --dataset json \
  --split 100,0,0 \
  --lr 5e-6 \
  --min-lr 0.0 \
  --lr-decay-style cosine \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --lr-warmup-fraction 0.00004 \
  --epochs 3 \
  --micro-batch-size 1 \
  --global-batch-size 4 \
  --seq-length 32768 \
  --max-padding-length 32768 \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 2 \
  --context-parallel-size ${CP_SIZE:-2} \
  --sequence-parallel \
  --use-distributed-optimizer \
  --overlap-grad-reduce \
  --bf16 \
  --use-flash-attn \
  --attention-backend flash \
  --transformer-impl transformer_engine \
  --eod-mask-loss \
  --train-mode sft \
  --log-interval 1 \
  --save-interval 235 \
  --eval-interval 10000 \
  --eval-iters 10

echo "=== Training completed ==="
echo "Checkpoints saved to: $SAVE_PATH"