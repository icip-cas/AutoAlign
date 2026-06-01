#!/bin/bash
# HF <-> Megatron checkpoint conversion for Ascend NPU
set -e

# ==============================
# Ascend Environment
# ==============================
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
START_TIME=$SECONDS
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

TP=${TP:-"2"}
PP=${PP:-"2"}
PRECISION=${PRECISION:-"bf16"}
USE_TE=${USE_TE:-"false"}
MG2HF=${MG2HF:-"false"}
HF_CKPT_PATH=${HF_CKPT_PATH:-"Qwen/Qwen2.5-3B-Instruct"}
TARGET_CKPT_PATH=${TARGET_CKPT_PATH:-"./mg_models/Qwen2.5-hf-to-mcore-tp${TP}-pp${PP}"}

if [ $MG2HF = true ]; then
    convert_options=" \
        --convert-checkpoint-from-megatron-to-transformers \
        --hf-ckpt-path ${HF_CKPT_PATH}"
else
    convert_options=""
fi

# NPU: default to local transformer impl
if [ $USE_TE = true ]; then
    te_options=" \
        --transformer-impl transformer_engine"
else
    te_options=" \
        --transformer-impl local"
fi

if [ "$PRECISION" = "fp16" ]; then
    pr_options="--fp16"
elif [ "$PRECISION" = "bf16" ]; then
    pr_options="--bf16"
fi

DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun ${DISTRIBUTED_ARGS} -m autoalign.megatron.toolkits.checkpoint.qwen.common \
    --model-path ${HF_CKPT_PATH} \
    --load ${HF_CKPT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP} \
    --micro-batch-size 1 \
    --save-interval 1 \
    --seq-length 1 \
    --no-async-tensor-model-parallel-allreduce \
    --no-bias-swiglu-fusion \
    --no-rope-fusion \
    --use-mcore-models \
    --save-safetensors \
    ${te_options} \
    ${convert_options} \
    ${pr_options}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
