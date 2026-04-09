#!/bin/bash

# Activate conda environment (required for torch-npu; no-op outside NPU containers)
source /home/ma-user/miniconda3/bin/activate 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
START_TIME=$SECONDS
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

TP=${TP:-"2"}
PP=${PP:-"2"}
PRECISION=${PRECISION:-"bf16"}
USE_TE=${USE_TE:-"false"}
MG2HF=${MG2HF:-"false"}
HF_CKPT_PATH=${HF_CKPT_PATH:-"Qwen/Qwen2.5-3B-Instruct"}
TARGET_CKPT_PATH=${TARGET_CKPT_PATH:-"./mg_models/Qwen2.5-hf-to-mcore-local-tp${TP}-pp${PP}"}

# Model architecture args are auto-derived from --model-path (HF config.json).
# No need for MODEL_SIZE-based hardcoding.

if [ $MG2HF = true ]; then
    convert_options=" \
                --convert-checkpoint-from-megatron-to-transformers \
                --hf-ckpt-path ${HF_CKPT_PATH}"

elif [ $MG2HF = false ]; then
    convert_options=""
fi

if [ $USE_TE = true ]; then
    te_options=" \
                --transformer-impl transformer_engine \
                "

elif [ $USE_TE = false ]; then
    te_options=" \
                --transformer-impl local \
                "
fi

if [ "$PR" = "fp16" ]; then
    pr_options=" \
		    --fp16"

elif [ "$PR" = "bf16" ]; then
    pr_options=" \
        --bf16"

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
    --use-cpu-initialization \
    ${te_options} \
    ${convert_options} \
    ${pr_options}


ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"