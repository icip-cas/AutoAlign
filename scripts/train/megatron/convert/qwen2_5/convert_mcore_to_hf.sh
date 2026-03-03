#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
START_TIME=$SECONDS
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)


TP=${TP:-"2"}
PP=${PP:-"2"}
MG_MODEL_PATH=${MG_MODEL_PATH:-"./checkpoints/sft/checkpoint/sft-mcore-qwen2_5-3B-lr-5e-6-minlr-0.0-bs-4-gbs-16-seqlen-4096-pr-bf16-tp-2-pp-2-cp-1-ac-none-do-true-sp-false-ti-10000-wf-.00016"}
HF_CKPT_PATH=${HF_CKPT_PATH:-"Qwen/Qwen2.5-3B-Instruct"}
PRECISION=${PRECISION:-"fp32"}
USE_TE=${USE_TE:-"true"}
MG2HF=${MG2HF:-"true"}
TARGET_CKPT_PATH=${TARGET_CKPT_PATH:-"./hf_models_from_mg/Qwen2.5-hf-to-mcore-te-tp${TP}-pp${PP}"}

# Model architecture args are auto-derived from --model-path (HF config.json).

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
    --load ${MG_MODEL_PATH} \
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
