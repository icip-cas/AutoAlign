#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
START_TIME=$SECONDS
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)


MODEL_SIZE=${MODEL_SIZE:-"3B"}
TP=${TP:-"2"}
PP=${PP:-"2"}
MG_MODEL_PATH=${MG_MODEL_PATH:-"./checkpoints/sft/checkpoint/sft-mcore-qwen2_5-3B-lr-5e-6-minlr-0.0-bs-4-gbs-16-seqlen-4096-pr-bf16-tp-2-pp-2-cp-1-ac-none-do-true-sp-false-ti-10000-wi-"}
HF_CKPT_PATH=${HF_CKPT_PATH:-"/mnt/data/hf_models/Qwen2.5-3B-Instruct"}
PRECISION=${PRECISION:-"fp32"}
USE_TE=${USE_TE:-"true"}
MG2HF=${MG2HF:-"true"}

SOURCE_CKPT_PATH=${HF_MODELS}/Qwen2.5-${MODEL_SIZE}
TARGET_CKPT_PATH="./mg_models/Qwen2.5-${MODEL_SIZE}-hf-to-mcore-te-tp${TP}-pp${PP}"


if [ $MODEL_SIZE = 0.5B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=896
NUM_ATTN_HEADS=14
INTERMEDIATE_SIZE=4864
NUM_KEY_VALUE_HEADS=2
MAX_POSITION_EMBEDDINGS=32768
EXTRA_VOCAB_SIZE=293
RMS_NORM_EPS=1e-6
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"


tie_option=""
cpu_options=""

elif [ $MODEL_SIZE = 1.5B ]; then

NUM_LAYERS=28
HIDDEN_SIZE=1536
NUM_ATTN_HEADS=12
INTERMEDIATE_SIZE=8960
NUM_KEY_VALUE_HEADS=2
MAX_POSITION_EMBEDDINGS=32768
EXTRA_VOCAB_SIZE=293
RMS_NORM_EPS=1e-6
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"

tie_option=""
cpu_options=""

elif [ $MODEL_SIZE = 3B ]; then

NUM_LAYERS=36
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
INTERMEDIATE_SIZE=11008
NUM_KEY_VALUE_HEADS=2
MAX_POSITION_EMBEDDINGS=32768
EXTRA_VOCAB_SIZE=293
RMS_NORM_EPS=1e-6
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"

tie_option="--untie-embeddings-and-output-weights"
cpu_options=""

elif [ $MODEL_SIZE = 7B ]; then

NUM_LAYERS=28
HIDDEN_SIZE=3584
NUM_ATTN_HEADS=28
INTERMEDIATE_SIZE=18944
NUM_KEY_VALUE_HEADS=4
MAX_POSITION_EMBEDDINGS=131072
EXTRA_VOCAB_SIZE=421
RMS_NORM_EPS=1e-6
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"

tie_option=" \
        --untie-embeddings-and-output-weights \
        "

cpu_options=""

elif [ $MODEL_SIZE = 14B ]; then

NUM_LAYERS=48
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
INTERMEDIATE_SIZE=13824
NUM_KEY_VALUE_HEADS=8
MAX_POSITION_EMBEDDINGS=131072
EXTRA_VOCAB_SIZE=421
RMS_NORM_EPS=1e-5
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"

tie_option=" \
        --untie-embeddings-and-output-weights \
        "
cpu_options=""

elif [ $MODEL_SIZE = 32B ]; then

NUM_LAYERS=64
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
INTERMEDIATE_SIZE=27648
NUM_KEY_VALUE_HEADS=8
MAX_POSITION_EMBEDDINGS=131072
EXTRA_VOCAB_SIZE=421
RMS_NORM_EPS=1e-5
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"

tie_option=" \
        --untie-embeddings-and-output-weights \
        "

cpu_options=""

elif [ $MODEL_SIZE = 72B ]; then

NUM_LAYERS=80
HIDDEN_SIZE=8192
NUM_ATTN_HEADS=64
INTERMEDIATE_SIZE=29568
NUM_KEY_VALUE_HEADS=8
MAX_POSITION_EMBEDDINGS=131072
EXTRA_VOCAB_SIZE=421
RMS_NORM_EPS=1e-5
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"

tie_option=" \
        --untie-embeddings-and-output-weights \
        "

cpu_options=" \
            --use-cpu-initialization"

fi

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

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16"

elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"

fi


DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun ${DISTRIBUTED_ARGS} -m autoalign_megatron.toolkits.checkpoint.qwen.dpo \
    --load ${MG_MODEL_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP} \
    --micro-batch-size 1 \
    --save-interval 1 \
    --swiglu \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${INTERMEDIATE_SIZE} \
    --num-attention-heads ${NUM_ATTN_HEADS} \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --seq-length 1 \
    --no-async-tensor-model-parallel-allreduce \
    --patch-tokenizer-type Qwen2Tokenizer \
    --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
    --no-bias-swiglu-fusion \
    --no-rope-fusion \
    --use-rotary-position-embeddings \
    --disable-bias-linear \
    --add-qkv-bias \
    --normalization RMSNorm \
    --norm-epsilon ${RMS_NORM_EPS} \
    --use-mcore-models \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --rotary-base 1000000 \
    --save-safetensors \
    ${te_options} \
    ${convert_options} \
    ${pr_options} \
    ${cpu_options} \
    ${tie_option} \
    ${gqa_options}


ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"