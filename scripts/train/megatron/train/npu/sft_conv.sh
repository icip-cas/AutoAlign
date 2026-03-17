#!/bin/bash
# Megatron SFT training script for Ascend NPU
set -e

# Activate conda environment (required for torch-npu)
source /home/ma-user/miniconda3/bin/activate

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1

# ==============================
# Ascend / HCCL Environment
# ==============================
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

# HCCL_IF_BASE_PORT: base port for HCCL communication (default 64000)
# IMPORTANT: HCCL occupies 16 consecutive ports starting from this base port
# Change this if port conflict occurs with other training jobs
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-64000}
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}

# ==============================
# Path Configuration
# ==============================
DATASET_PATH=${DATASET_PATH:-"./data/dummy_sft.json"}
PRETRAIN_CHECKPOINT_PATH=${PRETRAIN_CHECKPOINT_PATH:-"./mg_models/Qwen2.5-3B-hf-to-mcore-tp2-pp2"}
OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"./checkpoints/sft"}
HF_MODEL_PATH=${HF_MODEL_PATH:-"Qwen/Qwen2.5-3B-Instruct"}
TEMPLATE=${TEMPLATE:-"chatml-idsys"}
# SwanLab: set SWANLAB_PROJECT / SWANLAB_EXP_NAME / SWANLAB_MODE etc. in env
REPORT_TO=${REPORT_TO:-""}

# ==============================
# Compute Resources Configuration
# ==============================
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}

MASTER_ADDR=${MASTER_ADDR:-"localhost"}
# MASTER_PORT: avoid HCCL port range [HCCL_IF_BASE_PORT, HCCL_IF_BASE_PORT+15]
MASTER_PORT=${MASTER_PORT:-$(shuf -n 1 -i 20000-29999)}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# ==============================
# Training Hyperparameters
# ==============================
BATCH_SIZE=${BATCH_SIZE:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
LR=${LR:-5e-6}
MIN_LR=${MIN_LR:-0.0}
SEQ_LEN=${SEQ_LEN:-4096}
PAD_LEN=${PAD_LEN:-4096}
EPOCHS=${EPOCHS:-1000}

# ==============================
# Parallelism Configuration
# ==============================
TP=${TP:-2}
PP=${PP:-2}
SP=${SP:-false}
CP=${CP:-1}

if [ "$SP" = true ] && [ "$TP" -gt 1 ]; then
    sp_options=" \
        --sequence-parallel"
else
    sp_options=""
fi

# ==============================
# Dataset Configuration
# Default: online JSON mode (no preprocessing needed)
# Set DATASET_MODE=mmap to use offline preprocessed data
# ==============================
DATASET_MODE=${DATASET_MODE:-"json"}

if [ "$DATASET_MODE" = "json" ]; then
    dataset_option=" \
        --data-path ${DATASET_PATH} \
        --split 100,0,0 \
        --dataset json \
        --template ${TEMPLATE} \
        --epochs ${EPOCHS}"
else
    dataset_option=" \
        --data-path ${DATASET_PATH} \
        --split 100,0,0 \
        --dataset mmap \
        --epochs ${EPOCHS}"
fi

# ==============================
# SFT Settings
# ==============================
SAVE_INTERVAL=${SAVE_INTERVAL:-10}
TRAIN_ITERS=${TRAIN_ITERS:-10000}
LR_WARMUP_FRACTION=$(echo "${GLOBAL_BATCH_SIZE} * 0.00001" | bc -l)
PREFIX="sft-npu-mcore-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}"
sft_option=""

# ==============================
# Precision Configuration: fp16, bf16
# ==============================
PR=${PR:-"bf16"}
if [ "$PR" = "fp16" ]; then
    pr_options=" \
        --fp16"
elif [ "$PR" = "bf16" ]; then
    pr_options=" \
        --bf16"
fi

# ==============================
# Activation Checkpointing: sel, full, none
# ==============================
AC=${AC:-"none"}
MP_AC_LAYERS=${MP_AC_LAYERS:-1}

if [ "$AC" = "full" ]; then
    activation_checkpoint_options=" \
        --recompute-method uniform \
        --recompute-num-layers ${MP_AC_LAYERS} \
        --recompute-granularity full"
elif [ "$AC" = "sel" ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
else
    activation_checkpoint_options=""
fi

# ==============================
# Distributed Optimizer Configuration
# ==============================
DO=${DO:-true}
if [ "$DO" = true ]; then
    do_options=" \
        --use-distributed-optimizer"
else
    do_options=""
fi

# ==============================
# NPU: use local transformer impl (no Transformer Engine)
# --use-flash-attn: MindSpeed routes to NPU flash attention
# ==============================
te_options=" \
    --transformer-impl local \
    --use-flash-attn \
    --no-bias-swiglu-fusion \
    --no-rope-fusion"

# ==============================
# Output Configuration
# ==============================
NAME="${PREFIX}-pr-${PR}-tp-${TP}-pp-${PP}-cp-${CP}-ac-${AC}-do-${DO}-sp-${SP}-ti-${TRAIN_ITERS}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}
SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

mkdir -p ${SAVED_PRETRAIN_CHECKPOINT_PATH}
find ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}
find ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "merge*" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH} 2>/dev/null || true

# ==============================
# Full Configuration
# ==============================
load_options=" \
    --load $PRETRAIN_CHECKPOINT_PATH"
megatron_options="  \
    --model-path ${HF_MODEL_PATH} \
    --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-decay-style cosine \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0 \
    --init-method-std 0.008 \
    --hidden-dropout 0.0 \
    --lr-warmup-fraction ${LR_WARMUP_FRACTION} \
    --micro-batch-size ${BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --seq-length ${SEQ_LEN} \
    --log-interval 1 \
    --log-throughput \
    --eval-interval 10000 \
    --eval-iters 10 \
    --save-interval ${SAVE_INTERVAL} \
    --tensorboard-queue-size 1 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-timers-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --num-workers 8 \
    --rotary-seq-len-interpolation-factor 1 \
    ${REPORT_TO:+--report-to $REPORT_TO} \
    "

# ==============================
# Train!
# ==============================
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
run_cmd="torchrun $DISTRIBUTED_ARGS -m autoalign.megatron.entries.sft
 ${megatron_options} ${dataset_option} ${pr_options} ${load_options} ${te_options} ${activation_checkpoint_options} \
 ${do_options} ${sp_options} ${sft_option}"

echo ${run_cmd}
eval ${run_cmd}
