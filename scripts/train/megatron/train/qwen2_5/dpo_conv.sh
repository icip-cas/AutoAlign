#!/bin/bash
set -e

# Activate conda environment (required for torch-npu)
source /home/ma-user/miniconda3/bin/activate

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ==============================
# NPU/HCCL Configuration (Ascend only)
# ==============================
# HCCL_IF_BASE_PORT: base port for HCCL communication (default 64000)
# IMPORTANT: HCCL occupies 16 consecutive ports starting from this base port
# e.g., HCCL_IF_BASE_PORT=64000 uses ports 64000-64015
# Change this if port conflict occurs with other training jobs
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-64000}
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}

# ==============================
# Path Configuration
# ==============================
DATASET_PATH=${DATASET_PATH:-"./data/dummy_dpo_mg_conversations_maxlen_4096"}
VALID_DATASET_PATH=${VALID_DATASET_PATH:-"./data/dummy_dpo_mg_conversations_maxlen_4096"}
PRETRAIN_CHECKPOINT_PATH=${PRETRAIN_CHECKPOINT_PATH:-"./mg_models/Qwen2.5-3B-hf-to-mcore-te-tp2-pp2"}
OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"./checkpoints/dpo"}
# HF model path for auto-deriving model architecture args
HF_MODEL_PATH=${HF_MODEL_PATH:-"Qwen/Qwen2.5-3B-Instruct"}

# ==============================
# Compute Resources Configuration
# ==============================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

MASTER_ADDR=${MASTER_ADDR:-"localhost"}
# MASTER_PORT: avoid HCCL port range [HCCL_IF_BASE_PORT, HCCL_IF_BASE_PORT+15]
# Default HCCL range: 64000-64015, so use 20000-29999 for MASTER_PORT
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
        --sequence-parallel \
        --tp-comm-overlap "
elif [ "$SP" = false ]; then
    sp_options=""
fi

# ==============================
# Dataset Configuration
# ==============================
dataset_option=" \
    --data-path ${DATASET_PATH} \
    --split 100,0,0 \
    --dataset mmap  \
    --epochs ${EPOCHS}"


# ==============================
# DPO Settings
# ==============================
DPO=${DPO:-True}
SAVE_INTERVAL=${SAVE_INTERVAL:-10}

TRAIN_ITERS=${TRAIN_ITERS:-10000}
LR_WARMUP_FRACTION=$(echo "${GLOBAL_BATCH_SIZE} * 0.00001" | bc -l)
PREFIX="dpo-mcore-qwen2_5-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}"
dpo_option=" \
    --eod-mask-loss \
    --train-mode dpo"

# ==============================
# FlashAttention Or FusedAttention
# ==============================
FL=${FL:-true}
if [ "$FL" = true ]; then
    export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0
elif [ "$FL" = false ]; then
    export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1
fi

# ==============================
# Precision Configuration: fp16, bf16, fp8
# ==============================
PR=${PR:-"bf16"}
if [ "$PR" = "fp16" ]; then
    pr_options=" \
        --fp16 \
        --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ "$PR" = "bf16" ]; then
    pr_options=" \
        --bf16"
elif [ "$PR" = "fp8" ]; then
    pr_options=" \
        --bf16 \
        --fp8-format hybrid \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024"
fi


# ==============================
# Activation Checkpointing Mode: sel, full, offload, none
# ==============================
AC=${AC:-"none"}
MP_AC_LAYERS=${MP_AC_LAYERS:-1}

if [ "$AC" = "full" ]; then
    _check=$(( (NUM_LAYERS / PP) % MP_AC_LAYERS ))
    if [ "$_check" -ne 0 ]; then
        echo "The number of layers per PP rank must be a multiple of the recompute layers."
        exit 1
    fi
    activation_checkpoint_options=" \
        --recompute-method uniform \
        --recompute-num-layers ${MP_AC_LAYERS} \
        --recompute-granularity full"
elif [ "$AC" = "sel" ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ "$AC" = "offload" ]; then
    activation_checkpoint_options=" \
        --cpu-offloading \
        --cpu-offloading-num-layers ${MP_AC_LAYERS}"
    if [ "$TP_COMM_OVERLAP" -eq 1 ]; then
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when CPU offloading is on..."
        comm_overlap_option="\
            --tp-comm-overlap"
    else
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when CPU offloading is on..."
        comm_overlap_option=""
    fi
else
    activation_checkpoint_options=""
fi

# ==============================
# Optimize TP Communication
# ==============================
TP_COMM_OVERLAP=$(( TP > 1 ? 1 : 0 ))
comm_overlap_option="--overlap-grad-reduce"

if [ "$TP_COMM_OVERLAP" -eq 1 ]; then
    comm_overlap_option="--overlap-grad-reduce"
fi


# ==============================
# Distributed Optimizer Configuration
# ==============================
DO=${DO:-true}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-false}

if [ "$OPTIMIZER_OFFLOAD" != false ] && [ "$DO" = false ]; then
    echo "Offload optimizer is valid only if \$DO=true"
    DO=true
fi

if [ "$DO" = true ]; then
    do_options=" \
        --use-distributed-optimizer \
        --overlap-grad-reduce"
else
    do_options=""
fi

if [ "$OPTIMIZER_OFFLOAD" = "static" ]; then
    offload_option=" \
        --optimizer hybridadam \
        --optimizer-offload-policy static \
        --optimizer-offload-fraction 1.0"
elif [ "$OPTIMIZER_OFFLOAD" = "auto" ]; then
    offload_option=" \
        --optimizer hybridadam \
        --optimizer-offload-policy auto"
else
    offload_option=""
fi

# ==============================
# Model Architecture (auto-derived from --model-path)
# ==============================

te_options=" \
        --transformer-impl transformer_engine"


# ==============================
# Output Configuration
# ==============================
NAME="${PREFIX}-pr-${PR}-tp-${TP}-pp-${PP}-cp-${CP}-ac-${AC}-do-${DO}-sp-${SP}-ti-${TRAIN_ITERS}-wf-${LR_WARMUP_FRACTION}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}
SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

mkdir -p ${SAVED_PRETRAIN_CHECKPOINT_PATH}
find ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}
find ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "merge*" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}
# ==============================
# 模型完整配置
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
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-padding-length ${PAD_LEN} \
        --log-interval 1 \
        --log-throughput \
        --eval-interval 10000 \
        --eval-iters 10 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        --num-workers 8 \
        --rotary-seq-len-interpolation-factor 1 \
        "
        # --no-save-optim \
        # --no-load-optim \
        # --no-load-rng \

# ==============================
# Train!
# ==============================
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
run_cmd="torchrun $DISTRIBUTED_ARGS -m autoalign.megatron.entries.dpo
 ${megatron_options} ${dataset_option} ${pr_options} ${load_options} ${te_options} ${activation_checkpoint_options} \
 ${do_options} ${sp_options} ${offload_option} ${comm_overlap_option} ${dpo_option}"

echo ${run_cmd}
eval ${run_cmd}
set +x
