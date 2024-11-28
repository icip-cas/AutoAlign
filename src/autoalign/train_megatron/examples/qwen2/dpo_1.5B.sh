#!/bin/bash
set -e
# ==============================
# 路径设置
# ==============================
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
cd ${CURRENT_DIR}
export PYTHONPATH=${CURRENT_DIR}:${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM-240718:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET_PATH=/ciphome/zhangqingyu2023/data/dpo/ultrafeedback_binarized_conversations_maxlen_2048
VALID_DATASET_PATH=/ciphome/zhangqingyu2023/data/dpo/ultrafeedback_binarized_conversations_maxlen_2048
PRETRAIN_CHECKPOINT_PATH=/ciphome/zhangqingyu2023/mg_models/Qwen2-1.5B-hf-to-mcore-te-tp2-pp2
OUTPUT_BASEPATH=/ciphome/zhangqingyu2023/checkpoint/dpo/output_mcore_qwen2_1point5_ct_tp2_pp2


# ==============================
# 算力资源配置
# ==============================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NNODES=$1
NODE_RANK=$2
MASTER_ADDR=$3
MASTER_PORT=$4
GPUS_PER_NODE=8


# ==============================
# 训练超参数设置
# ==============================
MODEL_SIZE=$5
BATCH_SIZE=$6
GLOBAL_BATCH_SIZE=$7
SEQ_LEN=$8
PAD_LEN=$8
LR=$9
MIN_LR=${10}


# ==============================
# 并行设置
# ==============================
TP=${11}
PP=${12}
# 序列并行目前有bug，暂时不使用
SP=false
CP=1
if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
		    --sequence-parallel \
            --tp-comm-overlap "

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

# ==============================
# 数据集配置
# ==============================
EPOCHS=${13}
dataset_option=" \
    --data-path ${DATASET_PATH} \
    --split 100,0,0 \
    --dataset mmap  \
    --epochs ${EPOCHS} "


# ==============================
# DPO
# ==============================
SFT=True
# the following two values will not be used when SFT is true
SAVE_INTERVAL=100000
TRAIN_ITERS=10000
LR_WARMUP_ITERS=${14}
PREFIX="dpo-mcore-qwen2-1point5b-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}"
sft_option=" \
        --eod-mask-loss \
        --train-mode dpo"


# ==============================
# 模型架构配置
# ==============================
NUM_LAYERS=28
HIDDEN_SIZE=1536
NUM_ATTN_HEADS=12
INTERMEDIATE_SIZE=8960
NUM_KEY_VALUE_HEADS=2
MAX_POSITION_EMBEDDINGS=131072
EXTRA_VOCAB_SIZE=293
RMS_NORM_EPS=1e-6
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"

te_options=" \
        --transformer-impl transformer_engine"

load_options=" \
        --load $PRETRAIN_CHECKPOINT_PATH"

# ==============================
# FlashAttention Or FusedAttention
# ==============================
FL=true
if [ $FL = true ]; then
    export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0
elif [ $FL = false ]; then
    export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1
fi

# ==============================
# 精度配置: fp16, bf16, fp8
# ==============================
PR=bf16
if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16 \
            --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16 \
        --fp8-format hybrid \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024"
fi


# ==============================
# 激活检查点模式: sel, full, offload, false
# ==============================
MP_AC_LAYERS=1
AC=false
if [ $AC = full ]; then
    _check=$(( ($NUM_LAYERS / $PP) % ${MP_AC_LAYERS} ))
    if [ $_check != 0 ]; then
        echo "the num layers per pp rank must be a multiple of the recompute layers."
        exit -1
    fi
    activation_checkpoint_options=" \
        --recompute-method uniform \
        --recompute-num-layers ${MP_AC_LAYERS} \
        --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
    "
elif [ $AC = offload ]; then
    activation_checkpoint_options=" \
        --cpu-offloading \
        --cpu-offloading-num-layers ${MP_AC_LAYERS}"
    if [ $TP_COMM_OVERLAP -eq 1 ]; then
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option="\
            --tp-comm-overlap"
    else
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option=""
    fi
fi

# ==============================
# 优化TP通信
# ==============================
TP_COMM_OVERLAP=$(( ($TP > 1) ? 1 : 0 ))
comm_overlap_option="\
    --overlap-grad-reduce \
    --overlap-param-gather"
 

if [ $TP_COMM_OVERLAP -eq 1 ]; then
    comm_overlap_option="\
        --overlap-grad-reduce \
        --overlap-param-gather"
fi


# ==============================
# 分布式优化器配置
# ==============================
DO=true
OPTIMIZER_OFFLOAD=false
if [ $OPTIMIZER_OFFLOAD != false ] && [ $DO = false ]; then
    echo "Offload optimizer is valid only if \$DO=true"
    DO=true
fi

if [ $DO = true ]; then
    do_options=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi

if [ $OPTIMIZER_OFFLOAD = 'static' ]; then
    offload_option=" \
        --optimizer hybridadam \
        --optimizer-offload-policy static \
        --optimizer-offload-fraction 1.0"
elif [ $OPTIMIZER_OFFLOAD = 'auto' ]; then
    offload_option=" \
        --optimizer hybridadam \
        --optimizer-offload-policy auto"
else
    offload_option=""
fi



# ==============================
# 创建checkpoint目录
# ==============================
NAME="${PREFIX}-pr-${PR}-tp-${TP}-pp-${PP}-cp-${CP}-ac-${AC}-do-${DO}-sp-${SP}-ti-${TRAIN_ITERS}-wi-${LR_WARMUP_ITERS}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}
SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

mkdir -p ${SAVED_PRETRAIN_CHECKPOINT_PATH}
find ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}

# ==============================
# 模型完整配置
# ==============================
megatron_options="  \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --init-method-std 0.008 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
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
        --no-load-optim \
        --no-load-rng \
        --num-workers 8 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type Qwen2Tokenizer \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon ${RMS_NORM_EPS} \
        --use-rotary-position-embeddings \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --add-qkv-bias \
        --rotary-percent 1.0 \
        --rotary-base 1000000 \
        --rotary-seq-len-interpolation-factor 1 \
        --no-save-optim \
        "
# ==============================
# Tranin!
# ==============================
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
run_cmd="torchrun $DISTRIBUTED_ARGS dpo_qwen.py
 ${megatron_options} ${dataset_option} ${pr_options} ${load_options} ${te_options} ${activation_checkpoint_options} \
 ${do_options} ${sp_options} ${gqa_options} ${offload_option} ${comm_overlap_option} ${sft_option} ${tie_option}"

echo ${run_cmd}
eval ${run_cmd}
set +x
