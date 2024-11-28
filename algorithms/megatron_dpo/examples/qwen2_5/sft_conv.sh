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
export OMP_NUM_THREADS=1
cd ../qwen2
 
DATASET_PATH=/ciphome/zhangqingyu2023/data/sft/InfInstruct-Gen_infinite_9m_conversations_maxlen_4096
VALID_DATASET_PATH=/ciphome/zhangqingyu2023/data/sft/InfInstruct-Gen_infinite_9m_conversations_maxlen_4096
PRETRAIN_CHECKPOINT_PATH=/ciphome/zhangqingyu2023/mg_models/Qwen2.5-7B-hf-to-mcore-te-tp2-pp1
OUTPUT_BASEPATH=/ciphome/zhangqingyu2023/checkpoint/sft/Qwen2.5-7B-hf-to-mcore-te-tp2-pp1/debug


# ==============================
# 算力资源配置
# ==============================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# MASTER_ADDR=12.12.11.14
MASTER_ADDR=localhost
# MASTER_PORT=$(shuf -n 1 -i 10000-65535)
MASTER_PORT=29500
NNODES=1
NODE_RANK=$1
GPUS_PER_NODE=8

# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_DISABLE=1

# export NCCL_IB_DISABLE=0
# export NCCL_IB_HCA=mlx5_0
# unset NCCL_SOCKET_IFNAME

# ==============================
# 训练超参数设置
# ==============================
MODEL_SIZE=7B
BATCH_SIZE=4
GLOBAL_BATCH_SIZE=512
LR=5e-6
MIN_LR=0.0
SEQ_LEN=4096
PAD_LEN=4096


# ==============================
# 并行设置
# ==============================
TP=2
PP=1
SP=true
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
dataset_option=" \
    --data-path ${DATASET_PATH} \
    --split 100,0,0 \
    --dataset mmap  \
    --epochs 3"

# --shuffle-all-epochs \
# ==============================
# SFT
# ==============================
SFT=True
SAVE_INTERVAL=5000

TRAIN_ITERS=10000
LR_WARMUP_FRACTION=$(echo "${GLOBAL_BATCH_SIZE} * 0.00001" | bc -l)
PREFIX="sft-mcore-qwen2_5-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}"
sft_option=" \
        --eod-mask-loss \
        --train-mode sft"


# ==============================
# 模型架构配置
# ==============================
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

tie_option=""

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
fi

te_options=" \
        --transformer-impl transformer_engine"


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
AC=none
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
    --overlap-grad-reduce"

if [ $TP_COMM_OVERLAP -eq 1 ]; then
    comm_overlap_option="\
        --overlap-grad-reduce "
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
		    --use-distributed-optimizer \
            --overlap-param-gather"

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
find ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "merge*" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}
# ==============================
# 模型完整配置
# ==============================
load_options=" \
        --load $PRETRAIN_CHECKPOINT_PATH"
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
        --lr-warmup-fraction ${LR_WARMUP_FRACTION} \
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
        --num-workers 8 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type Qwen2Tokenizer \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon ${RMS_NORM_EPS} \
        --use-rotary-position-embeddings \
        --position-embedding-type rope \
        --disable-bias-linear \
        --add-qkv-bias \
        --rotary-percent 1.0 \
        --rotary-base 1000000 \
        --rotary-seq-len-interpolation-factor 1 \
        "
# --no-save-optim \
# --no-load-optim \
# --no-load-rng \

# ==============================
# Tranin!
# ==============================
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
run_cmd="torchrun $DISTRIBUTED_ARGS sft_qwen.py
 ${megatron_options} ${dataset_option} ${pr_options} ${load_options} ${te_options} ${activation_checkpoint_options} \
 ${do_options} ${sp_options} ${gqa_options} ${offload_option} ${comm_overlap_option} ${sft_option} ${tie_option}"

echo ${run_cmd}
eval ${run_cmd}
set +x
