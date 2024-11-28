#!/bin/bash

hosts=(
    "10.1.0.3"
    # "10.1.0.4"
    # "10.1.0.5"
    # "10.1.0.6"
    # "10.1.0.7"
    # "10.1.0.8"
    # "10.1.0.9"
    # "10.1.0.10"
    # "10.1.0.11"
    # "10.1.0.12"
)

function stop_all() {
    for host in "${hosts[@]}"; do
        ssh $host "pkill -f -9 dpo"
    done
    echo "所有节点的相关进程已停止。"
}

if [ "$1" == "stop" ]; then
    stop_all
    exit 0
fi

NNODES=${#hosts[@]}
MASTER_ADDR="10.1.0.3"
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
MODEL_SIZE="1.5B"
BATCH_SIZE=4
GLOBAL_BATCH_SIZE=64
SEQ_LEN=2048
LR=1e-5
MIN_LR=1e-6
TP=2
PP=2
EPOCHS=3
LR_WARMUP_ITERS=10
LOG_DIR="/share/zhangqingyu/logs/${MODEL_SIZE}_tp${TP}_pp${PP}_cp${CP}_mb${BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_nnode${NNODES}_$(date +%Y%m%d_%H%M%S)_profile"

# 创建日志目录
mkdir -p "$LOG_DIR"

NODE_RANK=0
for ((i=0; i<NNODES; i++)); do
    host=${hosts[i]}
    ssh $host "
        source /share/zhangqingyu/scripts/env.sh 
        cd /share/zhangqingyu/code/auto-alignment/algorithms/megatron_dpo/
        bash examples/qwen2/dpo_${MODEL_SIZE}.sh ${NNODES} ${NODE_RANK} ${MASTER_ADDR} ${MASTER_PORT} ${MODEL_SIZE} ${BATCH_SIZE} ${GLOBAL_BATCH_SIZE} ${SEQ_LEN} ${LR} ${MIN_LR} ${TP} ${PP} ${EPOCHS} ${LR_WARMUP_ITERS} > ${LOG_DIR}/node_${NODE_RANK}.log 2>&1 &
    "
    NODE_RANK=$((NODE_RANK+1))
done

echo "所有节点的训练脚本已启动。日志文件保存在 ${LOG_DIR} 目录中。"
