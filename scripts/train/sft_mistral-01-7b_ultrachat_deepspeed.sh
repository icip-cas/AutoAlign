export DATA_PATH=/ciphome/wenxueru2022/auto-alignment/data/ultrachat.json
export CONV_TEMPLATE=zephyr
export OUTPUT_DIR=./saved_models/mistral-01-7b_ultrachat
export MODEL_PATH=./pretrained_models/Mistral-7B-v0.1
export GA=8
export LR=2e-5
export WARMUP_RATIO=0.1
export DS_CONFIG=configs/zero2.json
export TRAIN_BATCH_SIZE=8
bash scripts/train_sft.sh
