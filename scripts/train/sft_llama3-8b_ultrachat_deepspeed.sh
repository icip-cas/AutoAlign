export DATA_PATH=/ciphome/wenxueru2022/auto-alignment/data/ultrachat.json
export CONV_TEMPLATE=llama-3-instruct
export OUTPUT_DIR=./saved_models/llama-3-8b_ultrachat
export MODEL_PATH=pretrained_models/Meta-Llama-3-8B
export GA=8
export DS_CONFIG=configs/zero2.json
export TRAIN_BATCH_SIZE=4
bash scripts/train_sft.sh
