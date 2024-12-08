export DATA_PATH=/ciphome/wenxueru2022/auto-alignment/data/ultrachat.json
export CONV_TEMPLATE=chatml-keep-system
export OUTPUT_DIR=./saved_models/qwen_14b_ultrachat
export MODEL_PATH=pretrained_models/Qwen1.5-14B
export GA=8
export DS_CONFIG=configs/zero2.json
export TRAIN_BATCH_SIZE=2
bash scripts/train_sft.sh
