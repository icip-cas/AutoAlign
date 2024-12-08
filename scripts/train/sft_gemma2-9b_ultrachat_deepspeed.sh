export DATA_PATH=/ciphome/wenxueru2022/auto-alignment/data/ultrachat.json
export CONV_TEMPLATE=gemma
export OUTPUT_DIR=./saved_models/gemma-2-9b_ultrachat/
export MODEL_PATH=/data7/pretrained_models/gemma-2-9b/
export GA=16
export DS_CONFIG=configs/zero2.json
export TRAIN_BATCH_SIZE=2
bash scripts/train_sft.sh
